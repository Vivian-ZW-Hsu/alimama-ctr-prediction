"""Execute the preregistered validation-only study and freeze checkpoint hashes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import polars as pl
import torch

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER, tabular
from f4_sequences import sha256, write_json
from sequence_ctr import SequenceResidual

STUDY = ROOT/'results/sequence_selection_v1'


def select_common(rows):
    """Pair the mechanisms; do not independently optimize either family."""
    candidates = []
    for rate in (5e-4, 2e-4):
        pair = [r for r in rows if r['learning_rate'] == rate]
        assert len(pair) == 2 and {r['pooling'] for r in pair} == {'mean','attention'}
        candidates.append({'learning_rate':rate,
                           'paired_validation_logloss':float(np.mean([r['log_loss'] for r in pair]))})
    selected = min(candidates,key=lambda r:(r['paired_validation_logloss'],-r['learning_rate']))
    return selected['learning_rate'], candidates


def run_name(pooling, source, rate, seed):
    return f'seq_{pooling}_{source}_l20_lr{int(rate*1e6)}_s{seed}_v1'


def verify_completed(name):
    directory = ROOT/'results'/name
    manifest = json.loads((directory/'manifest.json').read_text())
    assert manifest['complete'] and not manifest['smoke'] and not manifest['pilot']
    assert manifest['prediction_splits'] == ['val'] and not manifest['test_inference']
    assert manifest['train_rows'] == 20_015_245 and manifest['val_rows'] == 3_234_051
    assert sha256(directory/'config.json') == manifest['config_sha256']
    for path,digest in manifest['output_sha256'].items():assert sha256(ROOT/path) == digest
    config = json.loads((directory/'config.json').read_text())
    for path,digest in config['code_sha256'].items():assert sha256(ROOT/'src'/path) == digest
    return config, manifest


def command(args, log_path, remaining):
    assert remaining > 0, 'Preregistered compute budget exhausted'
    with log_path.open('x') as log:
        subprocess.run([sys.executable,'-u','-B',*args],cwd=ROOT,stdout=log,
                       stderr=subprocess.STDOUT,check=True,timeout=remaining)


def main():
    prepath = STUDY/'preregistration.json'
    pre = json.loads(prepath.read_text())
    assert pre['test_authorized'] is False and pre['max_length'] == 20
    if (STUDY/'freeze_manifest.json').exists():
        raise FileExistsError('Study already frozen; no additional tuning is allowed')
    logs = STUDY/'logs'; logs.mkdir(exist_ok=True)
    start = time.perf_counter()
    completed_before = sum(json.loads(p.read_text())['elapsed_seconds']
                           for p in (ROOT/'results').glob('seq_*_v1/manifest.json'))
    def remaining():return pre['budget_seconds']-completed_before-(time.perf_counter()-start)
    def ensure(pooling,source,rate,seed):
        name = run_name(pooling,source,rate,seed)
        if not (ROOT/'results'/name/'manifest.json').exists():
            print(f'Training {name}',flush=True)
            command(['src/22_train_sequence_ctr.py','--cache-run',pre['cache_run'],
                     '--run-name',name,'--pooling',pooling,'--sequence-source',source,
                     '--learning-rate',str(rate),'--seed',str(seed),'--epochs',str(pre['max_epochs']),
                     '--batch-size',str(pre['batch_size'])], logs/f'{name}_train.log',remaining())
        cfg,manifest = verify_completed(name)
        assert cfg['pooling']==pooling and cfg['sequence_source']==source
        assert cfg['learning_rate']==rate and cfg['seed']==seed
        assert cfg['cache_run']==pre['cache_run'] and cfg['batch_size']==pre['batch_size']
        metrics = json.loads((ROOT/'results'/name/'best_validation.json').read_text())
        row = {'run_name':name,'pooling':pooling,'source':source,'learning_rate':rate,
               'seed':seed,'best_epoch':manifest['best_epoch'],**{k:v for k,v in metrics.items() if k!='model'}}
        print(json.dumps(row),flush=True)
        return row
    search = [ensure(pooling,'matched',rate,pre['selection_seed'])
              for rate in pre['learning_rates'] for pooling in pre['poolings']]
    selected_rate, pairs = select_common(search)
    selection = {'preregistration_sha256':sha256(prepath),'criterion':'paired validation LogLoss',
                 'selected_learning_rate':selected_rate,'max_length':20,'candidates':pairs,'runs':search}
    selection_path = STUDY/'selection.json'
    if selection_path.exists():assert json.loads(selection_path.read_text())==selection
    else:write_json(selection_path,selection)
    print(f'Selected common LR={selected_rate}; controls inherit it',flush=True)
    final = [ensure(pooling,source,selected_rate,seed) for seed in pre['final_seeds']
             for pooling in pre['poolings'] for source in pre['sources']]
    # Shared evaluation only touches validation. Test predictions do not exist.
    all_runs = {row['run_name']:row for row in [*search,*final]}
    for name,row in all_runs.items():
        evaluation = ROOT/'results'/name/'evaluation'
        if not (evaluation/'evaluation_manifest.json').exists():
            command(['src/10_evaluate_predictions.py','--run-name',name,'--splits','val'],
                    logs/f'{name}_evaluate.log',remaining())
        assert json.loads((evaluation/'evaluation_manifest.json').read_text())['splits']==['val']
        overall = pl.read_csv(evaluation/'overall_metrics.csv').row(0,named=True)
        gauc = pl.read_csv(evaluation/'gauc_summary.csv').row(0,named=True)
        assert abs(overall['logloss']-row['log_loss']) < 1e-12
        row.update({k:overall[k] for k in ('copc','ece')})
        row.update({k:gauc[k] for k in ('gauc','eligible_users','eligible_row_coverage')})
    # Reload selected checkpoints and regenerate saved validation probabilities.
    spec = importlib.util.spec_from_file_location('sequence_trainer',ROOT/'src/22_train_sequence_ctr.py')
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    cache_path = ROOT/'results'/pre['cache_run']/'manifest.json'
    cache = json.loads(cache_path.read_text())
    f4path = ROOT/cache['f4_manifest']; f4 = json.loads(f4path.read_text())
    assert sha256(BASE_MODEL)==cache['base_model_sha256'] and sha256(BASE_SCALER)==cache['scaler_sha256']
    data = trainer.load_columns(ROOT/cache['directory']/'val.parquet')
    categories = pl.read_parquet(ROOT/f4['output_directory']/'events.parquet',columns=['cate_idx'])['cate_idx'].to_numpy()
    embedding = torch.from_numpy(np.load(ROOT/cache['directory']/'category_embedding.npy'))
    device = tabular.choose_device('auto')
    frozen = []
    for row in final:
        name = row['run_name']; cfg,manifest = verify_completed(name)
        checkpoint_path = ROOT/'artifacts/models'/name/'best_model.pt'
        checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert checkpoint['base_model_sha256']==cache['base_model_sha256']
        model = SequenceResidual(embedding,row['pooling']).to(device)
        model.load_state_dict(checkpoint['model_state_dict']); model.assert_neutral()
        pred = trainer.predict(model,data,categories,row['source'],device,pre['batch_size']*2)
        saved = pl.read_parquet(ROOT/'artifacts/predictions'/name/'val_predictions.parquet')
        np.testing.assert_array_equal(saved['impression_id'].to_numpy(),data['impression_id'])
        np.testing.assert_array_equal(saved['clk'].to_numpy(),data['clk'])
        np.testing.assert_array_equal(saved['prediction'].to_numpy(),pred)
        assert not (ROOT/'artifacts/predictions'/name/'test_predictions.parquet').exists()
        frozen.append({'run_name':name,'seed':row['seed'],'pooling':row['pooling'],'source':row['source'],
                       'best_epoch':manifest['best_epoch'],'checkpoint_sha256':sha256(checkpoint_path),
                       'validation_prediction_sha256':manifest['output_sha256'][f'artifacts/predictions/{name}/val_predictions.parquet'],
                       'reload_prediction_max_error':0.0})
    # Enrich final rows from the common evaluator, including search-run reuse.
    final = [all_runs[row['run_name']] for row in final]
    pl.DataFrame(final).write_csv(STUDY/'validation_metrics.csv')
    pl.DataFrame(list(all_runs.values())).write_csv(STUDY/'all_candidates.csv')
    summary = pl.DataFrame(final).group_by('pooling','source').agg(
        pl.len().alias('seeds'),*[expr for metric in ('auc','log_loss','pr_auc','gauc','copc','ece')
        for expr in (pl.col(metric).mean().alias(metric+'_mean'),pl.col(metric).std().alias(metric+'_std'))]
    ).sort('pooling','source')
    summary.write_csv(STUDY/'validation_seed_summary.csv')
    write_json(STUDY/'freeze_manifest.json',{
        'complete':True,'test_authorized':False,'test_inference_performed':False,
        'preregistration_sha256':sha256(prepath),'selection_sha256':sha256(selection_path),
        'cache_manifest_sha256':sha256(cache_path),'f4_manifest_sha256':sha256(f4path),
        'base_model_sha256':cache['base_model_sha256'],'base_scaler_sha256':cache['scaler_sha256'],
        'selected_learning_rate':selected_rate,'max_length':20,'seed_list':pre['final_seeds'],
        'configuration':pre,'checkpoints':frozen,'unique_training_runs':len(all_runs),
        'code_sha256':{p.name:sha256(p) for p in [ROOT/'src/sequence_ctr.py',ROOT/'src/ctr_nn_common.py',
                      ROOT/'src/22_train_sequence_ctr.py',Path(__file__),ROOT/'src/10_evaluate_predictions.py']},
        'execution_seconds_this_invocation':time.perf_counter()-start,
        'next_gate':'User approval required before the first formal sequence test. No further tuning.'})
    print(summary,flush=True)
    print('FROZEN. Stopped before formal test.',flush=True)


if __name__=='__main__':main()
