"""Validate the user-requested zero control, then assemble an immutable v2 freeze."""
import argparse
import importlib.util
import json
import subprocess
import sys

import numpy as np
import polars as pl
import torch

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER, tabular
from f4_sequences import sha256, write_json
from sequence_ctr import SequenceResidual
from zero_sequence_ctr import ZeroSequenceResidual

STUDY=ROOT/'results/sequence_selection_v2'
spec=importlib.util.spec_from_file_location('trainer',ROOT/'src/22_train_sequence_ctr.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def reload_check(name, cache_run, zero=False):
    cache=json.loads((ROOT/'results'/cache_run/'manifest.json').read_text())
    directory=ROOT/cache['directory']
    data=trainer.load_columns(directory/'val.parquet')
    f4=json.loads((ROOT/cache['f4_manifest']).read_text())
    categories=pl.read_parquet(ROOT/f4['output_directory']/'events.parquet',columns=['cate_idx'])['cate_idx'].to_numpy()
    ckpath=ROOT/'artifacts/models'/name/'best_model.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    device=tabular.choose_device('auto')
    cls=ZeroSequenceResidual if zero else SequenceResidual
    model=cls(torch.from_numpy(np.load(directory/'category_embedding.npy')),ck['config']['pooling']).to(device)
    model.load_state_dict(ck['model_state_dict']);model.assert_neutral()
    p=trainer.predict(model,data,categories,ck['config']['sequence_source'],device,8192)
    saved=pl.read_parquet(ROOT/'artifacts/predictions'/name/'val_predictions.parquet')
    for key in ('impression_id','clk'):np.testing.assert_array_equal(saved[key].to_numpy(),data[key])
    np.testing.assert_array_equal(p,saved['prediction'].to_numpy())
    if zero:
        with torch.no_grad():base=torch.sigmoid(torch.from_numpy(data['base_logit'].copy()).to(device)).cpu().numpy()
        np.testing.assert_array_equal(p,base)
        if not cache['smoke']:
            old=pl.read_parquet(ROOT/'artifacts/predictions/deepfm_f1_f2_f3/val_predictions.parquet').sort('impression_id')
            np.testing.assert_array_equal(saved.sort('impression_id')['prediction'].to_numpy(),old['prediction'].to_numpy())
    return {'run_name':name,'checkpoint_sha256':sha256(ckpath),
            'validation_prediction_sha256':sha256(ROOT/'artifacts/predictions'/name/'val_predictions.parquet'),
            'reload_max_error':0.0,'zero_baseline_max_error':0.0 if zero else None,'epoch':ck['epoch']}


def zero_validation():
    assert not (STUDY/'zero_validation_audit.json').exists()
    smoke=reload_check('seq_zero_v2_smoke','sequence_context_v1_smoke',True)
    logs=STUDY/'logs';logs.mkdir(exist_ok=True)
    records=[]
    for seed in (42,2026,3407):
        name=f'seq_zero_same_pipeline_l20_s{seed}_v2'
        result=ROOT/'results'/name
        if not (result/'manifest.json').exists():
            with (logs/f'{name}_train.log').open('x') as log:
                subprocess.run([sys.executable,'-u','-B','src/22_train_sequence_ctr.py','--cache-run','sequence_context_v1',
                    '--run-name',name,'--pooling','mean','--sequence-source','matched','--zero-sequence',
                    '--learning-rate','0.0005','--batch-size','4096','--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        cfg=json.loads((result/'config.json').read_text())
        assert cfg['zero_sequence'] and cfg['seed']==seed and cfg['batch_size']==4096 and cfg['learning_rate']==5e-4
        record=reload_check(name,'sequence_context_v1',True)
        record.update(seed=seed,pooling='mean',source='matched',condition='zero_sequence_same_pipeline',zero_sequence=True)
        if not (result/'evaluation/evaluation_manifest.json').exists():
            with (logs/f'{name}_eval.log').open('x') as log:
                subprocess.run([sys.executable,'-B','src/10_evaluate_predictions.py','--run-name',name,'--splits','val'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        records.append(record);print(record,flush=True)
    assert len({r['validation_prediction_sha256'] for r in records})==1
    # Regression: changed training entry point must reproduce all original frozen predictions.
    original=json.loads((ROOT/'results/sequence_selection_v1/freeze_manifest.json').read_text())
    regressions=[]
    for ck in original['checkpoints']:
        r=reload_check(ck['run_name'],'sequence_context_v1')
        assert r['checkpoint_sha256']==ck['checkpoint_sha256'] and r['validation_prediction_sha256']==ck['validation_prediction_sha256']
        regressions.append(r)
    write_json(STUDY/'zero_validation_audit.json',{'passed':True,'smoke_reload':smoke,'zero_conditions':records,
        'original_checkpoint_regression':regressions,'all_zero_seed_predictions_identical':True,
        'purpose':'Pipeline null control; batch/seed/interface parity with strictly zero whole residual, conditional on one frozen backbone'})


def freeze():
    path=STUDY/'freeze_manifest.json'
    assert not path.exists()
    original_path=ROOT/'results/sequence_selection_v1/freeze_manifest.json'
    original=json.loads(original_path.read_text())
    audit=json.loads((STUDY/'zero_validation_audit.json').read_text());assert audit['passed']
    records=[]
    for ck in original['checkpoints']:
        records.append({**ck,'condition':ck['pooling']+'_'+ck['source'],'zero_sequence':False})
    records.extend(audit['zero_conditions'])
    for ck in records:
        ck['best_epoch']=ck.get('best_epoch',ck.get('epoch'))
        ck['test_run_name']=ck['run_name']+'_test_v2'
        assert sha256(ROOT/'artifacts/models'/ck['run_name']/'best_model.pt')==ck['checkpoint_sha256']
    assert sha256(BASE_MODEL)==original['base_model_sha256'] and sha256(BASE_SCALER)==original['base_scaler_sha256']
    assert len(records)==15 and all(r['best_epoch']==1 for r in records)
    code=['16_train_tabular_nn.py','10_evaluate_predictions.py','22_train_sequence_ctr.py','sequence_ctr.py',
          'zero_sequence_ctr.py','ctr_nn_common.py','25_prepare_sequence_test.py','26_infer_sequence_test.py','27_report_sequence_test.py']
    write_json(path,{'version':2,'complete':True,'supersedes':str(original_path.relative_to(ROOT)),
        'original_freeze_sha256':sha256(original_path),'original_freeze_preserved':True,
        'comparison_plan':json.loads((STUDY/'comparison_plan.json').read_text()),
        'comparison_plan_sha256':sha256(STUDY/'comparison_plan.json'),
        'zero_validation_audit_sha256':sha256(STUDY/'zero_validation_audit.json'),
        'base_model_sha256':original['base_model_sha256'],'base_scaler_sha256':original['base_scaler_sha256'],
        'f4_manifest_sha256':original['f4_manifest_sha256'],'cache_manifest_sha256':original['cache_manifest_sha256'],
        'checkpoints':records,'code_sha256':{p:sha256(ROOT/'src'/p) for p in code},
        'test_authorized_conditionally':True,'test_inference_performed_at_freeze':False,
        'gate':'Before test require passing inference smoke certificate and clean pre-test commit pushed to origin/main'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    freeze() if args.freeze else zero_validation()
