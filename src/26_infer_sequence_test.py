"""One immutable streaming inference pass; smoke uses validation exclusively."""
import argparse
from datetime import datetime, timezone
import json
import resource
import subprocess
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER, tabular, load_frozen
from f4_sequences import sha256, source_fingerprints, write_json
from sequence_ctr import SequenceResidual, gather_tokens
from zero_sequence_ctr import ZeroSequenceResidual

FREEZE=ROOT/'results/sequence_selection_v2/freeze_manifest.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    split='val' if args.smoke_test else 'test'
    study=ROOT/'results'/('sequence_test_v2_smoke' if args.smoke_test else 'sequence_test_v2')
    if study.exists():raise FileExistsError('Inference already started; automatic replay is forbidden')
    freeze=json.loads(FREEZE.read_text())
    assert freeze['complete'] and freeze['test_authorized_conditionally']
    assert sha256(FREEZE.parent/'comparison_plan.json')==freeze['comparison_plan_sha256']
    assert sha256(FREEZE.parent/'zero_validation_audit.json')==freeze['zero_validation_audit_sha256']
    for name,digest in freeze['code_sha256'].items():assert sha256(ROOT/'src'/name)==digest,name
    assert sha256(BASE_MODEL)==freeze['base_model_sha256'] and sha256(BASE_SCALER)==freeze['base_scaler_sha256']
    if not args.smoke_test:
        certificate=json.loads((FREEZE.parent/'inference_smoke_certificate.json').read_text())
        assert certificate['passed'] and certificate['freeze_sha256']==sha256(FREEZE)
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0]
        assert head==remote,'Pre-test freeze must be pushed first'
    else:head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    build=json.loads((ROOT/'results/f4_click_daily_v1/build_config.json').read_text())
    assert source_fingerprints()==build['inputs']
    f4path=ROOT/'results/f4_click_daily_v1/manifest.json'
    assert sha256(f4path)==freeze['f4_manifest_sha256']
    f4=json.loads(f4path.read_text());directory=ROOT/f4['output_directory']
    date='2017-05-12' if args.smoke_test else '2017-05-13'
    refpath=directory/f'{split}_{date}.parquet'
    for path in (refpath,directory/'events.parquet'):
        assert sha256(path)==f4['output_sha256'][str(path.relative_to(ROOT))]
    refs=pl.read_parquet(refpath).sort('impression_id')
    if args.smoke_test:refs=refs.filter(pl.col('impression_id')%1000==0)
    ids_ref=refs['impression_id'].to_numpy()
    arrays={k:refs[k].to_numpy() for k in ('start','donor_start','length','clk')}
    categories=pl.read_parquet(directory/'events.parquet',columns=['cate_idx'])['cate_idx'].to_numpy()
    device=tabular.choose_device('auto')
    backbone,base_config,mean,scale=load_frozen(device)
    cat,binary,numeric=[base_config[k] for k in ('categorical_features','binary_features','numeric_features')]
    cat_position=cat.index('cate_idx');embedding=backbone.embeddings[cat_position].weight.detach()
    assert not backbone.training and all(not p.requires_grad for p in backbone.parameters())
    models=[]
    for record in freeze['checkpoints']:
        path=ROOT/'artifacts/models'/record['run_name']/'best_model.pt'
        assert sha256(path)==record['checkpoint_sha256']
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['epoch']==record['best_epoch'] and ck['base_model_sha256']==freeze['base_model_sha256']
        cls=ZeroSequenceResidual if record['zero_sequence'] else SequenceResidual
        model=cls(embedding,record['pooling']).to(device)
        model.load_state_dict(ck['model_state_dict']);model.eval();model.requires_grad_(False);model.assert_neutral()
        run=record['test_run_name']+('_smoke' if args.smoke_test else '')
        destination=ROOT/'artifacts/predictions'/run
        assert not destination.exists()
        models.append((record,model,run,destination))
    # Exclusive marker before the first forward call. A partial run is never retried silently.
    study.mkdir(parents=True)
    with (study/'inference_started.json').open('x') as out:
        json.dump({'split':split,'smoke':args.smoke_test,'freeze_sha256':sha256(FREEZE),'git_head':head,
                   'started_utc':datetime.now(timezone.utc).isoformat(),'passes_authorized_per_checkpoint':1},out,indent=2)
    writers={};seen=np.zeros(refs.height,dtype=bool);rows=0;started=time.perf_counter()
    columns=['impression_id','clk',*cat,*binary,*numeric]
    try:
        for batch in tabular.iter_record_batches(ROOT/f'data/processed/model_input_f3/{split}.parquet',columns,8192,None):
            ids=tabular.arrow_column_to_numpy(batch,'impression_id',np.uint32)
            if args.smoke_test:
                batch=batch.filter(pa.array(ids%1000==0))
                if batch.num_rows==0:continue
                ids=tabular.arrow_column_to_numpy(batch,'impression_id',np.uint32)
            ix=np.searchsorted(ids_ref,ids)
            assert (ix<len(ids_ref)).all() and np.array_equal(ids_ref[ix],ids) and not seen[ix].any()
            seen[ix]=True
            y=tabular.arrow_column_to_numpy(batch,'clk',np.uint8)
            np.testing.assert_array_equal(y,arrays['clk'][ix])
            categorical=tabular.categorical_numpy(batch,cat)
            dense=tabular.dense_numpy(batch,binary,numeric,mean,scale)
            q=torch.from_numpy(categorical[:,cat_position].copy()).to(device)
            tokens={source:torch.from_numpy(gather_tokens(categories,arrays[key][ix],arrays['length'][ix])).to(device)
                    for source,key in (('matched','start'),('mismatched','donor_start'))}
            with torch.inference_mode():
                base=backbone(torch.from_numpy(categorical).to(device),torch.from_numpy(dense).to(device))
                base_p=torch.sigmoid(base).cpu().numpy()
                for record,model,run,destination in models:
                    history=tokens[record['source']]
                    delta=model(q,history)
                    empty=~(history>1).any(dim=1)
                    assert torch.count_nonzero(delta[empty]).item()==0
                    if record['zero_sequence']:assert torch.count_nonzero(delta).item()==0
                    predictions=torch.sigmoid(base+delta).cpu().numpy()
                    assert np.isfinite(predictions).all()
                    if record['zero_sequence']:np.testing.assert_array_equal(predictions,base_p)
                    table=pa.table({'impression_id':ids,'clk':y,'prediction':predictions})
                    if run not in writers:
                        destination.mkdir(parents=True)
                        writers[run]=pq.ParquetWriter(destination/f'{split}_predictions.partial',table.schema,compression='zstd')
                    writers[run].write_table(table)
            rows+=len(ids)
    finally:
        for writer in writers.values():writer.close()
    assert rows==refs.height and seen.all()
    if not args.smoke_test:assert rows==3_308_665
    records=[]
    for record,_,run,destination in models:
        path=destination/f'{split}_predictions.parquet'
        (destination/f'{split}_predictions.partial').replace(path)
        error=None
        if args.smoke_test:
            expected=pl.read_parquet(ROOT/'artifacts/predictions'/record['run_name']/'val_predictions.parquet').filter(pl.col('impression_id')%1000==0).sort('impression_id')
            got=pl.read_parquet(path).sort('impression_id')
            np.testing.assert_array_equal(expected['impression_id'].to_numpy(),got['impression_id'].to_numpy())
            error=float(np.max(np.abs(expected['prediction'].to_numpy()-got['prediction'].to_numpy())))
            assert error<=1e-6, (run,error)
        records.append({'run_name':run,'training_run':record['run_name'],'condition':record['condition'],'seed':record['seed'],
            'rows':rows,'prediction_sha256':sha256(path),'checkpoint_sha256':record['checkpoint_sha256'],
            'validation_smoke_max_error':error,'inference_passes':1})
    assert sha256(BASE_MODEL)==freeze['base_model_sha256'] and sha256(BASE_SCALER)==freeze['base_scaler_sha256']
    manifest={'complete':True,'split':split,'smoke':args.smoke_test,'freeze_sha256':sha256(FREEZE),'git_head':head,
        'records':records,'elapsed_seconds':time.perf_counter()-started,
        'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'exact_zero_residual_assertions_passed':True,'f123_id_label_parity':True,'device':str(device)}
    write_json(study/'manifest.json',manifest)
    if args.smoke_test:
        cert=FREEZE.parent/'inference_smoke_certificate.json';assert not cert.exists()
        write_json(cert,{'passed':True,'freeze_sha256':sha256(FREEZE),'manifest_sha256':sha256(study/'manifest.json'),
            'split':'val','rows_per_condition':rows,'conditions':len(records),'max_error':max(r['validation_smoke_max_error'] for r in records)})
    print(f'Completed {split}: {len(records)} checkpoints x {rows} rows, one pass each',flush=True)


if __name__=='__main__':main()
