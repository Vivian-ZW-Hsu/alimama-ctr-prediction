"""Cache frozen DeepFM logits and compact F4 references for TRAIN/VAL ONLY."""
import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER, tabular, load_frozen
from f4_sequences import sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--f4-run',required=True)
    p.add_argument('--run-name',required=True)
    args=p.parse_args()
    if not args.run_name.replace('_','').isalnum():p.error('Invalid run name')
    out=ROOT/'data/processed'/args.run_name
    result=ROOT/'results'/args.run_name
    if out.exists() or result.exists():raise FileExistsError('Cache names are immutable; choose a new run')
    f4path=ROOT/'results'/args.f4_run/'manifest.json'
    f4=json.loads(f4path.read_text())
    assert f4['complete'] and f4['config']['max_length']==20
    directory=ROOT/f4['output_directory']
    for name,digest in f4['output_sha256'].items():
        assert sha256(ROOT/name)==digest,name
    device=tabular.choose_device('auto')
    model,checkpoint,mean,scale=load_frozen(device)
    cat,binary,numeric=[checkpoint[k] for k in ('categorical_features','binary_features','numeric_features')]
    assert all(not v.requires_grad for v in model.parameters()) and not model.training
    category_position=cat.index('cate_idx')
    baseline_digest=sha256(BASE_MODEL)
    scaler_digest=sha256(BASE_SCALER)
    out.mkdir(parents=True);result.mkdir(parents=True)
    np.save(out/'category_embedding.npy',model.embeddings[category_position].weight.detach().cpu().numpy())
    started=time.perf_counter()
    records=[]
    source_columns=['impression_id','clk',*cat,*binary,*numeric]
    # Global raw row IDs are compact. References consume O(N), never O(N*L).
    id_space=26_557_961
    for split in ('train','val'):
        present=np.zeros(id_space,dtype=bool)
        starts=np.zeros(id_space,dtype=np.uint32)
        donor=np.zeros(id_space,dtype=np.uint32)
        lengths=np.zeros(id_space,dtype=np.uint8)
        labels=np.zeros(id_space,dtype=np.uint8)
        for path in sorted(directory.glob(f'{split}_*.parquet')):
            refs=pl.read_parquet(path)
            ids=refs['impression_id'].to_numpy()
            assert not present[ids].any()
            present[ids]=True
            starts[ids]=refs['start'].to_numpy();donor[ids]=refs['donor_start'].to_numpy()
            lengths[ids]=refs['length'].to_numpy();labels[ids]=refs['clk'].to_numpy()
        expected=int(present.sum())
        old=None
        if split=='val':
            old=np.full(id_space,np.nan,dtype=np.float32)
            saved=pl.read_parquet(ROOT/'artifacts/predictions/deepfm_f1_f2_f3/val_predictions.parquet')
            old[saved['impression_id'].to_numpy()]=saved['prediction'].to_numpy()
        writer=None
        rows=0; max_error=0.0
        temporary=out/f'{split}.partial'
        seen=np.zeros(id_space,dtype=bool)
        try:
            for batch in tabular.iter_record_batches(ROOT/f'data/processed/model_input_f3/{split}.parquet',source_columns,8192,None):
                ids=tabular.arrow_column_to_numpy(batch,'impression_id',np.uint32)
                if f4['config']['smoke']:
                    batch=batch.filter(pa.array(present[ids]))
                    if batch.num_rows==0:continue
                    ids=tabular.arrow_column_to_numpy(batch,'impression_id',np.uint32)
                assert present[ids].all() and not seen[ids].any()
                seen[ids]=True
                y=tabular.arrow_column_to_numpy(batch,'clk',np.uint8)
                np.testing.assert_array_equal(y,labels[ids])
                c=tabular.categorical_numpy(batch,cat)
                dense=tabular.dense_numpy(batch,binary,numeric,mean,scale)
                with torch.no_grad():
                    logits=model(torch.from_numpy(c).to(device),torch.from_numpy(dense).to(device))
                    probs=torch.sigmoid(logits).cpu().numpy()
                    logits=logits.cpu().numpy()
                if old is not None:
                    assert np.isfinite(old[ids]).all()
                    error=float(np.max(np.abs(probs-old[ids])))
                    max_error=max(max_error,error)
                    assert error<=1e-6, f'Frozen baseline prediction mismatch: {error}'
                table=pa.table({'impression_id':ids,'clk':y,'candidate_category':c[:,category_position].astype(np.uint32),
                    'start':starts[ids],'donor_start':donor[ids],'length':lengths[ids],
                    'base_logit':logits.astype(np.float32)})
                if writer is None:writer=pq.ParquetWriter(temporary,table.schema,compression='zstd')
                writer.write_table(table)
                rows+=len(ids)
        finally:
            if writer:writer.close()
        assert rows==expected and np.array_equal(seen,present)
        temporary.replace(out/f'{split}.parquet')
        records.append({'split':split,'rows':rows,'baseline_max_probability_error':max_error,
                        'id_label_parity':True,'sha256':sha256(out/f'{split}.parquet')})
        print(records[-1],flush=True)
    assert sha256(BASE_MODEL)==baseline_digest and sha256(BASE_SCALER)==scaler_digest
    write_json(result/'manifest.json',{'complete':True,'run_name':args.run_name,'smoke':f4['config']['smoke'],
        'splits':['train','val'],'test_read':False,'f4_manifest':str(f4path.relative_to(ROOT)),
        'f4_manifest_sha256':sha256(f4path),'base_model_sha256':baseline_digest,'scaler_sha256':scaler_digest,
        'source_code_sha256':sha256(ROOT/'src/16_train_tabular_nn.py'),
        'category_embedding_sha256':sha256(out/'category_embedding.npy'),
        'device':str(device),'records':records,'directory':str(out.relative_to(ROOT)),
        'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':main()
