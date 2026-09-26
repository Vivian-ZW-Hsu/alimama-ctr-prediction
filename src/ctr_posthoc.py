"""Saved-prediction analysis utilities. No CTR model or inference imports."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl

from f4_sequences import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'results/sequence_posthoc_v1'
DATA=ROOT/'data/processed/sequence_posthoc_v1'


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'src'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


evaluator=module('shared_posthoc_evaluator','10_evaluate_predictions.py')
calibration=module('legacy_posthoc_calibration','14_calibrate_predictions.py')


def records():
    freeze=json.loads((ROOT/'results/sequence_selection_v2/freeze_manifest.json').read_text())
    return [r for r in freeze['checkpoints'] if not r['zero_sequence'] or r['seed']==42]


def prediction_path(record,split):
    run=record['run_name'] if split=='val' else record['test_run_name']
    return ROOT/'artifacts/predictions'/run/f'{split}_predictions.parquet'


def load_prediction(record,split,context):
    path=prediction_path(record,split)
    pred=pl.read_parquet(path).sort('impression_id')
    assert pred.height==context.height
    for c in ('impression_id','clk'):
        np.testing.assert_array_equal(pred[c].to_numpy(),context[c].to_numpy())
    p=pred['prediction'].to_numpy().astype(np.float64)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return p


def length_labels(length):
    return np.select([length==0,length==1,length<=5,length<=10,length<20],
                     ['0','1','2-5','6-10','11-19'],default='20')


def category_overlap(categories,starts,lengths,candidates):
    result=np.zeros(len(starts),dtype=bool)
    for start in range(0,len(starts),65536):
        stop=start+65536
        mask=np.arange(20)[None,:]<lengths[start:stop,None]
        idx=starts[start:stop,None]+np.arange(20)[None,:]
        tokens=categories[np.where(mask,idx,0)]
        result[start:stop]=((tokens==candidates[start:stop,None])&mask&(tokens>=0)&(candidates[start:stop,None]>=0)).any(axis=1)
    return result


def metrics(frame,split='test',segment='overall',extra=False):
    row=evaluator.metric_row(split,segment,frame)
    if extra:
        row['ece']=evaluator.weighted_ece(evaluator.build_reliability(split,frame,10))
        gauc,_=evaluator.build_gauc(split,frame)
        row.update({k:gauc[k] for k in ('gauc','eligible_users','eligible_rows','eligible_row_coverage')})
    return row


def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p)
    corrected=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result=np.empty_like(p);result[order]=corrected
    return result


ROW_DTYPE=np.dtype([('user','<u4'),('label','u1'),('end','u1'),('padding','<u2')])


def write_rank_model(path, user, labels, predictions, n_users, gauc_numerators):
    order=np.argsort(-predictions,kind='stable')
    scores=predictions[order]
    rows=np.zeros(len(user),dtype=ROW_DTYPE)
    rows['user']=user[order];rows['label']=labels[order]
    rows['end']=np.r_[scores[:-1]!=scores[1:],True]
    p=np.clip(predictions,1e-7,1-1e-7)
    losses=-(labels*np.log(p)+(1-labels)*np.log1p(-p))
    user_loss=np.bincount(user,weights=losses,minlength=n_users).astype('<f8')
    with Path(path).open('xb') as out:
        rows.tofile(out);user_loss.tofile(out);np.asarray(gauc_numerators,dtype='<f8').tofile(out)


def write_common(path,n_rows,n_users,n_models,counts,eligible_counts):
    with Path(path).open('xb') as out:
        np.array([n_rows,n_users,n_models],dtype='<u8').tofile(out)
        np.asarray(counts,dtype='<u4').tofile(out);np.asarray(eligible_counts,dtype='<u4').tofile(out)
