"""Freeze group membership and exact metric inputs using existing artifacts only."""
import json
import time
import numpy as np
import polars as pl
from ctr_posthoc import ROOT, STUDY, DATA, records, prediction_path, load_prediction, evaluator, length_labels, category_overlap, write_rank_model, write_common, sha256, write_json


def main():
    if DATA.exists() or (STUDY/'input_manifest.json').exists():raise FileExistsError('Analysis inputs are immutable')
    DATA.mkdir(parents=True)
    begin=time.perf_counter();fingerprints={};group_rows=[]
    protocol=STUDY/'protocol.json';fingerprints[str(protocol.relative_to(ROOT))]=sha256(protocol)
    freeze=ROOT/'results/sequence_selection_v2/freeze_manifest.json';fingerprints[str(freeze.relative_to(ROOT))]=sha256(freeze)
    # Bind every original predictor, including the three identical zero references.
    test_manifest=json.loads((ROOT/'results/sequence_test_v2/manifest.json').read_text())
    frozen=json.loads(freeze.read_text())
    for r in frozen['checkpoints']:
        path=prediction_path(r,'val');assert sha256(path)==r['validation_prediction_sha256']
        fingerprints[str(path.relative_to(ROOT))]=r['validation_prediction_sha256']
    for r in test_manifest['records']:
        path=ROOT/'artifacts/predictions'/r['run_name']/'test_predictions.parquet'
        assert sha256(path)==r['prediction_sha256'];fingerprints[str(path.relative_to(ROOT))]=r['prediction_sha256']
    f4=ROOT/'data/processed/f4_click_daily_v1'
    events=pl.read_parquet(f4/'events.parquet',columns=['cate_id_raw'])['cate_id_raw'].fill_null(-1).to_numpy()
    fingerprints[str((f4/'events.parquet').relative_to(ROOT))]=sha256(f4/'events.parquet')
    for name in ('user_raw_mapping.parquet','adgroup_id_raw_mapping.parquet'):
        path=ROOT/'artifacts/encoders/mappings'/name;fingerprints[str(path.relative_to(ROOT))]=sha256(path)
    for split,day in (('val','2017-05-12'),('test','2017-05-13')):
        basepath=ROOT/f'data/processed/base/{split}.parquet'
        base=pl.read_parquet(basepath,columns=['impression_id','clk','user_raw','adgroup_id_raw','cate_id_raw']).sort('impression_id')
        path=f4/f'{split}_{day}.parquet'
        refs=pl.read_parquet(path,columns=['impression_id','clk','start','length']).sort('impression_id')
        for c in ('impression_id','clk'):np.testing.assert_array_equal(base[c].to_numpy(),refs[c].to_numpy())
        hit=category_overlap(events,refs['start'].to_numpy(),refs['length'].to_numpy(),base['cate_id_raw'].fill_null(-1).to_numpy())
        ctx=evaluator.add_cold_start_flags(base).with_columns(
            refs['length'],pl.Series('history_bin',length_labels(refs['length'].to_numpy())),
            pl.Series('category_hit',hit)).sort('impression_id')
        ctx.write_parquet(DATA/f'{split}_groups.parquet',compression='zstd')
        for group in ('history_bin','category_hit','cold_start_segment'):
            for r in ctx.group_by(group).agg(pl.len().alias('rows'),pl.col('clk').mean().alias('ctr')).iter_rows(named=True):
                group_rows.append({'split':split,'group':group,'value':str(r[group]),'rows':r['rows'],'ctr':r['ctr']})
        for x in (path,basepath):fingerprints[str(x.relative_to(ROOT))]=sha256(x)
    ctx=pl.read_parquet(DATA/'test_groups.parquet')
    user_values,user=np.unique(ctx['user_raw'].to_numpy(),return_inverse=True)
    U=len(user_values);N=ctx.height;user=user.astype(np.uint32)
    labels=ctx['clk'].to_numpy().astype(np.uint8)
    counts=np.bincount(user,minlength=U)
    clicks=np.bincount(user,weights=labels,minlength=U)
    eligible=(clicks>0)&(clicks<counts);eligible_counts=counts*eligible
    rs=records();write_common(DATA/'common.bin',N,U,len(rs),counts,eligible_counts)
    model_records=[]
    for index,r in enumerate(rs):
        p=load_prediction(r,'test',ctx)
        gauc,users=evaluator.build_gauc('test',ctx.with_columns(pl.Series('prediction',p)))
        gu=np.searchsorted(user_values,users['user_raw'].to_numpy())
        numerator=np.zeros(U,dtype=float);numerator[gu]=users['rows'].to_numpy()*users['user_auc'].to_numpy()
        write_rank_model(DATA/f'model_{index}.bin',user,labels,p,U,numerator)
        model_records.append({**r,'index':index,'test_prediction_path':str(prediction_path(r,'test').relative_to(ROOT))})
        print(f'Prepared exact rank input {index+1}/{len(rs)}: {r["condition"]} seed {r["seed"]}',flush=True)
    pl.DataFrame(group_rows).write_csv(STUDY/'group_counts.csv')
    write_json(STUDY/'input_manifest.json',{'complete':True,'rows':N,'users':U,'models':model_records,
        'eligible_users':int(eligible.sum()),'eligible_rows':int(eligible_counts.sum()),
        'input_sha256':fingerprints,'derived_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in DATA.iterdir()},
        'code_sha256':{p:sha256(ROOT/'src'/p) for p in ('ctr_posthoc.py','28_prepare_prediction_analysis.py')},
        'elapsed_seconds':time.perf_counter()-begin})


if __name__=='__main__':main()
