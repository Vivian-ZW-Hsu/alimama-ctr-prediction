"""D4: completed-day descriptive drift/repeats, with existing held-out scores."""
import datetime as dt
import numpy as np
import polars as pl
from deep_dive_common import ROOT,DATA,Stage,TABULAR,SEEDS,evidence_band,count_band,grouped_metrics,read_p,predict_path
from ctr_posthoc import length_labels


def prior_pair_counts(daily):
    keys=['user_raw','adgroup_id_raw']
    return daily.sort(keys+['event_date_bj']).with_columns(
        (pl.col('pair_rows').cum_sum().over(keys)-pl.col('pair_rows')).alias('prior_pair_rows'))


def main():
    stage=Stage('deep_dive_d4_v1',45)
    stage.bind(ROOT/'src/35_temporal_repeat_diagnostics.py');stage.bind(ROOT/'src/deep_dive_common.py')
    pairs=[]
    for split in ['train','val','test']:
        path=ROOT/f'data/processed/base/{split}.parquet';stage.bind(path)
        pair=pl.scan_parquet(path).group_by(['event_date_bj','user_raw','adgroup_id_raw']).agg(pl.len().alias('pair_rows')).collect()
        pairs.append(pair)
    pairs=prior_pair_counts(pl.concat(pairs))
    first=pairs['event_date_bj'].min()
    assert pairs.filter(pl.col('event_date_bj')==first)['prior_pair_rows'].sum()==0
    # Independent sample cumulative check, no current-day counts in the prior.
    sampled=pairs.select('user_raw','adgroup_id_raw').unique().sort('user_raw','adgroup_id_raw').head(100)
    audit=pairs.join(sampled,on=['user_raw','adgroup_id_raw'])
    for _,sub in audit.partition_by(['user_raw','adgroup_id_raw'],as_dict=True).items():
        sub=sub.sort('event_date_bj');running=0
        for r in sub.iter_rows(named=True):assert r['prior_pair_rows']==running;running+=r['pair_rows']
    pairs.write_parquet(DATA/'daily_pair_counts.parquet',compression='zstd')
    days=[];distributions=[];repeats=[];repeat_metrics=[]
    for offset in range(8):
        day=dt.date(2017,5,6)+dt.timedelta(days=offset)
        split='train' if offset<6 else ('val' if offset==6 else 'test')
        path=ROOT/f'data/processed/features/{split}.parquet';stage.bind(path)
        f=pl.scan_parquet(path).filter(pl.col('event_date_bj')==day).select('impression_id','event_date_bj','user_raw','adgroup_id_raw','clk','event_count_14d').collect().sort('impression_id')
        path=ROOT/f'data/processed/past_ctr/{split}.parquet';stage.bind(path)
        h=pl.scan_parquet(path).filter(pl.col('event_date_bj')==day).select('impression_id','user_past_impressions','ad_past_impressions').collect()
        f=f.join(h,on='impression_id',validate='1:1').sort('impression_id')
        path=ROOT/f'data/processed/f4_click_daily_v1/{split}_{day}.parquet';stage.bind(path)
        seq=pl.read_parquet(path,columns=['impression_id','length']).sort('impression_id')
        np.testing.assert_array_equal(f['impression_id'].to_numpy(),seq['impression_id'].to_numpy())
        f=f.with_columns(seq['length'],pl.Series('history_bin',length_labels(seq['length'].to_numpy())),
            pl.Series('f2_activity',count_band(f['event_count_14d'].to_numpy(),[1,10,100,1000],['0','1-9','10-99','100-999','1000+'])),
            pl.Series('user_evidence',evidence_band(f['user_past_impressions'].to_numpy())),
            pl.Series('ad_evidence',evidence_band(f['ad_past_impressions'].to_numpy())))
        daypairs=pairs.filter(pl.col('event_date_bj')==day)
        f=f.join(daypairs,on=['event_date_bj','user_raw','adgroup_id_raw'],validate='m:1').sort('impression_id')
        f=f.with_columns(pl.Series('repeat_band',count_band(f['prior_pair_rows'].to_numpy(),[1,2,6,20],['0','1','2-5','6-19','20+'])))
        if split!='train':
            ctx=pl.read_parquet(DATA/f'{split}_context.parquet')
            np.testing.assert_array_equal(ctx['impression_id'].to_numpy(),f['impression_id'].to_numpy())
            f=f.with_columns(ctx['cold_start_segment'])
        else:
            # Final train vocabulary contains train IDs: retrospective, not online novelty.
            f=f.with_columns(pl.lit('known_user__known_ad').alias('cold_start_segment'))
        days.append({'day':str(day),'split':split,'rows':f.height,'clicks':int(f['clk'].sum()),'ctr':float(f['clk'].mean()),
            'f2_present':float((f['event_count_14d']>0).mean()),'f3_user_present':float((f['user_past_impressions']>0).mean()),
            'f3_ad_present':float((f['ad_past_impressions']>0).mean()),'f4_present':float((f['length']>0).mean()),
            'new_user_share':float(f['cold_start_segment'].str.starts_with('new_user').mean()),
            'new_ad_share':float(f['cold_start_segment'].str.ends_with('new_ad').mean()),
            'prior_pair_repeat_share':float((f['prior_pair_rows']>0).mean()),
            'within_day_repeat_exposure_share':float((daypairs['pair_rows'].sum()-daypairs.height)/f.height)})
        for col in ['f2_activity','user_evidence','ad_evidence','history_bin','cold_start_segment','repeat_band']:
            for (value,),sub in f.partition_by(col,as_dict=True).items():
                distributions.append({'day':str(day),'group':col,'value':str(value),'rows':sub.height,'share':sub.height/f.height})
        for (value,),sub in f.partition_by('repeat_band',as_dict=True).items():
            repeats.append({'day':str(day),'split':split,'repeat_band':value,'rows':sub.height,'clicks':int(sub['clk'].sum()),'ctr':float(sub['clk'].mean())})
        if split!='train':
            for name,seed in [(name,42) for name in TABULAR]+[('mean_matched',seed) for seed in SEEDS]:
                stage.bind(predict_path(name,split,seed))
                repeat_metrics.extend(grouped_metrics(f,read_p(name,split,f,seed),'repeat_band',split,name,seed))
            f.select('impression_id','prior_pair_rows','pair_rows','repeat_band').write_parquet(DATA/f'{split}_repeat_context.parquet',compression='zstd')
        print(f'D4 descriptive day {day}: {f.height:,} impressions',flush=True)
    distribution=pl.DataFrame(distributions);tv=[]
    for day in sorted(distribution['day'].unique()):
        for ref in ['2017-05-11','2017-05-12']:
            for group in distribution['group'].unique():
                a={r['value']:r['share'] for r in distribution.filter((pl.col('day')==day)&(pl.col('group')==group)).iter_rows(named=True)}
                b={r['value']:r['share'] for r in distribution.filter((pl.col('day')==ref)&(pl.col('group')==group)).iter_rows(named=True)}
                tv.append({'day':day,'reference_day':ref,'group':group,'total_variation':sum(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys())/2})
    rolling=[]
    for i in range(len(days)):
        window=days[max(0,i-2):i+1];n=sum(r['rows'] for r in window);clicks=sum(r['clicks'] for r in window)
        rolling.append({'end_day':days[i]['day'],'days':len(window),'rows':n,'clicks':clicks,'ctr':clicks/n,
                        'role':'descriptive rolling traffic; not rolling-origin model evaluation'})
    for name,rows in [('daily_summary',days),('daily_distributions',distribution),('repeat_counts',repeats),
                      ('repeat_metrics',repeat_metrics),('distribution_drift',tv),('rolling_three_day',rolling)]:stage.save(name+'.csv',rows)
    # Reuse D2 hour metrics and original overall metrics instead of evaluating twice.
    path=ROOT/'results/deep_dive_d2_v1/group_metrics.csv';stage.bind(path)
    stage.save('heldout_hour_metrics.csv',pl.read_csv(path).filter(pl.col('group')=='hour_block'))
    saved=[]
    for name,run in TABULAR.items():
        path=ROOT/'results'/run/'evaluation/overall_metrics.csv';stage.bind(path)
        saved.extend([{**r,'model':name,'seed':42} for r in pl.read_csv(path).iter_rows(named=True)])
    path=ROOT/'results/sequence_posthoc_v1/overall_before_after.csv';stage.bind(path)
    saved.extend([{**{k:r[k] for k in saved[0] if k in r},'model':'mean_matched','seed':r['seed']}
        for r in pl.read_csv(path).filter((pl.col('condition')=='mean_matched')&(pl.col('stage')=='raw')).iter_rows(named=True)])
    stage.save('heldout_daily_metrics.csv',saved)
    stage.finish(prior_day_pair_audit_passed=True,training_day_predictions_evaluated=False,
        train_vocabulary_warning='Train-day knownness is retrospective full-train membership, not online novelty',
        repeated_exposure_warning='Associations only; policy selection and left-censored histories preclude causal fatigue claims')


if __name__=='__main__':main()
