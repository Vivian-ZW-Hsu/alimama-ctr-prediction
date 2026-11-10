"""D2: fixed-group diagnostics using saved predictions only."""
import json
import numpy as np
import polars as pl
from scipy.stats import rankdata
from deep_dive_common import (ROOT, DATA, POSTHOC_DATA, Stage, TABULAR, SEEDS,
    read_p, predict_path, loss, count_band, evidence_band, top_budget, grouped_metrics)

GROUPS = ['raw_history', 'f2_activity', 'user_evidence', 'ad_evidence', 'recency', 'diversity',
          'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'has_profile', 'pid_raw', 'hour_block']


def make_context(split, stage):
    ctxpath = POSTHOC_DATA/f'{split}_groups.parquet'; stage.bind(ctxpath)
    ctx = pl.read_parquet(ctxpath)
    path = ROOT/f'data/processed/features/{split}.parquet'; stage.bind(path)
    features = pl.read_parquet(path, columns=['impression_id', 'event_count_14d', 'final_gender_code',
        'age_level', 'pvalue_level', 'shopping_level', 'has_profile', 'pid_raw', 'hour'])
    path = ROOT/f'data/processed/past_ctr/{split}.parquet'; stage.bind(path)
    f3 = pl.read_parquet(path, columns=['impression_id', 'user_past_impressions', 'ad_past_impressions'])
    ctx = ctx.join(features, on='impression_id', validate='1:1').join(f3, on='impression_id', validate='1:1').sort('impression_id')
    day = '2017-05-12' if split == 'val' else '2017-05-13'
    path = ROOT/f'data/processed/f4_click_daily_v1/{split}_{day}.parquet'; stage.bind(path)
    refs = pl.read_parquet(path, columns=['impression_id', 'user_raw', 'start', 'length', 'raw_length']).sort('impression_id')
    np.testing.assert_array_equal(ctx['impression_id'].to_numpy(), refs['impression_id'].to_numpy())
    users = refs.unique('user_raw').sort('user_raw')
    events_path = ROOT/'data/processed/f4_click_daily_v1/events.parquet'; stage.bind(events_path)
    events = pl.read_parquet(events_path, columns=['cate_id_raw', 'event_date_bj'])
    cats = events['cate_id_raw'].fill_null(-1).to_numpy(); dates = events['event_date_bj'].cast(pl.Int32).to_numpy()
    starts = users['start'].to_numpy(); lengths = users['length'].to_numpy()
    diversity = np.empty(len(users), dtype=object); recency = np.empty(len(users), dtype=object)
    for i, (start, length) in enumerate(zip(starts, lengths)):
        if length == 0:
            diversity[i] = 'empty'; recency[i] = 'empty'; continue
        cat = cats[start:start+length]; ratio = len(np.unique(cat[cat >= 0]))/length
        diversity[i] = '<=0.25' if ratio <= .25 else ('0.25-0.5' if ratio <= .5 else '>0.5')
        age = int(np.datetime64(day, 'D').astype(int))-int(dates[start+length-1]); assert age >= 1
        recency[i] = '1' if age == 1 else ('2-3' if age <= 3 else '4+')
    users = users.select('user_raw').with_columns(pl.Series('diversity', diversity.tolist()), pl.Series('recency', recency.tolist()))
    ctx = ctx.with_columns(refs['raw_length']).join(users, on='user_raw', validate='m:1').sort('impression_id')
    f2 = ctx['event_count_14d'].to_numpy(); u = ctx['user_past_impressions'].to_numpy(); a = ctx['ad_past_impressions'].to_numpy()
    ctx = ctx.with_columns(
        pl.Series('raw_history', count_band(ctx['raw_length'].to_numpy(), [1,2,6,11,20,21,50], ['0','1','2-5','6-10','11-19','20','21-49','50+'])),
        pl.Series('f2_activity', count_band(f2, [1,10,100,1000], ['0','1-9','10-99','100-999','1000+'])),
        pl.Series('user_evidence', evidence_band(u)), pl.Series('ad_evidence', evidence_band(a)),
        ((pl.col('hour')//6)*6).cast(pl.String).alias('hour_block'),
        pl.concat_str([(pl.col('event_count_14d')>0).cast(pl.UInt8), (pl.col('user_past_impressions')>0).cast(pl.UInt8),
                       (pl.col('ad_past_impressions')>0).cast(pl.UInt8), (pl.col('length')>0).cast(pl.UInt8)], separator='').alias('evidence_bits'))
    ctx = ctx.with_columns(pl.concat_str(['cold_start_segment', 'evidence_bits'], separator=':').alias('cold_evidence'))
    ctx.write_parquet(DATA/f'{split}_context.parquet', compression='zstd')
    return ctx


def common_errors(ctx, predictions, split, seed, fifth, rows, overlaps, correlations, oracle, profiles):
    names = list(TABULAR)+[fifth]
    ps = np.column_stack([predictions[n] for n in names]); y = ctx['clk'].to_numpy(); ids = ctx['impression_id'].to_numpy()
    matrices = {'pearson': np.corrcoef(ps.T), 'residual': np.corrcoef((ps-y[:,None]).T),
                'spearman': np.corrcoef(np.column_stack([rankdata(ps[:,i]) for i in range(5)]).T)}
    for kind, matrix in matrices.items():
        for i, left in enumerate(names):
            for j in range(i+1, 5):
                correlations.append({'split': split, 'seed': seed, 'fifth': fifth, 'kind': kind,
                                     'left': left, 'right': names[j], 'correlation': matrix[i,j]})
    policies = ['p>=0.5', 'top_1pct', 'top_5pct', 'top_10pct', 'legacy_quantile_10pct']
    for policy in policies:
        if policy == 'p>=0.5': selected = ps >= .5
        elif policy == 'legacy_quantile_10pct': selected = ps >= np.quantile(ps, .9, axis=0)
        else:
            q = {'top_1pct': .01, 'top_5pct': .05, 'top_10pct': .1}[policy]
            selected = np.column_stack([top_budget(ps[:,i], ids, q) for i in range(5)])
        caught = selected.sum(axis=1); positive = y == 1
        for n in range(6):
            ix = caught == n
            rows.append({'split': split, 'seed': seed, 'fifth': fifth, 'policy': policy, 'models_catching': n,
                         'rows': int(ix.sum()), 'clicks': int(y[ix].sum()), 'all_clicks': int(y.sum()),
                         'click_share': float(y[ix].sum()/y.sum())})
        union = selected.any(axis=1)
        for i, left in enumerate(names):
            for j in range(i+1, 5):
                fn1 = positive & ~selected[:,i]; fn2 = positive & ~selected[:,j]
                fp1 = ~positive & selected[:,i]; fp2 = ~positive & selected[:,j]
                overlaps.append({'split': split, 'seed': seed, 'fifth': fifth, 'policy': policy,
                    'left': left, 'right': names[j], 'left_recall': float((positive & selected[:,i]).sum()/y.sum()),
                    'right_recall': float((positive & selected[:,j]).sum()/y.sum()),
                    'fn_jaccard': float((fn1 & fn2).sum()/max(1,(fn1 | fn2).sum())),
                    'fp_jaccard': float((fp1 & fp2).sum()/max(1,(fp1 | fp2).sum())),
                    'left_selected': int(selected[:,i].sum()), 'right_selected': int(selected[:,j].sum()),
                    'left_false_positives': int(fp1.sum()), 'right_false_positives': int(fp2.sum())})
        oracle.append({'split': split, 'seed': seed, 'fifth': fifth, 'policy': policy,
                       'oracle': 'union_recall_expanded_coverage', 'value': float(y[union].sum()/y.sum()),
                       'coverage': float(union.mean()), 'rows': len(y)})
        if policy == 'top_10pct' and fifth == 'mean_matched':
            f = ctx.with_columns(pl.Series('common_miss', (caught == 0) & positive), pl.Series('any_hit', union & positive))
            for group in ['cold_evidence', 'f2_activity', 'history_bin', 'has_profile', 'age_level']:
                for (value,), sub in f.partition_by(group, as_dict=True).items():
                    profiles.append({'split': split, 'seed': seed, 'group': group, 'value': str(value), 'rows': sub.height,
                        'clicks': int(sub['clk'].sum()), 'common_missed_clicks': int(sub['common_miss'].sum()),
                        'within_group_common_fnr': float(sub['common_miss'].sum()/max(1,sub['clk'].sum())),
                        'share_of_all_common_misses': float(sub['common_miss'].sum()/max(1,((caught==0)&positive).sum()))})
    losses = loss(y[:,None], ps)
    for i, name in enumerate(names):
        oracle.append({'split': split, 'seed': seed, 'fifth': fifth, 'policy': 'none', 'oracle': 'loss_gap_'+name,
                       'value': float(losses[:,i].mean()-losses.min(axis=1).mean()), 'coverage': 1., 'rows': len(y)})
    oracle.append({'split': split, 'seed': seed, 'fifth': fifth, 'policy': 'none', 'oracle': 'label_informed_min_logloss',
                   'value': float(losses.min(axis=1).mean()), 'coverage': 1., 'rows': len(y)})


def long_loss(ctx, matched, refs, split, seed):
    rows = []; mask = ctx['length'].to_numpy() == 20
    for name, ref in refs.items():
        y = ctx['clk'].to_numpy(); delta = loss(y,matched)-loss(y,ref)
        frame = ctx.with_columns(pl.Series('loss_delta', delta), pl.Series('p_delta', matched-ref),
            pl.Series('zero_band', count_band(refs['zero'], [.01,.03,.05,.1,.2,.5], ['<.01','.01-.03','.03-.05','.05-.1','.1-.2','.2-.5','.5+']))).filter(pl.Series(mask))
        for group in ['raw_history','category_hit','cold_start_segment','diversity','recency','zero_band']:
            for (value, label), sub in frame.partition_by([group,'clk'],as_dict=True).items():
                rows.append({'split': split, 'seed': seed, 'reference': name, 'group': group, 'value': str(value), 'label': int(label),
                    'rows': sub.height, 'long_history_rows': frame.height, 'loss_delta_sum': float(sub['loss_delta'].sum()),
                    'contribution_to_long_history_delta': float(sub['loss_delta'].sum()/frame.height),
                    'mean_probability_delta': float(sub['p_delta'].mean()), 'mean_loss_delta': float(sub['loss_delta'].mean())})
        for group in ['raw_history','category_hit','cold_start_segment','diversity','recency','zero_band']:
            sub = [r for r in rows if r['reference']==name and r['group']==group]
            np.testing.assert_allclose(sum(r['contribution_to_long_history_delta'] for r in sub), delta[mask].mean(), atol=1e-13)
    return rows


def gauc_contributions(ctx, split, seed, stage):
    def load(name):
        run = TABULAR['DeepFM'] if name == 'zero' else f'seq_{name}_l20_lr500_s{seed}_v1'+('' if split=='val' else '_test_v2')
        path = ROOT/'results'/run/'evaluation/eligible_user_auc.parquet'; stage.bind(path)
        return pl.read_parquet(path).filter(pl.col('split')==split).sort('user_raw')
    m = load('mean_matched'); total = m['rows'].sum(); rows = []
    cols = ['history_bin','known_user','f2_activity','user_evidence']
    for col in cols:
        assert ctx.group_by('user_raw').agg(pl.col(col).n_unique())[col].max() == 1
    users = ctx.unique('user_raw').select('user_raw',*cols)
    for name in ['zero','mean_mismatched']:
        ref = load(name)
        np.testing.assert_array_equal(m['user_raw'].to_numpy(),ref['user_raw'].to_numpy())
        np.testing.assert_array_equal(m['rows'].to_numpy(),ref['rows'].to_numpy())
        joined = m.with_columns(pl.Series('reference_auc',ref['user_auc'])).join(users,on='user_raw',validate='1:1')
        joined = joined.with_columns(((pl.col('user_auc')-pl.col('reference_auc'))*pl.col('rows')).alias('numerator'))
        for col in cols:
            for (v,), sub in joined.partition_by(col,as_dict=True).items():
                rows.append({'split':split,'seed':seed,'reference':name,'group':col,'value':str(v),'eligible_users':sub.height,
                    'eligible_rows':int(sub['rows'].sum()),'all_eligible_rows':int(total),
                    'group_gauc_delta':float(sub['numerator'].sum()/sub['rows'].sum()),
                    'contribution_to_global_delta':float(sub['numerator'].sum()/total)})
    return rows


def cold_composition(ctx, raw, calibrated, split, name, seed):
    mask = ctx['cold_start_segment'].to_numpy() == 'new_user__new_ad'
    rows=[]
    for stage,p in [('raw',raw),('calibrated',calibrated)]:
        f=ctx.with_columns(pl.Series('prediction',p)).filter(pl.Series(mask))
        total_excess=f['prediction'].sum()-f['clk'].sum()
        for group in ['evidence_bits','has_profile','user_evidence','ad_evidence','history_bin']:
            for (value,),sub in f.partition_by(group,as_dict=True).items():
                expected=float(sub['prediction'].sum()); clicks=int(sub['clk'].sum())
                rows.append({'split':split,'model':name,'seed':seed,'stage':stage,'group':group,'value':str(value),
                    'rows':sub.height,'clicks':clicks,'expected_clicks':expected,'excess_expected_clicks':expected-clicks,
                    'share_total_excess':(expected-clicks)/total_excess,'ctr':clicks/sub.height,
                    'mean_prediction':expected/sub.height,'copc':clicks/expected})
    return rows


def main():
    stage=Stage('deep_dive_d2_v1',60)
    DATA.mkdir(exist_ok=False)
    for file in ['src/33_deep_dive_saved_predictions.py','src/deep_dive_common.py']:
        stage.bind(ROOT/file)
    for file in (ROOT/'results/sequence_posthoc_v1').glob('*'):
        if file.is_file():stage.bind(file)
    errors=[];overlaps=[];correlations=[];oracle=[];profiles=[];metrics=[];long=[];gauc=[];cold=[]
    for split in ['val','test']:
        ctx=make_context(split,stage)
        for name in TABULAR:
            stage.bind(predict_path(name,split))
        tabular={name:read_p(name,split,ctx) for name in TABULAR}
        for name,p in tabular.items():
            for group in GROUPS+['cold_evidence']:
                metrics.extend(grouped_metrics(ctx,p,group,split,name,42))
        for seed in SEEDS:
            ps={**tabular}
            for name in ['mean_matched','mean_mismatched','attention_matched']:
                stage.bind(predict_path(name,split,seed));ps[name]=read_p(name,split,ctx,seed)
            for fifth in ['mean_matched','attention_matched']:
                common_errors(ctx,ps,split,seed,fifth,errors,overlaps,correlations,oracle,profiles)
            for name in ['mean_matched','mean_mismatched']:
                for group in GROUPS+['cold_evidence']:
                    metrics.extend(grouped_metrics(ctx,ps[name],group,split,name,seed))
            long.extend(long_loss(ctx,ps['mean_matched'],{'mean_mismatched':ps['mean_mismatched'],'zero':ps['DeepFM']},split,seed))
            gauc.extend(gauc_contributions(ctx,split,seed,stage))
            stage.bind(predict_path('mean_matched',split,seed,True))
            cp=read_p('mean_matched',split,ctx,seed,True)
            cold.extend(cold_composition(ctx,ps['mean_matched'],cp,split,'mean_matched',seed))
            print(f'D2 saved predictions: {split} residual seed {seed} complete',flush=True)
        stage.bind(predict_path('zero',split,42,True))
        cold.extend(cold_composition(ctx,tabular['DeepFM'],read_p('zero',split,ctx,42,True),split,'zero',42))
    for name,rows in [('common_errors',errors),('pairwise_overlap',overlaps),('correlations',correlations),('oracle',oracle),
                      ('common_miss_profiles',profiles),('group_metrics',metrics),('long_history_loss',long),
                      ('gauc_contributions',gauc),('double_cold_composition',cold)]:
        stage.save(name+'.csv',rows)
    stage.finish(descriptive_only=True,zero_reference='one frozen DeepFM',
        group_context_sha256={s:__import__('deep_dive_common').sha256(DATA/f'{s}_context.parquet') for s in ['val','test']})


if __name__=='__main__':main()
