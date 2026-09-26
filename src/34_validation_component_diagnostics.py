"""D3: frozen LR/DeepFM reliance diagnostics; validation only, no fitting."""
import json
import sys
import time
import joblib
import numpy as np
import polars as pl
import torch
from scipy.special import expit
from ctr_posthoc import module, metrics
from ctr_nn_common import load_frozen, tabular, BASE_MODEL, BASE_SCALER
from deep_dive_common import ROOT, DATA, Stage, predict_path, read_p, write_json


def component_logits(model, categorical, dense):
    embedded=torch.stack([e(categorical[:,i]) for i,e in enumerate(model.embeddings)],dim=1)
    wide=torch.stack([e(categorical[:,i]).squeeze(-1) for i,e in enumerate(model.first_order)],dim=1).sum(dim=1)+model.wide_dense(dense).squeeze(-1)
    deep=model.deep(torch.cat([embedded.flatten(start_dim=1),dense],dim=1)).squeeze(-1)
    fm=.5*(embedded.sum(dim=1).square()-embedded.square().sum(dim=1)).sum(dim=1)
    return wide,fm,deep


def layer(feature,f3):
    if feature in f3:return 'F3'
    if feature.startswith('log1p_') or feature.startswith('days_since_') or feature in ['has_behavior_history_14d','has_buy_history_14d']:
        return 'F2'
    return 'F1'


def main():
    stage=Stage('deep_dive_d3_v1',30)
    # Explicitly hard-coded validation input: this script has no split/test argument.
    input_path=ROOT/'data/processed/model_input_f3/val.parquet'
    for path in [input_path,BASE_MODEL,BASE_SCALER,ROOT/'src/34_validation_component_diagnostics.py',ROOT/'src/deep_dive_common.py']:
        stage.bind(path)
    context=pl.read_parquet(DATA/'val_context.parquet')
    f3m=json.loads((ROOT/'results/model_input_f3/feature_manifest.json').read_text())
    f3=set(f3m['binary_features']+f3m['numeric_features'])
    device=tabular.choose_device('auto')
    model,ck,mean,scale=load_frozen(device)
    cat,binary,numeric=[ck[k] for k in ('categorical_features','binary_features','numeric_features')]
    groups=np.array([layer(f,f3) for f in binary+numeric])
    lr_path=ROOT/'artifacts/models/lr_f1_f2_f3/best_model.joblib';stage.bind(lr_path)
    # Trusted historical checkpoint refers to the Cython loss module by short name.
    # Only stored coefficient/scaler arrays are used; validate against saved val scores.
    import sklearn._loss._loss as sklearn_loss
    sys.modules.setdefault('_loss',sklearn_loss)
    lr=joblib.load(lr_path)
    lr_module=module('diagnostic_frozen_lr','08_train_logistic_regression.py')
    coef=lr['model'].coef_.ravel().astype(np.float64); intercept=float(lr['model'].intercept_[0])
    lr_groups=np.zeros(len(coef),dtype=np.uint8)
    for name,index in lr['feature_layout']['dense_offsets'].items():lr_groups[index]={'F1':0,'F2':1,'F3':2}[layer(name,f3)]
    coef_groups=[coef*(lr_groups==i) for i in range(3)]
    columns=list(dict.fromkeys(['impression_id','clk']+cat+binary+numeric))
    def batch_predictions(batch):
        c=torch.from_numpy(tabular.categorical_numpy(batch,cat)).to(device)
        dense=torch.from_numpy(tabular.dense_numpy(batch,binary,numeric,mean,scale)).to(device)
        with torch.no_grad():
            w,f,d=component_logits(model,c,dense)
            full=w+d+f
            torch.testing.assert_close(full,model(c,dense),rtol=1e-6,atol=2e-6)
            values={'nn_wide':w.cpu().numpy(),'nn_fm':f.cpu().numpy(),'nn_deep':d.cpu().numpy(),
                'DeepFM_full':torch.sigmoid(full).cpu().numpy()}
            for label,neutral in [('no_F2',['F2']),('no_F3',['F3']),('no_F2_F3',['F2','F3'])]:
                changed=dense.clone();changed[:,np.flatnonzero(np.isin(groups,neutral))]=0
                values['DeepFM_'+label]=torch.sigmoid(model(c,changed)).cpu().numpy()
            for label,logit in [('no_wide',d+f),('no_FM',w+d),('no_deep',w+f),('wide_only',w),('deep_only',d)]:
                values['DeepFM_'+label]=torch.sigmoid(logit).cpu().numpy()
        X=lr_module.build_sparse_matrix(batch,None,lr['categorical_features'],lr['binary_features'],lr['numeric_features'],
                                        lr['vocabulary_sizes'],lr['feature_layout'],lr['scaler'])
        contrib=[np.asarray(X@v).ravel() for v in coef_groups]
        for name,v in zip(['F1','F2','F3'],contrib):values['lr_'+name]=v
        logit=sum(contrib)+intercept
        # The original CSR reduction is float32; regrouping terms changes its
        # rounding. Coefficients themselves are untouched; saved-p parity below
        # separately bounds the effect on the evaluated probabilities.
        np.testing.assert_allclose(logit,lr['model'].decision_function(X),rtol=2e-6,atol=3e-6)
        values['LR_full']=expit(logit)
        values['LR_no_F2']=expit(logit-contrib[1]);values['LR_no_F3']=expit(logit-contrib[2])
        values['LR_no_F2_F3']=expit(contrib[0]+intercept)
        values['impression_id']=batch['impression_id'].to_numpy();values['clk']=batch['clk'].to_numpy()
        return values
    smoke=next(tabular.iter_record_batches(input_path,columns,8192,8192));t=time.perf_counter()
    first=batch_predictions(smoke);elapsed=time.perf_counter()-t
    order=np.argsort(first['impression_id'])
    for name in ['LR','DeepFM']:
        saved=pl.read_parquet(predict_path(name,'val')).filter(pl.col('impression_id').is_in(first['impression_id'].tolist())).sort('impression_id')
        np.testing.assert_allclose(first[name+'_full'][order],saved['prediction'].to_numpy(),rtol=1e-5,atol=2e-7)
    projected=elapsed*context.height/8192
    assert projected < stage.cap*.8,'Pilot exceeds diagnostic budget; pause'
    write_json(stage.out/'smoke.json',{'passed':True,'rows':8192,'seconds':elapsed,'conservative_projection_seconds':projected,
        'full_decomposition_matches_checkpoint':True,'saved_validation_parity':True})
    arrays={k:[v] for k,v in first.items()}
    batches=tabular.iter_record_batches(input_path,columns,8192,None);next(batches)
    for i,batch in enumerate(batches):
        values=batch_predictions(batch)
        for k,v in values.items():arrays[k].append(v)
        if i%100==99:print(f'D3 validation only: {(i+2)*8192:,} rows',flush=True)
        assert time.perf_counter()-stage.begin<stage.cap
    output=pl.DataFrame({k:np.concatenate(v) for k,v in arrays.items()}).sort('impression_id')
    for c in ['impression_id','clk']:np.testing.assert_array_equal(output[c].to_numpy(),context[c].to_numpy())
    parity={}
    for name in ['LR','DeepFM']:
        stage.bind(predict_path(name,'val'));saved=read_p(name,'val',context)
        parity[name]=float(np.max(np.abs(output[name+'_full'].to_numpy()-saved)))
        np.testing.assert_allclose(output[name+'_full'].to_numpy(),saved,rtol=1e-5,atol=2e-7)
    target=DATA/'d3_validation_diagnostics.parquet'
    if target.exists():raise FileExistsError(target)
    output.write_parquet(target,compression='zstd')
    rows=[]
    for col in output.columns:
        if col.startswith(('LR_','DeepFM_')):
            rows.append({**metrics(context.with_columns(output[col].alias('prediction')),'val',extra=True),
                'diagnostic':col,'refitted':False,'scope':'frozen parameter reliance; not retrained ablation'})
    stage.save('validation_metrics.csv',rows)
    comp=['lr_F1','lr_F2','lr_F3','nn_wide','nn_fm','nn_deep']
    stage.save('logit_components.csv',[{'component':c,'mean':output[c].mean(),'std':output[c].std(),'min':output[c].min(),
        'max':output[c].max(),'intercept_separate':intercept if c.startswith('lr_') else None} for c in comp])
    cor=np.corrcoef(output.select(comp).to_numpy().T)
    stage.save('component_correlations.csv',[{'left':a,'right':b,'pearson':cor[i,j]} for i,a in enumerate(comp) for j,b in enumerate(comp) if j>i])
    reuse=[];recipes=[]
    for name in ['lr_f1','lr_f1_f2','lr_f1_f2_f3']:
        path=ROOT/'results'/name/'evaluation/overall_metrics.csv';stage.bind(path)
        reuse.extend([{**r,'run':name} for r in pl.read_csv(path).filter(pl.col('split')=='val').iter_rows(named=True)])
        path=ROOT/'results'/name/'run_manifest.json';stage.bind(path)
        recipe_path=ROOT/'results'/name/'metrics.json';stage.bind(recipe_path)
        recipe=json.loads(recipe_path.read_text())
        recipes.append({'run':name,'manifest':json.loads(path.read_text()),
                        'hyperparameters':recipe['hyperparameters'],'best_epoch':recipe['best_epoch']})
    stage.save('existing_lr_information_validation.csv',reuse)
    write_json(stage.out/'existing_lr_recipes.json',recipes)
    stage.finish(validation_rows=output.height,device=str(device),new_models_fitted=0,
        sklearn_checkpoint_compatibility='1.8.0 checkpoint read with 1.9.1; _loss alias, arrays only; validation parity checked',
        saved_validation_max_probability_error=parity,derived_validation_sha256=__import__('deep_dive_common').sha256(target),
        interpretation='Post-fit neutralization/deletion is off-distribution reliance, not retrained information/component ablation')


if __name__=='__main__':main()
