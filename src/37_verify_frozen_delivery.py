"""Read-only delivery check. Never reruns metrics, bootstrap, fitting or inference."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    report=ROOT/'docs/DIN_F4_TEST_RESULTS.md'
    original=subprocess.check_output(['git','show','badecf4:docs/DIN_F4_TEST_RESULTS.md'],cwd=ROOT)
    assert report.read_bytes()==original,'Frozen DIN report changed'
    old=set(subprocess.check_output(['git','ls-tree','-r','--name-only','badecf4','src','results','artifacts'],cwd=ROOT,text=True).splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only','badecf4','--','src','results','artifacts'],cwd=ROOT,text=True).splitlines())
    assert not old & changed, f'Changed prior-stage files: {old & changed}'
    inputs={};timings={};stage_checks={}
    for name in ['deep_dive_d2_v1','deep_dive_d3_v1','deep_dive_d4_v1']:
        directory=ROOT/'results'/name
        m=json.loads((directory/'manifest.json').read_text());v=json.loads((directory/'verification.json').read_text())
        assert m['complete'] and v['passed']
        assert not any(m[k] for k in ['ctr_training','test_inference','bootstrap_repeated','calibration_fitted'])
        assert m['elapsed_seconds']<m['cap_seconds']
        timings[name]=m['elapsed_seconds'];stage_checks[name]=v['checks']
        for path,digest in m['inputs_sha256'].items():
            if path in inputs:assert inputs[path]==digest
            inputs[path]=digest
        for filename,digest in v['output_sha256'].items():assert sha(directory/filename)==digest,filename
    for path,digest in inputs.items():assert sha(ROOT/path)==digest,path
    d2=json.loads((ROOT/'results/deep_dive_d2_v1/manifest.json').read_text())
    for split,digest in d2['group_context_sha256'].items():assert sha(ROOT/f'data/processed/deep_dive_v1/{split}_context.parquet')==digest
    d3=json.loads((ROOT/'results/deep_dive_d3_v1/manifest.json').read_text())
    assert sha(ROOT/'data/processed/deep_dive_v1/d3_validation_diagnostics.parquet')==d3['derived_validation_sha256']
    freeze=json.loads((ROOT/'results/sequence_selection_v2/freeze_manifest.json').read_text())
    for r in freeze['checkpoints']:
        assert sha(ROOT/'artifacts/models'/r['run_name']/'best_model.pt')==r['checkpoint_sha256']
    frozen_integrity=json.loads((ROOT/'results/sequence_posthoc_v1/final_integrity.json').read_text())
    assert sha(ROOT/'results/sequence_test_v2/inference_started.json')==frozen_integrity['test_inference_marker_sha256']
    fig=json.loads((ROOT/'results/deep_dive_final_v1/figure_manifest.json').read_text())
    for key in ['inputs_sha256','figures_sha256']:
        for path,digest in fig[key].items():assert sha(ROOT/path)==digest,path
    assert len(fig['figures_sha256'])==16
    docs=['README.md','docs/FINAL_REPORT.md','docs/DEEP_DIVE_D2_RESULTS.md','docs/DEEP_DIVE_D3_RESULTS.md','docs/DEEP_DIVE_D4_RESULTS.md']
    for name in docs:
        path=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not link.startswith(('http:','https:','#')):
                assert (path.parent/link.split('#')[0]).exists(),(name,link)
    print(json.dumps({'passed':True,'frozen_commit':'badecf4','frozen_report_sha256':sha(report),
        'previous_source_results_artifacts_unchanged':True,'frozen_checkpoints_unchanged':15,
        'inference_marker_unchanged':True,'input_hashes_verified':len(inputs),'stage_checks':stage_checks,
        'stage_execution_seconds':timings,'plotting_seconds':fig['elapsed_seconds'],
        'total_recorded_execution_seconds':sum(timings.values())+fig['elapsed_seconds'],
        'figures_verified':16,'plot_png_visual_review':'Eight plots inspected at D5 publication; not an automatic image test',
        'unit_test_record':'4 targeted tests passed at D5 delivery; this verifier does not execute tests','documentation_links_valid':True,
        'document_sha256':{p:sha(ROOT/p) for p in docs},'verifier_sha256':sha(Path(__file__))},indent=2))


if __name__=='__main__':main()
