"""Whole-dataset verification against receipts, manifests and the index."""

from pathlib import Path

from ..core.jsonio import load_json
from .constants import BANDS, REFERENCE_PREFIX
from .publish import valid_receipt


def _check_observations(published,results,where):
    if set(published)!=set(results):raise ValueError(f'{where} bands mismatch')
    for band,(receipt,summary) in results.items():
        observation=published[band]
        if observation['record_sha256']!=receipt['record_sha256'] or observation['record']!=receipt['record']:raise ValueError('sample/receipt mismatch')
        for k,v in summary['products'].items():
            if observation['products'][k]['path']!=(Path(receipt['record']).parent/v['path']).as_posix():raise ValueError('sample product path mismatch')


def verify_dataset(out,pixels=False):
    plan=load_json(out/'dataset.plan.json');complete=partial=0;completed=[]
    planned_references=published_references=0
    for s in plan['samples']:
        results={};references={}
        for band in BANDS:
            found=valid_receipt(out,s,band,pixels)
            if found:results[band]=found
        for band in s.get('references',{}):
            found=valid_receipt(out,s,REFERENCE_PREFIX+band,pixels)
            if found:references[band]=found
        if s.get('references'):planned_references+=1
        if len(results)==3:
            manifest=load_json(out/'samples'/s['sample_id']/'sample.json')
            if manifest['dataset_id']!=plan['dataset_id'] or manifest['state']!='complete':raise ValueError('sample identity mismatch')
            _check_observations(manifest['observations'],results,'observations')
            # A manifest lists references only once all three are published.
            _check_observations(manifest.get('references',{}),references if len(references)==3 else {},'references')
            if manifest.get('references'):published_references+=1
            complete+=1
            completed.append((s['sample_id'],bool(manifest.get('references'))))
        elif results:partial+=1
    index=load_json(out/'dataset.index.json') if (out/'dataset.index.json').exists() else {'complete_triplets':0}
    if index['complete_triplets']!=complete:raise ValueError('index completion count mismatch')
    if [s['sample_id'] for s in index.get('samples',[])]!=[sid for sid,_ in completed]:raise ValueError('index sample membership mismatch')
    if [s.get('references',False) for s in index.get('samples',[])]!=[flag for _,flag in completed]:raise ValueError('index reference flags mismatch')
    tail=f', references={published_references}/{planned_references}' if planned_references else ''
    print(f'Verified: complete={complete}, partial={partial}, planned={len(plan["samples"])}{tail}',flush=True)
    return 0
