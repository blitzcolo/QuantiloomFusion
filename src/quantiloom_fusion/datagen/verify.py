"""Whole-dataset verification against receipts, manifests and the index."""

from pathlib import Path

from ..core.jsonio import load_json
from .constants import BANDS
from .publish import valid_receipt


def verify_dataset(out,pixels=False):
    plan=load_json(out/'dataset.plan.json');complete=partial=0;completed_ids=[]
    for s in plan['samples']:
        results={}
        for band in BANDS:
            found=valid_receipt(out,s,band,pixels)
            if found:results[band]=found
        if len(results)==3:
            manifest=load_json(out/'samples'/s['sample_id']/'sample.json')
            if manifest['dataset_id']!=plan['dataset_id'] or manifest['state']!='complete':raise ValueError('sample identity mismatch')
            for band,(receipt,summary) in results.items():
                observation=manifest['observations'][band]
                if observation['record_sha256']!=receipt['record_sha256'] or observation['record']!=receipt['record']:raise ValueError('sample/receipt mismatch')
                for k,v in summary['products'].items():
                    if observation['products'][k]['path']!=(Path(receipt['record']).parent/v['path']).as_posix():raise ValueError('sample product path mismatch')
            complete+=1
            completed_ids.append(s['sample_id'])
        elif results:partial+=1
    index=load_json(out/'dataset.index.json') if (out/'dataset.index.json').exists() else {'complete_triplets':0}
    if index['complete_triplets']!=complete:raise ValueError('index completion count mismatch')
    if [s['sample_id'] for s in index.get('samples',[])]!=completed_ids:raise ValueError('index sample membership mismatch')
    print(f'Verified: complete={complete}, partial={partial}, planned={len(plan["samples"])}',flush=True)
    return 0
