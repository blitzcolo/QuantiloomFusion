"""Per-sample manifests and the rolling dataset.index.json."""

from pathlib import Path

from ..core.jsonio import atomic_json, load_json
from .constants import BANDS, MODES, SCHEMA, VERSION


def sample_manifest(out,plan,sample,results):
    if set(results)!=set(BANDS):return None
    manifest={'schema':'quantiloomfusion.sample','schema_version':1,'dataset_id':plan['dataset_id'],'sample_id':sample['sample_id'],
              'scene_id':sample['scene_id'],'split_group':sample['split_group'],'state':'complete','parameters':sample['parameters'],
              'rig_to_world':sample['rig_to_world'],'registration':'unregistered',
              'geometry_ground_truth':plan['recipe']['settings']['dataset']['geometry_truth'],'observations':{}}
    for band,(receipt,summary) in results.items():
        base=Path(receipt['record']).parent
        manifest['observations'][band]={'record':receipt['record'],'record_sha256':receipt['record_sha256'],'camera_to_world':sample['cameras'][band]['camera_to_world'],
            'spectral_mode':MODES[band],'products':{k:{**v,'path':(base/v['path']).as_posix()} for k,v in summary['products'].items()},
            'extra_products':{(base/k).as_posix():v for k,v in summary['extra_products'].items()},'pixel_qa':summary['pixel_qa']}
    path=out/'samples'/sample['sample_id']/'sample.json';atomic_json(path,manifest)
    return path.relative_to(out).as_posix()


def update_index(out,plan):
    complete=[]
    for s in plan['samples']:
        p=out/'samples'/s['sample_id']/'sample.json'
        if p.exists():
            m=load_json(p)
            if m['dataset_id']==plan['dataset_id'] and m['state']=='complete':complete.append({'sample_id':s['sample_id'],'manifest':p.relative_to(out).as_posix(),'split_group':s['split_group']})
    atomic_json(out/'dataset.index.json',{'schema':SCHEMA,'schema_version':VERSION,'dataset_id':plan['dataset_id'],'complete_triplets':len(complete),'planned_triplets':len(plan['samples']),'samples':complete})
