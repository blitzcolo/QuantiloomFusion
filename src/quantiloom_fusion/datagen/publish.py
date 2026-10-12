"""Receipt validation and atomic publication of rendered captures."""

import os
import shutil
import uuid

from ..core.hashing import digest
from ..core.jsonio import atomic_json, load_json
from .export_record import safe_path, verify_capture


def receipt_path(out,sample,band):return out/'state'/sample['sample_id']/(band+'.json')


def valid_receipt(out,sample,band,pixels):
    path=receipt_path(out,sample,band)
    if not path.exists():return None
    receipt=load_json(path)
    if receipt.get('status')!='complete':return None
    camera=sample['cameras'][band]
    if receipt['job_sha256']!=camera['job_sha256']:raise ValueError('receipt recipe mismatch')
    record=safe_path(out,receipt['record'])
    if digest(record)!=receipt['record_sha256']:raise ValueError('sidecar hash mismatch')
    summary=verify_capture(record,camera,pixels)
    return receipt,summary


def publish(out,sample,band,attempt,camera,pixels):
    record_path=attempt/'frame.metadata.json'
    summary=verify_capture(record_path,camera,pixels)
    destination=out/'samples'/sample['sample_id']/band/('v'+attempt.name.rsplit('-',1)[-1])
    if destination.exists():
        verify_capture(destination/record_path.name,camera,pixels)
        if digest(destination/record_path.name)!=digest(record_path):raise ValueError('publication version does not match attempt')
    else:
        temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
        temporary.mkdir(parents=True)
        record=load_json(record_path)
        for artifact in [record['replay'],*record['products']]:
            target=safe_path(temporary,artifact['path']);target.parent.mkdir(parents=True,exist_ok=True)
            # Copy file content, not the renderer's Windows owner-only ACLs.
            shutil.copyfile(safe_path(attempt,artifact['path']),target)
        shutil.copyfile(record_path,temporary/record_path.name)
        verify_capture(temporary/record_path.name,camera,pixels)
        os.replace(temporary,destination)
    receipt={'status':'complete','job_sha256':camera['job_sha256'],'record':(destination/record_path.name).relative_to(out).as_posix(),
             'record_sha256':digest(destination/record_path.name),'record_id':summary['record_id'],'verified':'pixels_and_hashes' if pixels else 'hashes_and_effective_parameters','pixel_qa':summary['pixel_qa']}
    atomic_json(receipt_path(out,sample,band),receipt)
    return receipt,summary
