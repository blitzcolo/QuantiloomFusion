#!/usr/bin/env python3
"""Plan, render, resume and verify static, unregistered three-band datasets.

Python 3.11+; planning and hash verification require only the standard library.
No renderer/SDK imports. Optional pixel QA uses NumPy and OpenEXR.

Runs under Windows Python or a WSL2 Linux Python. The renderer is always the
Windows CLI; on WSL it is invoked via interop, so the dataset must live on a
Windows-accessible drive (under /mnt/<drive>/). Recipe identity is gated by the
CLI `--version` string plus content hashes of the scene assets, settings and
this script; renderer binaries are recorded for provenance but do not gate.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantiloom_fusion.core.hashing import digest, fingerprint
from quantiloom_fusion.core.host import POSIX
from quantiloom_fusion.core.jsonio import atomic_json, load_json
from quantiloom_fusion.core.locking import alive, runner_lock
from quantiloom_fusion.core.paths import canonical_path, is_drive_path, native_path
from quantiloom_fusion.core.tomlio import scalar, toml_text
from quantiloom_fusion.core.vec import add, camera_pose, cross, dot, local_rotation, norm, pose, scale, unit
from quantiloom_fusion.datagen.constants import BANDS, MODES, PATH_KEYS, SCHEMA, VERSION
from quantiloom_fusion.datagen.export_record import check_expected, safe_path, verify_capture
from quantiloom_fusion.datagen.index import sample_manifest, update_index
from quantiloom_fusion.datagen.planner import plan
from quantiloom_fusion.datagen.publish import publish, receipt_path, valid_receipt
from quantiloom_fusion.datagen.renderer import CalledProcessError, QuantiloomCli, run_closure_audit
from quantiloom_fusion.datagen.runner import run_dataset
from quantiloom_fusion.datagen.scene import canonicalize, nk_span_nm, read_gltf, reject_motion, resolve_asset, scene_for_band
from quantiloom_fusion.datagen.settings import check_keys, positive, read_settings, vector
from quantiloom_fusion.datagen.solar import solar_extension
from quantiloom_fusion.datagen.sensor import sensor_config
from quantiloom_fusion.datagen.verify import verify_dataset


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','run'):
        p=sub.add_parser(name);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
        if name=='run':
            p.add_argument('--scene',action='append',help='select a scene; repeatable')
            p.add_argument('--max-captures',type=int,help='new camera captures this run; may stop between bands')
            p.add_argument('--keep-going',action='store_true')
    p=sub.add_parser('verify');p.add_argument('--out',type=Path,required=True);p.add_argument('--inspect-exr',action='store_true')
    args=parser.parse_args(argv);out=args.out.resolve()
    try:
        if args.command=='verify':return verify_dataset(out,args.inspect_exr)
        if args.command=='run' and args.max_captures is not None and args.max_captures<1:raise ValueError('max-captures must be positive')
        out.mkdir(parents=True,exist_ok=True)
        with runner_lock(out) as (lease,lock):
            planned=plan(args.config.resolve(),out)
            if args.command=='run':return run_dataset(out,planned,args,lease,lock)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Run the same command to resume.',file=sys.stderr);return 130
    except (ValueError,OSError,KeyError,CalledProcessError) as e:
        print(f'ERROR: {e}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
