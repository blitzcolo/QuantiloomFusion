"""Plan, render, resume and verify static, unregistered three-band datasets.

Python 3.11+; planning and hash verification require only the standard library.
No renderer/SDK imports. Optional pixel QA uses NumPy and OpenEXR.

Runs under Windows Python or a WSL2 Linux Python. The renderer is always the
Windows CLI; on WSL it is invoked via interop, so the dataset must live on a
Windows-accessible drive (under /mnt/<drive>/). Recipe identity is gated by the
CLI `--version` string plus content hashes of the scene assets, settings and
this script; renderer binaries are recorded for provenance but do not gate.
"""

import argparse
from pathlib import Path
import sys

from ..core.locking import runner_lock
from .planner import plan
from .renderer import CalledProcessError
from .runner import run_dataset
from .verify import verify_dataset


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
