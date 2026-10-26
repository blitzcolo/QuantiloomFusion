---
name: dataset-inspect
description: Summarise a QuantiloomFusion three-band dataset directory (plan, completion, one sample's cameras) and print per-product EXR statistics, validity breakdown and IR apparent-vs-true temperature offsets. Use when asked to look at, check or compare a dataset or its EXR outputs.
allowed-tools:
  - read
  - grep
  - glob
---

# 数据集检查

只读操作，不修改数据集。需要 numpy 与 OpenEXR（WSL 的 `python3` 已有；Windows 用项目 venv 或借用的 `/mnt/h/paper-exp/.venv/Scripts/python.exe`）。

`quantiloom_fusion.dataset` 读取器落地后，优先用它代替下面的内联脚本，并把本 skill 改成调用它。

## 用法

```bash
cd /mnt/d/QuantiloomFusion
DS=build/env-smoke-0.8.3          # 数据集根目录
SAMPLE=                           # 留空 = 第一个完整样本；或填 sample_id
python3 - "$DS" $SAMPLE <<'EOF'
import json, pathlib, sys
import numpy as np
import OpenEXR

root = pathlib.Path(sys.argv[1])
want = sys.argv[2] if len(sys.argv) > 2 else None
load = lambda p: json.loads(p.read_text(encoding='utf-8'))
plan = load(root / 'dataset.plan.json')
index = load(root / 'dataset.index.json') if (root / 'dataset.index.json').exists() else {'complete_triplets': 0, 'samples': []}
print('dataset_id', plan['dataset_id'])
print('renderer  ', plan['recipe']['cli_version'])
print('scenes    ', [s['id'] for s in plan['scenes']])
print('complete  ', index['complete_triplets'], '/', len(plan['samples']))
entries = [s for s in index['samples'] if want in (None, s['sample_id'])]
if not entries:
    sys.exit('no matching complete sample')
m = load(root / entries[0]['manifest'])
print('sample    ', m['sample_id'], m['parameters'])

def read(path):
    with OpenEXR.File(str(path), separate_channels=True) as f:
        return {name: np.asarray(ch.pixels, dtype=np.float64) for name, ch in f.channels().items()}

def first(path):
    return next(iter(read(path).values()))

for band, obs in m['observations'].items():
    g = obs['products']['corrected']['provenance']['geometry']
    print(f"\n[{band}] {g['width']}x{g['height']}  fy={g['intrinsics'][4]:.3f}  distortion={g['distortion']['coefficients']}")
    paths = {k: v['path'] for k, v in obs['products'].items()}
    paths.update({pathlib.PurePosixPath(p).stem.removeprefix('frame_'): p for p in obs['extra_products'] if p.endswith('.exr')})
    for name in ('measurement', 'rawdn', 'corrected', 'display'):
        s = obs['products'][name]['signal']
        print(f"  signal {name:12s} kind={s.get('kind')} unit={s.get('unit')} transfer={s.get('transfer')} cfa={s.get('camera_cfa')}")
    for name, rel in sorted(paths.items()):
        for ch, a in read(root / rel).items():
            fin = np.isfinite(a)
            if not fin.any():
                print(f'  {name:26s} {ch:10s} NO FINITE PIXELS')
                continue
            v = a[fin]
            print(f'  {name:26s} {ch:10s} min={v.min():.6g} max={v.max():.6g} mean={v.mean():.6g} median={np.median(v):.6g} finite={fin.mean():.4f}')
    if 'truth_validity' in paths:
        valid = first(root / paths['truth_validity']).astype(int)
        codes, counts = np.unique(valid, return_counts=True)
        print('  validity counts (0 miss, 1 opaque, 2 lens-invalid, 3 partial, 4 transmissive):', dict(zip(codes.tolist(), counts.tolist())))
        if 'tapp' in paths and 'truth_surface_temperature' in paths:
            tapp = first(root / paths['tapp'])
            ts = first(root / paths['truth_surface_temperature'])
            sel = (valid == 1) & np.isfinite(tapp) & (ts > 0)
            if sel.any():
                d = tapp[sel] - ts[sel]
                print(f'  tapp - T_s on validity==1: mean={d.mean():.3f} K median={np.median(d):.3f} K n={int(sel.sum())}')
EOF
```

## 判读要点

- `tapp` 是带噪信号反演的亮温（假设 ε=1），不是表面温度；`tapp - T_s` 的负偏差对高发射率材质应在数 K 以内，低发射率金属会更负（冷镜反射天空）。逐材质分解用 `truth_material_id` + `frame_truth_materials.json`。
- 红外 `display` 含场景相关拉伸，不是线性测量。
- `frame.exr` / `frame.png` 是理想针孔主渲染，和相机产品不是逐像素对应。
- 三个波段的分辨率、FOV、畸变、位姿各不相同，不要直接逐像素比较不同波段；跨波段对应用真值深度做重投影。
- 更细的 EXR 格式问题可参考 Quantiloom-dev 的 `read-exr` skill（`/mnt/h/Quantiloom-dev/.agents/skills/read-exr/SKILL.md`）。

## 汇报

给出数据集身份（dataset_id、渲染器版本、完成度）、所看样本，以及异常项（非有限像素、全平、饱和、validity 异常、温度偏差离谱）。
