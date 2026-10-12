---
name: datagen-parity
description: Prove a refactor of the dataset generator preserves behaviour by planning the same config with the original single-file generator and the current code, then comparing job TOMLs, derived solar tables and the plan byte for byte. Use after every pure-refactor commit touching tools/generate_env_dataset.py, quantiloom_fusion.core or quantiloom_fusion.datagen.
allowed-tools:
  - read
  - grep
  - glob
---

# datagen 对拍

纯重构必须让 `plan` 的产出保持不变：每个 `jobs/**/*.toml` 和 `provenance/*.csv` 逐字节相同；`dataset.plan.json` 除 `dataset_id` 与 `recipe.generator_*`（生成器自身指纹，必然不同）外完全相同。`plan` 只调用渲染器的 `--version`，不渲染，不占 GPU。

## 关键点

- 基线 = 首次把 `tools/generate_env_dataset.py` 加入仓库的 commit 里的那个文件。
- 红外任务 TOML 内嵌了 `--out` 的绝对路径（派生太阳谱），所以**两次 plan 必须写到同一个目录名**：先让基线写到 `out`，改名为 `legacy`，再让当前代码写到 `out`。
- 每次用新的时间戳目录，不删除旧的对拍目录。

## 步骤（WSL）

```bash
cd /mnt/d/QuantiloomFusion
CONFIG=configs/env_dataset_smoke.toml          # 收尾验证时再用 configs/env_dataset.toml（7776 个任务）
BASE=$(git log --diff-filter=A --format=%H -- tools/generate_env_dataset.py | tail -1)
P=build/_parity/$(date +%Y%m%d-%H%M%S)
mkdir -p "$P"
git show "$BASE:tools/generate_env_dataset.py" > "$P/legacy_generate_env_dataset.py"
python3 "$P/legacy_generate_env_dataset.py" plan --config "$CONFIG" --out "$P/out"
mv "$P/out" "$P/legacy"
python3 tools/generate_env_dataset.py plan --config "$CONFIG" --out "$P/out"
python3 - "$P/legacy" "$P/out" <<'EOF'
import json, pathlib, sys
a, b = (pathlib.Path(x) for x in sys.argv[1:3])
def files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.name.startswith('.')}
fa, fb = files(a), files(b)
problems = []
if fa != fb:
    problems.append(f'file sets differ: only legacy={sorted(fa - fb)[:10]} only current={sorted(fb - fa)[:10]}')
for rel in sorted(fa & fb):
    if rel != 'dataset.plan.json' and (a / rel).read_bytes() != (b / rel).read_bytes():
        problems.append(f'bytes differ: {rel}')
def strip(path):
    plan = json.loads(path.read_text(encoding='utf-8'))
    plan.pop('dataset_id', None)
    for key in [k for k in plan['recipe'] if k.startswith('generator_')]:
        plan['recipe'].pop(key)
    return plan
pa, pb = strip(a / 'dataset.plan.json'), strip(b / 'dataset.plan.json')
if pa != pb:
    top = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    rec = sorted(k for k in set(pa['recipe']) | set(pb['recipe']) if pa['recipe'].get(k) != pb['recipe'].get(k))
    problems.append(f'plan differs: top-level keys {top}, recipe keys {rec}')
print(f'compared {len(fa & fb)} files')
for p in problems[:50]:
    print('PARITY FAIL', p)
print('PARITY OK' if not problems else f'PARITY FAIL ({len(problems)} problems)')
sys.exit(1 if problems else 0)
EOF
```

## 跨系统对拍（收尾时做）

用 Windows 解释器重复"当前代码写到同名目录"一步，再与 WSL 的结果比较——两侧 canonical 路径相同，产出必须一致：

```bash
mv "$P/out" "$P/wsl"
<WINPY> tools/generate_env_dataset.py plan --config "$CONFIG" --out "$P/out"
# 再运行上面的比较脚本：python3 - "$P/wsl" "$P/out"
```

`<WINPY>`：`pyproject.toml` 落地后用 `cmd.exe /c "set UV_PROJECT_ENVIRONMENT=.venv-win&& uv run python tools\generate_env_dataset.py plan --config configs\env_dataset_smoke.toml --out <P 的 Windows 写法>\out"`；之前用 `/mnt/h/paper-exp/.venv/Scripts/python.exe`（生成器只需标准库）。

## 判读

- `PARITY OK` 才能提交纯重构 commit。
- 失败时先看是哪类文件不同。任务 TOML 不同 = 行为变了，回去找原因；只有 `recipe` 下非 `generator_*` 的键不同，多半是哈希范围或路径规范化被改了。
- 有意改变产出的 commit（如配方指纹规则、schema 升级）不适用本对拍，要在 commit 正文写明改变了什么，并在之后把基线换成新的参考点（在 `docs/work-guide.md` 决策记录里写下新基线 commit）。
