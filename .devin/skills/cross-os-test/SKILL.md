---
name: cross-os-test
description: Run the QuantiloomFusion test suite under both the WSL2 Linux interpreter and a Windows interpreter (via interop) and report both results. Use after any code change, before committing, and whenever asked to run or check tests.
allowed-tools:
  - read
  - grep
  - glob
---

# 双系统测试

本仓库要求 Windows 原生与 WSL2 都能跑。只在一侧通过不算验证完成。

## 选择命令形式

仓库根目录有 `pyproject.toml` → 用 uv 形式；没有 → 用 unittest 回退形式。

### uv 形式

```bash
cd /mnt/d/QuantiloomFusion
# WSL（Linux 解释器，venv 在 ext4 上）
UV_PROJECT_ENVIRONMENT=$HOME/.venvs/quantiloomfusion uv run pytest -q
# Windows（经 interop；cwd 必须在 /mnt/d/QuantiloomFusion）
cmd.exe /c "set UV_PROJECT_ENVIRONMENT=.venv-win&& uv run pytest -q"
```

- **必须**设置 `UV_PROJECT_ENVIRONMENT`，两侧 venv 不同；漏设会在仓库里建 `.venv`，另一侧会把它当坏环境重建。
- 首次运行会 `uv sync`，Windows 侧可能较慢；给 600000 ms 以上超时或放后台。
- 需要 torch 的测试在未装 `torch` extra 时应跳过而非失败；要跑它们加 `--extra torch`（Windows 下载约 2.4 GB）。
- 默认不跑 `gpu` 标记的测试。要跑：`-m gpu`（需要渲染器与 RTX GPU，且不能和正在进行的正式采集抢 GPU，先查 `tasklist.exe | grep -i quantiloom`）。

### unittest 回退形式（`pyproject.toml` 落地之前）

```bash
cd /mnt/d/QuantiloomFusion
python3 -m unittest discover -s tests -v
/mnt/h/paper-exp/.venv/Scripts/python.exe -m unittest discover -s tests -v
```

第二个解释器是从 paper-exp 借来的 Windows Python 3.13.3（有 numpy + OpenEXR），只读使用，不要往里装包。基线：两侧都是 `Ran 11 tests ... OK (skipped=2)`（2026-10-11）。

## 真实数据集测试

依赖真实数据集的测试默认跳过。要跑：

```bash
# WSL 解释器：变量直接生效
QUANTILOOM_TEST_DATASET=/mnt/d/QuantiloomFusion/build/env-smoke-0.7.4 <WSL 命令>
# Windows 解释器：必须经 WSLENV 传递，/p 让 interop 翻译路径
WSLENV=QUANTILOOM_TEST_DATASET/p QUANTILOOM_TEST_DATASET=/mnt/d/QuantiloomFusion/build/env-smoke-0.7.4 <Windows 命令>
```

该数据集必须由运行测试的同一 Windows 账户渲染（`attempts/` 有仅创建者可读的 ACL）。没经 `WSLENV` 传过去时测试会被静默跳过——看 skipped 数是否变化来确认确实跑了。

## 汇报

分别给出两侧的 passed / failed / skipped 计数和失败用例名。一侧失败时先判断是不是平台差异（路径拼写、换行、文件锁、控制台编码、ACL），再修。
