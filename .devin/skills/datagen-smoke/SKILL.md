---
name: datagen-smoke
description: Render the smoke three-band dataset end to end with the real Quantiloom CLI (GPU) into a fresh directory and verify it, including EXR inspection. Use after changes to the generator's run/publish/verify path, after a renderer upgrade, or when asked to check that dataset generation still works.
allowed-tools:
  - read
  - grep
  - glob
---

# smoke 端到端渲染

`configs/env_dataset_smoke.toml`：6 个场景条目（含 1 个夜间、1 个月光变体）各 1 个三元组，共 18 次采集，64×36 分辨率、SPP 1。0.8.3 实测约 3 分钟（`build/env-smoke-0.8.3`，热求解缓存已有时；冷缓存加上首次规划的资产哈希约 7 分钟）。只验证采集链路，数值不能当质量基准。

## 前置检查

```bash
cd /mnt/d/QuantiloomFusion
/mnt/h/Quantiloom-SDK/windows_amd64/bin/Quantiloom.exe --version   # 记录版本
tasklist.exe | grep -i quantiloom || echo "no renderer running"          # 不要和正式采集抢 GPU
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
```

有渲染进程在跑（例如正式采集）就停下来告诉用户，不要启动。

## 运行（WSL）

每次用新目录，不复用、不删除已有的数据集目录：

```bash
OUT=build/env-smoke-$(date +%Y%m%d-%H%M%S)
python3 tools/generate_env_dataset.py run --config configs/env_dataset_smoke.toml --out "$OUT"
python3 tools/generate_env_dataset.py verify --out "$OUT" --inspect-exr
```

- 给 900000 ms 以上超时，或放后台再用 get_output 轮询。
- 中断后用同一条 `run` 命令即可续跑。
- Windows 侧运行：用 Windows 解释器执行同样两条命令（生成器只需标准库；`--inspect-exr` 需要 numpy + OpenEXR）。

## 判据

1. `run` 退出码 0，输出中没有 `FAILED`；`QA warning`（RAW 饱和比例 > 5%）要记录。
2. `verify` 输出 `Verified: complete=6, partial=0, planned=6`。
3. 每 3 个完整三元组会打印一行 `CLOSURE ...`，两侧都必须给出 `closure residual` 残差行；出现 `audit unavailable` 算失败（审计依赖 numpy + OpenEXR，确认运行生成器的解释器装了它们）。
5. 要中断测试请用真正的 Ctrl+C（前台、带 PTY）。后台 `&` 启动的进程会继承被忽略的 SIGINT，`kill -INT` 对它无效。
6. 被中断后续跑的目录不要拿来做 `QUANTILOOM_TEST_DATASET`：其中 `attempt-0001` 是中断留下的残骸，导出恢复类测试会失败。用一次跑完的目录。
4. 抽查一个样本：用 `dataset-inspect` 看各波段产品统计；`samples/<id>/<band>/vNNNN/frame.png` 可直接用 read 工具看（那是理想针孔主渲染，不是相机产品）。夜间样本 VIS rawdn 中位应接近 0，月光样本 VIS rawdn 中位为个位数 DN（0.8.3 smoke 实测 5.0，0.7.4 为 6.0）。

## 汇报

写明渲染器版本、输出目录、耗时、`verify` 原文、QA 警告与 CLOSURE 行，以及抽查了什么。"命令跑完了"不是验证。
