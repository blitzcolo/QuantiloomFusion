# QuantiloomFusion

独立于渲染器 SDK 的 VIS／MWIR／LWIR 融合成像实验工作区。

三波段环境数据集生成脚本：`tools/generate_env_dataset.py`。

- 完整参数：`configs/env_dataset.toml`
- 小样本参数：`configs/env_dataset_smoke.toml`
- 使用和实验边界：[数据集生成说明](docs/env-dataset-generation.md)
- 代码验证：`python -m unittest discover -s tests -v`

先试跑：

```powershell
python tools/generate_env_dataset.py run --config configs/env_dataset_smoke.toml --out build/env-smoke
```

再次执行同一条命令即可恢复。默认像素检查需要 Python 3.11 以上、NumPy 和 OpenEXR。
