---
name: port-from-paper-exp
description: Port a module (fusion baseline, metric, trainer piece) from the sibling two-band project H:/paper-exp into QuantiloomFusion's N-band interfaces, with source attribution and an equivalence check against the original. Use when a task says to reuse or port something from paper-exp.
allowed-tools:
  - read
  - grep
  - glob
---

# 从 paper-exp 移植

`/mnt/h/paper-exp` 是已完成的 VIS–IR 两波段失配融合论文工程（PyTorch + numpy + OpenCV）。本仓库按需移植其代码，**不在运行时依赖它**，也**不修改它**。

## 步骤

1. **定位源头**：`/mnt/h/paper-exp/paper_exp/{baselines,metrics,models,training,evaluation}/` 下的模块，以及 `/mnt/h/paper-exp/tests/` 里对应的测试。读 `/mnt/h/paper-exp/AGENTS.md` 中与它相关的段落——那里记录了很多踩过的坑（显存悬崖、torch 版本固定、校验点格式、指标口径）。
2. **记录来源版本**：
   ```bash
   git -C /mnt/h/paper-exp log -1 --format='%h %ad' --date=short -- <源文件路径>
   ```
3. **判断要不要搬**：与 paper-exp 的扫描/渲染/场景框定绑死的代码（`paper_exp/sweep/` 等）不搬；本仓库的数据来自数据集契约。
4. **适配接口**（`docs/work-guide.md` §3.3）：
   - 融合方法 → `fusion.FusionMethod`：输入是 `FusionInput`（任意波段子集、原生分辨率、可选 `Warp`），在 `Capabilities` 里如实声明支持的波段、是否需要预配准、是否接受异分辨率。paper-exp 的 `fuse(vis, ir)` 是两波段特例。
   - 指标 → `evaluation.Metric`：源参考类指标（Qabf、SCD 等）从"两个源"推广到"N 个源"时，要写清推广方式（逐源平均、取最大等），并在 docstring 里说明与原定义的关系。
   - torch 依赖的代码放在标注为可选的模块里。
5. **等价性检查**：对两波段特例，新实现在固定随机输入上必须复现原实现的输出。写成测试：若 `/mnt/h/paper-exp` 存在则把它加入 `sys.path` 导入原函数对比，不存在则 `skip`——这是测试期的交叉核对，不构成运行时依赖。同时移植原有单元测试。
6. **提交**（`commit` skill）：每个移植的模块一个 commit；正文写明来源，例如：
   ```
   feat: port the Laplacian-pyramid baseline to the N-band fusion interface

   Ported from paper-exp paper_exp/baselines/classical.py (a1b2c3d, 2026-09-30).
   Two-band output matches the original on a fixed random input.
   ```
   不要在 message 里放 URL。

## 不做的事

- 不改 `/mnt/h/paper-exp` 下任何文件，不往它的 `.venv` 里装包。
- 不为了迁就原接口而改动本仓库已定的边界契约；契约要改，先改 `docs/work-guide.md` §3.3。
