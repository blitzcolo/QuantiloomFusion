# QuantiloomFusion

VIS / MWIR / LWIR 三波段**未配准异构传感器**融合实验工作区。渲染器（Quantiloom）是外部 Windows 程序，本仓库只负责：生成数据集（驱动渲染器）→ 读取数据集 → 几何/配准 → 融合 → 评价与消融。

开工前必读：

- `docs/work-guide.md`：干活指引与进度备忘录（目标、架构边界、任务分解、进度、决策记录）。**每完成一个任务就更新它的进度与交接段。**
- `docs/env-dataset-generation.md`：数据集生成说明、物理建模与实验边界。

## Git 规则（硬性）

- **提交身份**：author 与 committer 都必须是 `blitzcolo <19224718+blitzcolo@users.noreply.github.com>`。
  - 先看 `git config user.name` / `git config user.email`；不符时用
    `git -c user.name=blitzcolo -c user.email=19224718+blitzcolo@users.noreply.github.com commit ...`。
  - **不要修改任何 git config**（包括 `core.autocrlf`）。
  - 提交后用 `git log -1 --format='%an <%ae> | %cn <%ce>'` 核对。
- **一个 commit 只干一件事**。不要把一整个阶段或一组任务塞进一个 commit。
  - 纯重构（搬移、改名、拆模块）不改行为；改行为的 commit 不顺带搬代码；格式化单独提交。
  - 与该改动直接相关的测试和文档放在同一个 commit；顺手发现的无关问题另开 commit。
  - 能用一句话说清"这个 commit 做了什么"，才算一件事。
- **commit message**：Conventional Commits，英文祈使句，subject ≤ 72 字符，可选正文写"为什么"。
  前缀：`feat:` `fix:` `refactor:` `test:` `docs:` `chore:` `build:` `perf:`。
- **commit message 中禁止出现**：
  - "里程碑"、"milestone"、"M1/M2/…"、"阶段 N"、"Phase N"、"Step N"、任务编号（如 `T2.4`）、"进度 1/5"、"part 2 of 5"、"WIP" 之类的进度标记；
  - 对话链接、session link、任何 AI/工具会话相关的 URL 或 trailer（`Session:`、`Generated-by:`、机器人 `Co-authored-by:` 等）。
- 不 push、不改写已有历史、不 `--amend` 已存在的 commit，除非用户明确要求。
- 不提交 `build/`、数据集、EXR、模型权重、`.venv*`、本地密钥。

示例：
- 好：`refactor: move path translation into quantiloom_fusion.core.paths`
- 好：`fix: resolve the atmosphere model pack from the plan instead of a drive-letter literal`
- 坏：`feat: M2 datagen package (3/5)`、`chore: finish phase 1`、`refactor: split generator, add reader, fix docs`

## 环境与命令

工作树位于 `D:\QuantiloomFusion` = `/mnt/d/QuantiloomFusion`，Windows 与 WSL2 共用同一份文件。

| 场景 | 命令 |
| --- | --- |
| 测试（WSL） | `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/quantiloomfusion uv run pytest -q` |
| 测试（从 WSL 调 Windows） | `cmd.exe /c "set UV_PROJECT_ENVIRONMENT=.venv-win&& uv run pytest -q"`（cwd 在 `/mnt/d/QuantiloomFusion`） |
| 测试（Windows 原生 PowerShell） | `$env:UV_PROJECT_ENVIRONMENT='.venv-win'; uv run pytest -q` |
| 生成数据集 | `python tools/generate_env_dataset.py plan/run/verify ...`（只需标准库，见数据集文档） |

- **两侧虚拟环境必须分开**：WSL 用 `$HOME/.venvs/quantiloomfusion`（ext4，避免 9p 慢且不与 Windows 冲突），Windows 用仓库内 `.venv-win`。
  不设 `UV_PROJECT_ENVIRONMENT` 就跑 uv 会在仓库里建 `.venv`，另一侧再跑会把它当成坏环境重建——不要这样做。
- 两侧 uv 版本一致：均为 0.12.10（WSL 用 `~/.local/bin/uv` 独立安装版，排在 PATH 中 miniconda 自带的旧版之前）。`pyproject.toml` 的 `required-version` 拦住旧版 uv；升级 uv 要两侧同时升，再重写 `uv.lock`。改依赖后两侧都要 `uv lock --check` 通过。
- `torch` 是可选 extra（`uv sync --extra torch`）；Windows 从 cu128 索引取（PyPI 的 win_amd64 版是 CPU 版），Linux 用 PyPI。
- 渲染器：`H:/Quantiloom-SDK/windows_amd64/bin/Quantiloom.exe`（`./build_wsl.sh` 过全部门禁后安装的 SDK；当前 `--version` = `Quantiloom 0.10.0`）。0.10.0 起着色器等运行时资源只经 `share/quantiloom/` 查找，开发构建树里的 `build/src/app/Release/Quantiloom.exe` 旁边没有这个目录、在 attempt 目录里启动会报 `no Quantiloom resources found`，所以不再用它。0.7.4/0.8.3 渲染的 `build/env-*-0.7.4`、`build/env-*-0.8.3*` 只能 `verify`。需要 NVIDIA RTX + Vulkan RT。
- 一个改动在两侧都跑过测试才算验证完成（用 `cross-os-test` skill）。

## 代码边界（硬性）

包为 `src/quantiloom_fusion/`，分层与允许的依赖方向见 `docs/work-guide.md`「系统架构与边界」。要点：

- `core/` 与 `datagen/` 运行时**只用标准库**（像素 QA 与审计可在函数内惰性导入 numpy/OpenEXR）。数据生成必须能用裸 Python ≥ 3.11 跑。
- 消费端（`dataset/` `geometry/` `signal/` `registration/` `fusion/` `evaluation/` `experiments/`）**不得 import `datagen`**；它们只读发布出来的数据集契约（`dataset.index.json`、`sample.json` 及其指向的产品）。
- 只有 `datagen/renderer.py` 可以启动渲染器进程；只有 `datagen/export_record.py` 解析 Quantiloom 的 `frame.metadata.json`；消费端只有 `dataset/exr.py` 导入 OpenEXR（datagen 的像素 QA 与审计可在函数内惰性导入）。
- `torch` 只出现在模块级声明了 `__optional_deps__ = ("torch",)` 的模块里；没装 torch 时其余部分必须可导入、可测试。
- 边界由 `tests/test_boundaries.py`（AST 级 import 检查）锁定；改边界先改文档与该测试。
- 物理和渲染算法属于 Quantiloom-dev，不在这里"修"。发现物理疑点：记录到 work-guide 的待核实清单，并说明证据。

## 跨平台硬规则（Windows 原生 + WSL2 interop）

- 路径：配置里写 `H:/...` 或 `/mnt/h/...` 都可以；打开文件一律经 `native_path()`，写入计划/任务 TOML 一律用 `canonical_path()`（Windows 拼写）。
  **代码里不要写 `Path("H:/...")` 字面量**——在 Linux Python 下它是相对路径。
- 渲染器读写的位置（输出目录、场景资产）必须在 `/mnt/<盘符>/` 下；`/home/...` 对 Windows CLI 不可达。
- 经 interop 传给 Windows 进程的环境变量必须走 `WSLENV`（例：`WSLENV=QUANTILOOM_THERMAL_CACHE`）。
- 状态行/日志输出只用 ASCII（中文 Windows 控制台是 GBK，非 ASCII 字符会让进程崩溃）；文档可以写中文。
- 文本文件一律 LF（由 `.gitattributes` 保证）；生成的 JSON/TOML 用 `newline="\n"` 写出。哈希参与配方指纹的源文件在哈希前先把 CRLF 规范为 LF。
- 同一输出目录同一时间只能有一个写入者，且不能跨系统并发（Windows 与 WSL 文件锁互不可见）。
- 渲染器给 `attempts/` 设置仅创建者可读的 ACL：依赖真实数据集的测试必须由渲染它的那个 Windows 账户运行。
- 文件排序/去重按 canonical 字符串，不按 `Path` 比较（Windows 大小写不敏感）。

## 数据与复现规则

- 所有产出写在 `build/` 下（数据集、对拍、实验、权重），不要写到仓库其他位置。`build/` 是指向 `H:\QuantiloomFusion-build` 的目录 junction（见 work-guide「存储布局」）：
  只能从 Windows 用 `mklink /J` 建，不要在 WSL 里 `ln -s`；不要在数据集目录内部再套链接；删除 junction 用 Windows 的 `rmdir <链接>`，**绝不**对它 `rm -rf`。
  计划里记录的是 junction 目标路径，数据集规划后不得改指向。
- 改任何相机配置或配方开关 → 用新的 `--out` 目录。旧目录只 `verify`，不续跑。
- 正式采集（`configs/env_dataset.toml`，576 个三元组，约 7 GPU 小时）开始后，`datagen` 及其导入闭包内的 `core` 模块**冻结**：任何改动都会改变配方指纹，使目录无法续跑。
- 渲染器：凡改变数值结果的修复都必须升版本号（见数据集文档）。
- 文档中的数字必须来自实测，并写明来源（命令、目录、版本）；推断要标成"待核实"。

## Skills（`.devin/skills/`）

| Skill | 用途 |
| --- | --- |
| `commit` | 按本文件规则做一次"只干一件事"的提交 |
| `cross-os-test` | 在 WSL 与 Windows 两侧跑测试 |
| `datagen-parity` | datagen 重构时对拍：新旧实现生成的计划与任务 TOML 必须逐字节一致 |
| `datagen-smoke` | 用 smoke 配置端到端渲染 + 校验（需要 GPU） |
| `dataset-inspect` | 检查一个数据集目录与其中的 EXR 产品 |
| `progress-update` | 完成任务后更新 `docs/work-guide.md` |
| `port-from-paper-exp` | 从 `H:/paper-exp` 移植模块到三波段接口 |
