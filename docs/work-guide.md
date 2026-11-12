# 干活指引与进度备忘录

本文是 QuantiloomFusion 的工作底稿：要达成什么、系统怎么切分、按什么顺序做、做到哪了。规则性的东西（git、跨平台、边界）以根目录 `AGENTS.md` 为准，本文不重复。

使用方式：

1. 开工前读「当前状态」与「交接备忘」，再到「任务分解」里领一个未完成的任务。
2. 一个任务可以对应多个 commit，但每个 commit 只干一件事；任务编号（`T2.4` 等）只在本文出现，**不要写进 commit message**。
3. 完成后更新「进度表」「决策记录」「交接备忘」（`progress-update` skill）。

最后更新：2026-11-12

---

## 1. 目标

### 1.1 研究目标

本仓库服务的研究问题：在**异构原生传感器**（不同分辨率、FOV、畸变、PSF、噪声、位姿，未配准）条件下，VIS / MWIR / LWIR 三波段融合能带来什么、代价是什么。数据集提供条件，不提供结论（见数据集文档「数据集定位」）。

| 编号 | 目标 | 交付物 |
| --- | --- | --- |
| G1 | 受控基准数据集：正式配置 576 个三波段样本 / 1728 次采集（2026-11-11 前为 2592 / 7776），可断点续跑、可在 Windows 与 WSL 间交替执行、可独立校验 | `build/env-three-band-0.8.3/`（不入库）+ 发布清单 |
| G2 | 消费端数据读取：不依赖 Quantiloom 即可类型化地读取产品、信号语义、相机几何、真值 | `quantiloom_fusion.dataset` |
| G3 | 几何与配准：用真值深度 + 位姿 + 内参 + 畸变做 oracle 重投影与双向遮挡；配准方法接口与误差度量 | `geometry`、`registration` |
| G4 | 融合：N 波段融合方法接口、注册表、经典基线、未配准深度基线适配 | `fusion` |
| G5 | 评价：无参考指标、任务驱动指标、互补信息量化（VIS→T_s 残差、条件互信息）、四轴消融、配准敏感度；结果增量持久化、可复现 | `evaluation`、`experiments` |
| G6 | 工程：Windows 原生与 WSL2 interop 都能跑；一事一提交；边界由测试锁定；通用流程沉淀为 skills | `AGENTS.md`、`.devin/skills/`、`tests/` |

### 1.2 非目标

- 不在本仓库实现或修正物理/渲染算法——那属于 Quantiloom-dev；这里只记录疑点和证据。
- 数据生成端不做重采样、配准、裁剪或公共视场转换（数据集契约规定产品保持原生网格）。
- 不主张"三波段在物理上必要"；只提供能检验该主张的实验条件和度量。

---

## 2. 当前状态（2026-10-11 盘点）

### 2.1 已有且可用

| 项 | 状态 | 证据 |
| --- | --- | --- |
| `tools/generate_env_dataset.py` | 968 行单文件，`plan`/`run`/`verify` 完整；配方指纹、断点续跑、发布、校验、周期性闭合审计 | 源码 |
| `tests/test_generate_env_dataset.py` | 11 项（9 项契约 + 2 项需真实数据集，默认跳过），WSL Python 3.13.9 与 Windows Python 3.13.3 均通过 | `python -m unittest discover -s tests`，两侧均 `Ran 11 tests ... OK (skipped=2)` |
| 配置 | `configs/env_dataset.toml`（正式）、`env_dataset_pilot.toml`（全分辨率 pilot）、`env_dataset_smoke.toml`（64×36 smoke） | 文件 |
| 审计工具 | `tools/_closure_audit.py`、`_mask_audit.py`、`_spectral_integration_audit.py` | 文件 |
| 渲染器 | `Quantiloom 0.8.3`（2026-10-26 起；此前 0.7.4），RTX 4060 Ti 8 GB，驱动 617.42 | `Quantiloom.exe --version`、`nvidia-smi` |
| smoke 数据集 | `build/env-smoke-0.7.4`：6/6 完整，18 次采集约 4 分钟；`build/env-smoke-0.8.3`（0.8.3，生成器 `f2b5a22`）：6/6 完整，约 3 分钟 | `dataset.index.json`、render.log 时间戳 |
| pilot 数据集 | `build/env-pilot-0.7.4`：24/24 完整（concrete 白天/夜间/月光 × 8 方位），samples 304 MB + attempts 624 MB | 同上 |
| 场景资产 | `H:/Quantiloom-Qt/assets/configs/env_{concrete,desert,grassland,seaside}.toml` | 文件 |
| `src/`、`paper/`、`assets/` | 空目录 | `ls` |
| git | 仓库基线已入库（2026-10-12）；索引与工作树全部为 LF；工作区干净 | `git log`、`git ls-files --eol`，WSL 与 Windows git `status` 均为空 |

### 2.2 实测吞吐（决定正式采集的排期）

pilot 全分辨率（640×360 / 320×256 / 256×192，SPP 32，`wavelength_samples=12`）单次采集中位耗时：VIS 24 s、MWIR 152 s、LWIR 191 s，三元组约 6.1 分钟（24 个三元组共 2.35 小时，与之吻合）。按此外推，正式配置 2592 个三元组约 **264 GPU 小时（约 11 天连续运行）**——这是 0.7.4 的数。**0.8.3**（2026-10-26，同一批任务 TOML 的前 3 个三元组，`build/env-pilot-0.8.3-asis`）：VIS 25 s、MWIR 9.8 s、LWIR 8.8 s，三元组约 43 s，外推正式配置约 **31 GPU 小时**（待核实：同样假设各场景耗时与 concrete 相近）；样本约 33 GB，attempts 约 67 GB（`build/` 在 H 盘，剩余约 1.1 TB，见 §2.4）。外推假设各场景耗时与 concrete 相近，待正式运行头几十个样本校正。**2026-11-11 正式配置缩为 576 个三元组**（8 方位 × 俯仰 12°/25° × 3 距离 × 12 条目）：按 0.8.3 的三元组 43 s 外推约 **6.9 GPU 小时**（576 × 43 s；按 0.7.4 的 6.1 分钟则约 59 小时），样本约 7.3 GB、attempts 约 15 GB（按上面 2592 规模的体量等比折算）；**0.10.0 实测**（2026-11-11，`build/env-pilot-0.10.0`，24 个三元组 + 12 个参考样本共 108 次渲染，墙钟 716 s，热缓存已热）：各采集 `render.log` 首末时间戳中位 VIS 11.8 s、MWIR 2.8 s、LWIR 1.9 s，三元组 16.5 s；参考 `ref_vis`/`ref_mwir`/`ref_lwir` 11.6/3.2/2.7 s，一组 17.5 s。正式配置 576 个三元组约 **2.6 GPU 小时**、48 组参考（144 次渲染）约 **0.23 小时**，合计约 2.9 小时；按墙钟（108 次 716 s，含进程启动与资产加载）折算 1872 次渲染约 3.4 小时，另加 8 次热求解（4 场景 × 白天/23 时，各约一分钟级）。待核实：只测了 concrete。

### 2.3 环境

| 项 | WSL2 | Windows |
| --- | --- | --- |
| Python | miniconda 3.13.9（numpy 2.4.0、OpenEXR 3.4.13、safetensors、torch 2.9.1+cu126 可用） | 无独立安装；`python.exe` 是商店占位符。可借用 `H:/paper-exp/.venv`（3.13.3，numpy 2.5.2 + OpenEXR） |
| uv | 0.12.10（`~/.local/bin/uv` 独立安装，PATH 中排在 miniconda 自带的 0.9.17 之前；2026-10-12 统一） | 0.12.10（`C:\Users\pog\.local\bin\uv.exe`） |
| 项目 venv | `$HOME/.venvs/quantiloomfusion`（基于 miniconda CPython 3.13.9） | `.venv-win`（CPython 3.13.3） |
| git | 2.43.0，身份已是 blitzcolo | Git for Windows，`user.name=blitzcolo`，**`core.autocrlf=true`** |

### 2.4 存储布局

所有产出都写在仓库的 `build/` 下（已被 gitignore）：数据集 `build/env-*`（内含 `attempts/`、`samples/` 等，正式配置合计约 100 GB）、对拍 `build/_parity/`、实验与模型权重 `build/experiments/`（规划中）。仓库外只有小体量缓存：渲染器的 `%LOCALAPPDATA%\Quantiloom\cache`（热求解缓存 96 MB、pipeline cache 14 MB，可用 `QUANTILOOM_THERMAL_CACHE_DIR` 改位置，WSL 下需经 `WSLENV` 传递）、uv 缓存、WSL venv（`~/.venvs`，ext4，剩余 915 GB）、Windows venv（仓库内 `.venv-win`）。

现状（2026-10-12 起）：`D:\QuantiloomFusion\build` 是指向 `H:\QuantiloomFusion-build`（NTFS 机械盘）的目录 junction，与 `D:\Quantiloom-dev` → `H:\Quantiloom-dev` 同一机制（`mklink /J`）；代码和 venv 留在 D 盘（ReFS SSD）。搬迁后核对：

- `fsutil reparsepoint query` 显示挂载点 `\??\H:\QuantiloomFusion-build`；WSL 下表现为 `build -> /mnt/h/QuantiloomFusion-build`。
- 全部 375 个 `frame.metadata.json` 列出的 5543 个产品（含 0.7.3 归档）sha256 全部匹配，无缺失；`verify` 结果：pilot `complete=24/24`（WSL），smoke `complete=6/6` 带 `--inspect-exr`（WSL 与 Windows 各一次）；归档顶层 30 项与搬迁前一致。
- 归档目录的占用空间从 842 MiB（D 盘 ReFS 上的 du）变为 761 MiB（H 盘 NTFS，实际字节 738 MiB），pilot/smoke 基本不变。H 盘上两者的单文件平均富余都是约 2 KB（4 KB 簇的预期值），差异应来自 ReFS 的空间记账；日志、脚本等非产品文件没有内嵌哈希，无法逐个比对。

机制在搬迁前（2026-10-11）已在临时仓库中实测：

- 生成器对 `--out` 做 `resolve()`：WSL 得到 `/mnt/h/<目标>/...`，Windows 得到 `H:\<目标>\...`，`canonical_path` 两侧都是 `H:/<目标>/...`——计划与任务 TOML 记录的是 **junction 的目标**，两侧一致，跨系统交替执行不受影响。
- 单写者锁（WSL `flock` / Windows `msvcrt`）、`atomic_json`、同目录 `os.replace` 两侧都正常。
- WSL git 把 junction 看成符号链接，`.gitignore` 的 `build/` 不再生效，需改为 `/build`（T0.2）。搬迁后真实仓库复现了 `?? build`，已改为 `/build`（`0c79007`）。

约束：junction 要在正式采集规划前建好，之后不要改指向（等同于搬迁数据集，`run` 会拒绝，`verify` 不受影响）；只在 `build/` 这一层做链接，不在数据集目录内部再套链接；`build/` 下现有目录（pilot、smoke、0.7.3 归档，共 1.9 GB）规划时记录的是 `D:/QuantiloomFusion/build/...`，搬迁后只能 `verify`，不能续跑（它们都已完成）。训练/评价从机械盘随机读大量 EXR 可能受 IO 限制，见 T3.9。

### 2.5 已发现的问题（按影响排序）

1. ~~**审计工具在 WSL 下不可用**：`_closure_audit.py`、`_spectral_integration_audit.py` 把模型包写成 `Path("H:/Quantiloom-dev/assets/atmos_models")`，Linux Python 下这是相对路径。生成器在 WSL 下运行时的周期性闭合审计因此必然报 "audit unavailable"。~~ 已修（T2.14）：闭合审计从 `capture.toml` 的 `atmosphere.model_pack` 取模型包，光谱积分审计默认值经 `native_path`。
2. ~~**换行符风险**：`.gitattributes` 为空，而 Windows 侧 `core.autocrlf=true`。若由 Windows git 检出，源文件变成 CRLF，`generator_sha256` 随之改变，"同一数据集目录跨系统交替执行"失效。~~ 已由 `1fae36e`（`* text=auto eol=lf`）解决。
3. **生成器指纹只覆盖单文件**：包化后必须改成覆盖 datagen 的导入闭包，否则拆出去的模块改了也不会触发拒绝续跑。
4. ~~**interop 下中断渲染**：WSL 下 `child.terminate()` 作用于 Linux 侧 interop 代理进程，Windows 侧 `Quantiloom.exe` 是否随之退出未验证（待核实）；`alive(child_pid)` 在 WSL 下检查的也是代理进程。~~ 已实测（T2.16）：WSL 下真正的 Ctrl+C 后 Windows 侧 `Quantiloom.exe` ≤1 s 退出，两次一致，状态记为 `interrupted`，锁文件释放，续跑完成。
5. ~~数据集文档与现状不符：测试数写作"10 项"（实为 11）；证据文件写作 `docs/env-dataset-generation-evidence-2026-10-06.json`（实际在 `build/`，被 gitignore）；使用示例目录为 `build/env-smoke`、`build/env-three-band`（现有为 `*-0.7.4`）；pilot 配置未被文档提及。~~ 已由 `61341e0`、`81d5286`、`f36ed47`、`0d29c43` 修正（T0.4）。
6. `sample.json` 的 `extra_products` 以文件路径为键而非产品名，消费端只能按文件名反查产品名（可在 sample schema v2 中补 `truth` 映射，需单独决策）。另外 `frame_truth_material_id.exr` 的通道名是 `instance_id`（与实例 ID 文件同名），读取器必须按文件而不是按通道名区分两者（pilot 实测）。
7. ~~`.clang-tidy`、`.clangd` 是 Quantiloom C++ 仓库的配置，与本仓库无关。~~ 已加入 `.gitignore`（`460a57a`）。
8. ~~**跨系统重规划失败**（T2.15 发现，重构前就存在）：Windows 与 Linux libm 末位差异使正式配置 549 个任务 TOML 在另一系统上逐字节不同，`plan` 报 "planned file changed"，正式数据集无法跨系统续跑。~~ 已由 `aa07475` 修正：同 `dataset_id` 重规划时保留已存计划与任务文件，容差比对。
9. ~~物理待核实：pilot LWIR 元数据 diagnostics 含 "material 'Car_Car_body' legacy scalar IR fractions are constant assumptions extended across camera response [4000, 12000] nm"，而数据集文档称车身绑定 Al n,k。可能只是热求解边界参数的提示，也可能说明 n,k 绑定未生效——需在 Quantiloom 侧核对，不在本仓库修。~~ 已关闭（2026-11-11）：Quantiloom-Qt 当前的 `env_*.toml` 里车身绑定的是白色涂层实测谱（`White rubberized coating (0795UUURBR)`，LWIR ε≈0.92，热求解 `ir_emissivity = 0.92`），根本不是 Al 的 n,k；文档按 Qt 配置改正，Al n,k 只在裸铝部件 `Car_M_IRON.004` 上。
10. **已关闭（2026-11-12）**：生成器对全部条目从 forcing 行推导大气气温与湿度（`atmosphere_air = "forcing_row"`，见数据集文档"大气"节）。原记录：Qt 白天场景的 NN 湿度与其 forcing 不一致——concrete 不写 `rh`（`clear` 预设值 0.50）而 `concrete_day.csv` 14 时湿度 30.4 %；desert 0.15 对 11 时 11.0 %、grassland 0.50（预设）对 10 时 54.1 %、seaside 0.70 对 16 时 68.3 %（2026-11-11 按 Qt `406deb8` 之后的文件重读）。热求解读 forcing 的湿度列，只有 NN 大气用 `rh`，所以大气与热平衡描述的不是同一团空气。场景文件属 Quantiloom-Qt，本仓库不改；要统一，可在配置的 `overrides.atmosphere` 里写全白天大气段（改配方，需用户决定）。
11. `tests/test_dataset_splits.py::test_real_dataset_single_shared_group` 假设 `QUANTILOOM_TEST_DATASET` 只有一个 `split_group`（pilot），而文档与 skill 指定的测试数据集是 smoke（4 组）：用 `build/env-smoke-0.8.3` 与 `build/env-smoke-0.7.4` 都失败（`assert 4 == 1`）。是测试的既有假设问题，与 0.8.3 无关，待修。
12. 渲染器待核实（属 Quantiloom-dev）：太阳在地平线下（`sun_direction` 的 y < 0）时，带 `solar_lut` 的光谱模式仍按太阳表的天空项照明——不缩放的 G-173 让 23 时的 concrete VIS rawdn 中位从 0 变成 23 DN（`build/_qa/night-lut-0.10.0`）。本数据集用 1e-12 缩放绕开（见决策记录），未在渲染器里改。

---

## 3. 系统架构与边界

### 3.1 数据流

```text
configs/*.toml ──► datagen.planner ──► jobs/*.toml ──► datagen.renderer ──► Quantiloom.exe (Windows)
                         │                                    │
                         ▼                                    ▼
                 dataset.plan.json                 attempts/.../frame.metadata.json + EXR
                                                              │ datagen.export_record (校验)
                                                              ▼
                                   samples/<id>/sample.json + dataset.index.json   ← 数据集契约（边界 B3）
                                                              │
════════════════════════ 生产端 / 消费端分界：以下模块不得 import datagen ════════════════════════
                                                              ▼
              dataset.reader ──► signal (归一化) ──► registration (Warp) ──► fusion ──► evaluation ──► results.jsonl
                     │                                   ▲
                     └────────► geometry (相机模型、oracle 重投影、遮挡) ┘
```

### 3.2 包布局与依赖方向

src 布局，导入名 `quantiloom_fusion`。箭头表示"可以导入"：

| 子包 | 职责 | 允许依赖 | 运行时第三方依赖 |
| --- | --- | --- | --- |
| `core` | 主机识别、路径翻译、哈希/指纹、原子 JSON、TOML 写出、单写者锁、纯 Python 位姿数学、波段常量 | — | 无（标准库） |
| `datagen` | 配置校验、场景组装、传感器配置、太阳谱派生、计划、渲染器适配、导出记录校验、运行/发布/索引、审计、CLI | `core` | 无；像素 QA、审计在函数内惰性导入 numpy/OpenEXR/safetensors |
| `dataset` | 数据集契约的严格解析与类型化读取、EXR IO、按 `split_group` 分折、可选 torch 适配 | `core` | numpy、OpenEXR；torch 可选 |
| `geometry` | Brown–Conrady 相机模型、像素网格、oracle 重投影、双向 z 测试、边缘带与 validity 掩码 | `core`、`dataset` | numpy |
| `signal` | 产品信号 → 融合输入的显式归一化策略（VIS 去马赛克/显示、IR 动态范围） | `core`、`dataset` | numpy |
| `registration` | 配准方法接口与实现、配准误差度量 | `core`、`dataset`、`geometry`、`signal` | numpy；torch 可选 |
| `fusion` | 融合方法接口、能力声明、注册表、基线 | `core`、`dataset`、`geometry`、`signal`、`registration` | numpy；OpenCV/torch 按需 |
| `evaluation` | 指标、互补信息分析、结果存储 | 以上全部（除 `datagen`、`experiments`） | numpy；scipy/torch 按需 |
| `experiments` | 实验配置（TOML）解析与驱动 | 以上全部（除 `datagen`） | 同上 |

硬约束由 `tests/test_boundaries.py` 用 AST 检查：消费端不 import `datagen`；`core`/`datagen` 模块顶层不 import 第三方包；只有 `datagen/renderer.py` 使用 `subprocess`；消费端只有 `dataset/exr.py` 导入 `OpenEXR`/`Imath`（`datagen/` 只能在函数内惰性导入）；`torch` 只出现在模块级声明了 `__optional_deps__ = ("torch",)` 的模块中；包内不得出现 `Path("H:/...")` 这类盘符字面量；不在上表中的子包（含包根下的散落模块）直接报错，新增子包先改上表与测试的分层表。

`tools/generate_env_dataset.py` 保留为薄壳：把 `<repo>/src` 插入 `sys.path` 后调用 `quantiloom_fusion.datagen.cli.main`，保证裸 Python 不安装也能生成数据集，且文档里的命令不变。

### 3.3 边界契约

以下是框架阶段要落地的接口形状（签名级），实现阶段只往里填，不改形状；要改形状先改本节。

**B1 渲染器边界**（`datagen/renderer.py`，唯一启动渲染器的地方）

```python
class RenderProcess(Protocol):
    pid: int
    def wait(self) -> int: ...
    def terminate(self, timeout_s: float = 10.0) -> None: ...

class Renderer(Protocol):
    def version(self) -> str: ...                      # "Quantiloom 0.8.3"，配方判定用
    def start(self, job_toml: Path, workdir: Path, log: IO[str]) -> RenderProcess: ...

class QuantiloomCli(Renderer): ...   # Windows 原生直接调用；WSL 经 interop，参数传 canonical（Windows）路径
class FakeRenderer(Renderer): ...    # 测试用：写出合成的 frame.metadata.json + 小 EXR，使 run/publish/verify 无需 GPU 可测
```

**B2 Quantiloom 导出记录边界**（`datagen/export_record.py`，唯一解析 `frame.metadata.json` 的地方；schema `quantiloom.dataset.export` v1/v2）

```python
def read_export_record(path: Path) -> ExportRecord
def verify_capture(record_path: Path, camera: dict, inspect_pixels: bool = False) -> CaptureSummary
```

**B3 数据集契约**（类型已落地于 `dataset/types.py`：`Sample`、`SampleParameters`、`Observation`、`Product`、`CameraGeometry`、`SignalSemantics`、`Condition` + `condition_of()`；`Band` 在 `core/bands.py`，生产端与消费端共用）（生产端写、消费端读；消费端只依赖这些文件）

| 文件 | schema | 消费端用途 |
| --- | --- | --- |
| `dataset.index.json` | `quantiloomfusion.env_dataset` v1 | 完整样本列表、`split_group`、完成度；每条另有布尔 `references`（共视点参考三幅都已发布；缺省读作 false，故未升版本） |
| `samples/<id>/sample.json` | `quantiloomfusion.sample` v2（v1 仍可读，视为无参考） | 每波段 `observations`：产品路径、`signal`（kind/unit/transfer/cfa）、`provenance.geometry`（内参、畸变 `k1_k2_p1_p2_k3`、`camera_to_world`、`right_down_forward`、像素中心偏移 0.5、左上原点）、`extra_products`（真值）；v2 另有 `references`：`{}`，或三幅都发布后 `{vis, mwir, lwir}`，结构与 `observations` 相同，外加 `geometry_from = "vis"`、`psf_sigma_px`（共视点无噪声参考，见数据集文档） |
| `dataset.plan.json` | `quantiloomfusion.env_dataset` v1 | 配方、场景假设、相机固定配置（只读，用于溯源与消融配对） |

`dataset_id` 标识配方；消融臂之间按 `sample_id` 配对（同场景/方位/俯仰/距离的 `sample_id` 在不同配方中相同）。schema 版本变化必须同时改生产端、消费端解析器与本表。

**B4 信号归一化**（`signal/`）

```python
class Normalizer(Protocol):
    name: str
    product: str                            # 消费的产品："display" / "rawdn" / "corrected" / "tapp"
    def params(self) -> dict: ...           # 上一次调用实际应用的参数，写入结果
    def __call__(self, image: np.ndarray, observation: Observation) -> np.ndarray: ...  # float32
```

`measurement`/`rawdn`/`corrected` 量纲不同；红外 `display` 含场景相关拉伸，不当线性测量用。归一化策略必须显式选择并记录。已有实现：`VisDisplay`（VIS `display`，sRGB 编码原样保留）、`VisRawLinear`（VIS `rawdn`：黑电平 → rggb 去马赛克 → 白平衡 + `device_to_linear_srgb` → adc 固定尺度或百分位尺度，输出线性 RGB）、`IrLinear`（IR `corrected`：每图百分位或固定窗）、`IrTemperature`（IR `tapp`：K 温窗）。`normalized_input(dataset, observation, normalizer)` 负责取产品并调用。VIS 输入可在 `display` 与线性 RAW 之间切换（用户决策）；颜色约定为 BT.601 full-range YCbCr（Y∈[0,1]，Cb/Cr 以 0.5 为中心）：融合只融 Y，彩色输出 = 融合 Y + VIS 的 CbCr。

**B5 几何**（`geometry/`；坐标约定：像素索引，像素中心为整数，见 T3.6）

```python
class BrownConradyCamera:
    @classmethod
    def from_geometry(cls, g: CameraGeometry) -> "BrownConradyCamera": ...
    def project(self, points_cam: np.ndarray) -> np.ndarray: ...            # (N,3) → (N,2) 像素坐标
    def unproject(self, pixels: np.ndarray, z: np.ndarray) -> np.ndarray: ...

@dataclass(frozen=True)
class Warp:
    ref_band: Band; src_band: Band
    map_xy: np.ndarray        # (H_ref, W_ref, 2)：参考网格每个像素在源网格中的坐标
    valid: np.ndarray         # (H_ref, W_ref) bool：可对应（排除未命中、遮挡、边缘带、validity 3/4）
    provenance: dict

@dataclass(frozen=True)
class DepthView:
    geometry: CameraGeometry
    camera_z: np.ndarray        # (H, W) truth_camera_z
    world_position: np.ndarray  # (H, W, 3) truth_world_position (X, Y, Z)
    validity: np.ndarray        # (H, W) truth_validity

def load_depth_view(dataset: ThreeBandDataset, observation: Observation) -> DepthView
def oracle_warp(ref: DepthView, src: DepthView, *, ref_band: Band, src_band: Band,
                edge_px: int = 1, edge_rel_tol: float = 0.02, pos_rel_tol: float = 0.005) -> Warp
def oracle_diagnostics(ref: DepthView, src: DepthView, *, edge_px: int = 1,
                       edge_rel_tol: float = 0.02, pos_rel_tol: float = 0.005) -> dict[str, int]
def apply_warp(src_image: np.ndarray, warp: Warp, interp: str = "linear") -> np.ndarray
```

**B6 配准**（`registration/`）

```python
class Registrar(ABC):
    name: str
    def estimate(self, ref: Observation, src: Observation,
                 ref_image: np.ndarray, src_image: np.ndarray) -> Warp: ...

def registration_error(estimate: Warp, oracle: Warp) -> dict[str, float]   # EPE 均值/中位/P95（像素）
```

首批实现：`OracleRegistrar`（上界）、`ResizeRegistrar`（只按 FOV 缩放对齐，下界）、`InfinityHomographyRegistrar`（已知 K/R 的无穷远单应，忽略视差——基线 3.5–8 cm 下近处必然出错，正好量化视差影响）。

**B7 融合**（`fusion/`）

```python
@dataclass(frozen=True)
class Capabilities:
    bands: frozenset[Band]            # 支持的波段子集（波段消融用）
    needs_registration: bool          # True：输入已按 Warp 重采样到参考网格
    heterogeneous_resolution: bool
    color_output: bool
    trainable: bool

@dataclass
class FusionInput:
    reference: Band
    images: dict[Band, np.ndarray]    # 原生分辨率、已归一化
    warps: dict[Band, Warp] | None
    geometry: dict[Band, CameraGeometry]

@dataclass
class FusionOutput:
    image: np.ndarray                 # 参考网格
    extras: dict

class FusionMethod(ABC):
    name: str
    capabilities: Capabilities
    def fuse(self, x: FusionInput) -> FusionOutput: ...

def register(name: str): ...          # 装饰器
def create(name: str, **config) -> FusionMethod
```

实现：`fusion/align.py` 的 `aligned_sources(x)` 把各波段重采样到参考网格（参考波段原样，其余经 `apply_warp` 线性插值；warp 无效处用参考图像值填充，使融合退化为参考），同时返回各波段 support 掩码供指标使用；`colourize(fused_y, vis_rgb)` 把融合 Y 与 VIS 的 Cb/Cr 合回 RGB。`fusion/classical.py` 从 paper-exp 移植经典基线并推广到 N 波段，注册名 `average`（可选 `weights` 按波段加权，未指定的均分剩余权重）、`max_abs`、`laplacian_pyramid`（各级取 |系数| 最大者，基准层取均值）；三者 `needs_registration=True`，输出 `extras["support"]` 携带各波段掩码。

**B8 评价**（`evaluation/`）

```python
class Metric(ABC):
    name: str; version: int
    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]: ...

class ResultStore:   # 追加写 JSONL；键 = (dataset_id, sample_id, method 指纹, metric 名, metric 版本)；已有则跳过
    def has(self, key) -> bool: ...
    def append(self, key, values: dict) -> None: ...
```

指标从 paper-exp 移植：`evaluation/no_reference.py`（EN/SD/SF/AG，0–255 显示尺度，支持区域掩码）与 `evaluation/full_ref.py`（PSNR/SSIM）逐字搬运；`evaluation/source_ref.py` 中 `scd`/`qabf`/`ms_ssim_fusion` 逐字搬运，另有我们的 N 源推广 `scd_n`/`qabf_n`/`ms_ssim_fusion_n`（双源输入时与原式相等，docstring 注明属自行推广而非已发表定义）。`evaluation/metrics.py` 提供 `Metric` 包装类 `Entropy/StandardDeviation/SpatialFrequency/AverageGradient/SCD/Qabf/MsSsim`（name 取 `EN/SD/SF/AG/SCD/Qabf/MS_SSIM`，`version=1`），统一在融合输出 Y（BT.601 亮度）上评分；评价区域 = 所有波段 support 的交集，源参考指标另把各波段 support 传给 `*_n`（某波段仅在自身足迹内参与）。

实验配置用 TOML（与 `configs/` 一致，标准库 `tomllib` 双系统可读），放 `configs/experiments/`；输出到 `build/experiments/<name>/`。

### 3.4 datagen 配方指纹的新规则

包化后 `recipe.generator_sha256` 改为：从 `quantiloom_fusion.datagen` 出发，用 AST 静态计算其在 `quantiloom_fusion.*` 内的**导入闭包**，对闭包内每个 `.py`（CRLF 规范为 LF 后）求 sha256，按 canonical 相对路径排序后取指纹；同时写出 `recipe.generator_files` 清单便于诊断。`datagen/audit/` 不进指纹（审计不影响产品）。效果：改了 datagen 用到的 `core` 模块会拒绝续跑；改消费端代码不会。

---

## 4. 技术选型

| 项 | 选择 | 理由 / 备注 |
| --- | --- | --- |
| 语言 | Python ≥ 3.11（`.python-version` 定 3.13） | `tomllib`；两侧现有解释器均为 3.13 |
| 项目管理 | uv + `pyproject.toml` + `uv.lock`，src 布局 | 两侧均已装 uv；需先统一版本 |
| 依赖分层 | 主依赖 numpy、OpenEXR(≥3.2)；可选 `torch`；dev 组 pytest | 生成端不需要任何依赖；消费端最小集可在无 GPU 机器上运行 |
| 深度学习 | `torch==2.9.1`，可选 extra，固定版本；Windows 经 `[tool.uv.sources]`（`sys_platform == 'win32'`）取 cu128 索引，Linux 用 PyPI（自带 CUDA） | PyPI 的 win_amd64 torch 是 CPU 版（paper-exp 的教训）；与 paper-exp 的显存测量同版本、同索引 |
| 图像 | OpenEXR 官方绑定读写 EXR；OpenCV-headless 在经典基线需要时引入 | OpenCV wheel 不带 EXR 支持 |
| 测试 | pytest；标记 `gpu`（需渲染器）、`real_dataset`（需 `QUANTILOOM_TEST_DATASET`）、`windows_only`；默认不跑 `gpu` | 现有 unittest 测试可直接被 pytest 收集 |
| 配置 | TOML | 与现有配置一致 |
| 复用 | 按需从 `H:/paper-exp` 移植（融合 ABC、经典基线、指标、训练器），适配 N 波段接口，commit 正文注明来源 | 不在运行时依赖 paper-exp |
| commit | Conventional Commits，英文 | 与 Quantiloom-dev、Quantiloom-Qt、paper-exp 一致 |

---

## 5. 任务分解

每个任务写明"完成判据"。标 ⚠ 的任务改变数据集配方指纹或产出格式，必须单独提交并在正文说明。

### 阶段 0：仓库基线（只整理，不改行为）

- **T0.1** `.gitattributes`：`* text=auto eol=lf`；`*.exr *.png *.glb *.bin *.safetensors binary`；`*.bat text eol=crlf`。判据：WSL 与 Windows git 下 `git status` 都干净。
- **T0.2** `.gitignore`：`build/` 改为 `/build`（`build` 是 junction 时，WSL git 把它当符号链接，`build/` 模式不匹配，会显示 `?? build`；Windows git 不受影响——2026-10-11 在临时仓库中实测）；补 `.venv*/`、`.pytest_cache/`、`.clang-tidy`、`.clangd`；`assets/`、`paper/` 保持忽略。
- **T0.3** 首次入库，按"每类一个 commit"：三个配置 → 生成器+其测试（测试读取正式配置，故配置先入库） → 审计工具 → 数据集文档 → README → `AGENTS.md` → 本文 → 每个 skill 各一个。**生成器首次入库的 commit 是对拍基线**（`datagen-parity` 用 `git log --diff-filter=A` 找它）。
- **T0.4** 修正 §2.5 第 5 条的文档不符，每处一个 commit。

### 阶段 1：工程骨架（定接口与边界，空实现）

- **T1.1** 统一两侧 uv 版本；`pyproject.toml`（依赖分层、torch 索引、pytest 配置与标记）、`.python-version`、`uv.lock`。判据：两侧 `uv sync` 成功，`uv run pytest` 收集到现有 11 项且通过（2 skip）。
- **T1.2** 包骨架：§3.2 的各子包只含 `__init__.py` 与职责 docstring。
- **T1.3** `tests/test_boundaries.py`：实现 §3.2 末尾的 AST 规则。判据：故意加一条违规 import 时测试失败。
- **T1.4** 接口模块（每个边界一个 commit）：`dataset` 的类型（`Band`、`CameraGeometry`、`SignalSemantics`、`Observation`、`Sample`）、`geometry.Warp`、`signal.Normalizer`、`registration.Registrar`、`fusion`（`Capabilities`/`FusionInput`/`FusionOutput`/`FusionMethod`/注册表）、`evaluation`（`Metric`/`ResultStore`）。只有签名、dataclass 与 `NotImplementedError`，外加形状层面的测试。
- **T1.5** 把 `cross-os-test` skill 的命令从 unittest 回退形式切换为 uv 形式（若 T1.1 后有变化）。

### 阶段 2：datagen 包化（行为保持）

前置：T0.3 完成（有对拍基线）、T1.1 完成。T2.1–T2.11 每个都是纯重构，**每个 commit 后跑 `datagen-parity`（smoke 配置，WSL）**，要求任务 TOML 与派生太阳谱逐字节一致、`dataset.plan.json` 除 `dataset_id` 与 `recipe.generator_*` 外完全一致；同时跑 `cross-os-test`。

- **T2.1** `core.host` + `core.paths`（`native_path`、`canonical_path`、可达性判断）
- **T2.2** `core.hashing`、`core.jsonio`、`core.tomlio`
- **T2.3** `core.locking`（`runner_lock`、`alive`）
- **T2.4** `core.vec`（纯 Python 位姿数学）与 `core.bands`（`Band`/`BANDS` 已由 T1.4 建好，此处把生成器的 `BANDS`、`MODES` 改为引用它；`Band` 是 StrEnum，格式化结果与原字符串相同，对拍应不受影响）
- **T2.5** `datagen.settings`（`read_settings` 与校验）
- **T2.6** `datagen.scene`（资产解析、overrides、运动拒绝、glTF 读取、n,k 覆盖、`scene_for_band`）、`datagen.solar`、`datagen.sensor`（三个 commit）
- **T2.7** `datagen.renderer`：`QuantiloomCli.version()` 与 `start()`；`plan` 与 `run` 改为经它调用
- **T2.8** `datagen.export_record`（`safe_path`、`check_expected`、`verify_capture`）
- **T2.9** `datagen.planner`
- **T2.10** `datagen.runner`、`datagen.index`
- **T2.11** `datagen.cli` + `tools/generate_env_dataset.py` 薄壳；测试改为导入包
- **T2.12** ⚠ 配方指纹改为导入闭包规则（§3.4）。判据：改闭包内 `core` 模块 → 拒绝续跑；改 `dataset/` 模块 → 不受影响；两侧算出的 `dataset_id` 相同。
- **T2.13** `FakeRenderer` + 无 GPU 的 run/publish/恢复/损坏检测/verify 测试
- **T2.14** 审计工具修复：模型包路径从 `capture.toml` 的 `atmosphere.model_pack` 经 `native_path` 解析。审计保留为 `tools/` 下的独立脚本、由生成器以子进程调用（见决策记录）。判据：WSL 下对 pilot 的一个 LWIR attempt 跑通闭合审计。
- **T2.15** 收尾验证：两侧对 smoke 与正式配置做全量对拍；用新包实跑 smoke（`datagen-smoke`），`verify --inspect-exr` 得到 `complete=6`、`planned=6`；更新数据集文档中的命令与指纹描述。
- **T2.16** 验证 interop 中断行为（§2.5 第 4 条）：WSL 下 Ctrl+C 后检查 `tasklist.exe` 中是否残留 `Quantiloom.exe`；若残留，在 `QuantiloomCli` 中补 Windows 侧终止方案。

### 阶段 3：数据读取与几何真值

- **T3.1** `dataset.exr`：读通道为 ndarray、读 header 属性。
- **T3.2** `dataset.schema`：index/sample/plan 子集的严格解析与版本检查。
- **T3.3** `dataset.reader`：`ThreeBandDataset` / `Sample` / `Observation`；产品名 ↔ 文件名映射（数据集文档「产品清单」）；工况（day/night/moonlight）由 `scene_id` 后缀在单一函数中推导（是否在配置中显式化待决）。
- **T3.4** 合成夹具数据集构建器（微型 EXR + sample.json），消费端测试不依赖真实数据。
- **T3.5** `dataset.splits`：按 `split_group` 留一组交叉验证；夜间/月光变体与白天共享 glTF，因而同组，不会泄漏。
- **T3.6** `geometry.camera`：Brown–Conrady 投影/反投影。判据（在 pilot 上实测）：用 `truth_world_position` 经 `world_to_camera` 和 `project` 回投到本相机，应落在像素中心；先测出残差分布再定阈值并写入测试。
- **T3.7** `geometry.reprojection`：`oracle_warp`（双向 z 测试、边缘带、validity 3/4 与 0 排除）。判据：VIS→LWIR 用 warp 采样的 `truth_world_position` 与 LWIR 自身的 `truth_world_position` 在有效区一致（阈值实测后定）。
- **T3.8** 可选 torch `Dataset` 适配。
- **T3.9** 评估 WSL 经 9p 读 `/mnt/d` 的吞吐；若训练数据加载成为瓶颈，再决定 Windows 原生训练或本地缓存。

### 阶段 4：信号、配准、融合、评价的首批实现

- **T4.1** 归一化策略。VIS 两种输入都实现、按实验配置切换：`display`（sRGB，已含色调映射）与线性 RAW（RGGB 去马赛克）。融合输出同时给彩色与灰度：先在亮度通道融合，灰度输出即融合后的 Y，彩色输出把 VIS 的色度合回去；指标统一在 Y 上算，彩色版另给色彩保真度指标。红外不用 `display`（含场景相关拉伸）。
- **T4.2** `OracleRegistrar`、`ResizeRegistrar`、`InfinityHomographyRegistrar` 与 `registration_error`。
- **T4.3** 从 paper-exp 移植经典基线并推广到 N 波段（`port-from-paper-exp`）。
- **T4.4** 移植指标（EN、SD、SF、AG、SCD、Qabf、MS-SSIM、PSNR、SSIM），源参考类指标推广到多源。
- **T4.5** `ResultStore` + `experiments` 驱动 + 实验 TOML；第一次端到端：pilot 上「波段子集 × 经典基线 × oracle/naive 配准」。
- **T4.6** 互补信息量化：VIS → `truth_surface_temperature` 回归残差（留一场景组，报告折数与组构成）；条件互信息 I(Y; LWIR | VIS)。口径按数据集文档「评价协议建议」。
- **T4.7** 任务驱动评价（基于实例/材质 ID 的分割）——先写设计再实现。

### 阶段 5：正式采集与消融渲染（GPU 长任务）

- **T5.1** 冻结 datagen（记录冻结时的 commit），正式运行 `configs/env_dataset.toml` → `build/env-three-band-0.8.3`；定期 `verify`；每 50 个三元组的闭合审计结果汇总进本文。
- **T5.2** 消融臂配置：`ir_materials="constant"`、`thermal_mode="authored"`，在 1 个俯仰角 × 8 个方位角子集上（配置内容需用户确认）。
- **T5.3** 可选：同 `job_sha256` 的 VIS 采集跨消融臂复用（VIS 任务 TOML 在这两个开关下逐字节一致）。
- **T5.4** 发布打包（index、plan、state、samples）；按 Datasheets for Datasets 拆分数据集文档。

### 阶段 6：深度方法与论文实验（细化留到阶段 4 之后）

- 未配准深度基线适配（RFNet、MURF、UMF-CMGR 等）；自有方法；四轴消融表；配准误差 → 融合指标敏感度曲线。

---

## 6. 进度表

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 调研与本文、`AGENTS.md`、skills 初稿 | 完成 | 已入库（`0b225f4`、`abfa13c`、`1c2e5ee`…`9e8ad20`） |
| T0.1 | 完成 | `1fae36e`；WSL 与 Windows（`core.autocrlf=true`）git `status` 均干净 |
| T0.2 | 完成 | `0c79007`：`/build`、`.venv*/`、`.pytest_cache/`；`460a57a`：`.clang-tidy`、`.clangd`；`assets/`、`paper/` 保持忽略 |
| T0.3 | 完成 | 配置 `5622f5a` → 生成器与测试 `bad41d3`（对拍基线）→ 审计 `9d2c4b6` → 文档、README、`AGENTS.md`、本文、7 个 skill 各一个 commit；提交后两侧 unittest 均 `OK (skipped=2)` |
| T0.4 | 完成 | 四处各一个 commit：测试数 11（`61341e0`）；证据文件与旧目录的实际位置（`81d5286`，旧 `*-ready` 目录在 `build/_archive-0.7.3-buggy/`）；示例目录改为 `*-0.7.4` 并说明命名与只能 verify 的现有目录（`f36ed47`）；pilot 配置说明（`0d29c43`，与正式配置的差异经 `tomllib` 逐键比对：仅名称、采样网格、`closure_audit_every=8`） |
| T1.1 | 完成 | `fc6f11b`：`pyproject.toml`（uv_build、numpy+OpenEXR、torch extra、dev 组 pytest、`required-version >=0.12.10`、pytest 标记）、`.python-version`、`uv.lock`、`src/quantiloom_fusion/__init__.py`（uv_build 需要包存在）。两侧 `uv run pytest -q` 均 `9 passed, 2 skipped`；两侧 `uv lock --check` 通过；lock 中 torch 为 Linux `2.9.1`（PyPI）/ Windows `2.9.1+cu128`；torch extra 未 sync |
| T1.2 | 完成 | `0be6121`：9 个子包，各只有职责 docstring |
| T1.3 | 完成 | `d78efb5`：`tests/test_boundaries.py`，`check_tree()` + 真实包检查 + 每条规则的反例测试；故意在 `fusion/__init__.py` 加 `import quantiloom_fusion.datagen` 时报 `fusion/__init__.py:2: fusion must not import quantiloom_fusion.datagen`。两侧 `uv run pytest -q` 均 `20 passed, 2 skipped` |
| T1.4 | 完成 | 每个边界一个 commit：`a00edc0` `core.bands.Band`（StrEnum，与生成器 `BANDS` 逐值相同）→ `6307aeb` `dataset.types` → `039a772` geometry → `42ab89c` signal → `bef38e0` registration → `916f1cf` fusion（含已实现的注册表与 `check_input`）→ `1f91daf` evaluation。两侧 `uv run pytest -q` 均 `60 passed, 2 skipped`；导入全部消费端子包后 `torch` 不在 `sys.modules` |
| T1.5 | 完成 | `1ae708b`；`AGENTS.md` 同步（`266a9df`） |
| T2.1–T2.11 | 完成 | 17 个 commit（`1dbe826`…`eb94380`）：core（paths、hashing/jsonio/tomlio、locking、vec）→ datagen（constants、settings、scene、solar、sensor、renderer、export_record、planner、publish、index、runner+verify、cli）；`tools/generate_env_dataset.py` 现为 11 行薄壳。**每一步**都在 smoke 配置上 `PARITY OK`（目录 `build/_parity/20261012-0947*`…`-101558-17-cli`），两侧 pytest 通过，两个裸解释器 `--help` 输出一致。AST 逐函数比对：原 43 个函数中 40 个逐字节相同，只有 `plan`（`is_drive_path`、`QuantiloomCli.version()`）、`run_dataset`（`is_drive_path`、`renderer.start`、`RenderProcess.terminate`、`run_closure_audit`、审计脚本路径改为从仓库根定位）、`main`（`CalledProcessError` 来自 renderer）有意改动。`RenderProcess.terminate` 在两侧用真实子进程验证过。测试改为直接导入包（61 passed / 2 skipped）。顺带：`test_boundaries` 放宽为 datagen 函数内可惰性导入 OpenEXR（`24915e5`，与 AGENTS.md 原有约定一致） |
| T2.12 | 完成 | `b74fb8e`：`datagen/fingerprint.py`；闭包 24 个文件（13 个 datagen 模块 + 7 个 core 模块 + 4 个 `__init__`），`core/bands.py` 与消费端不在内。smoke 与正式配置对拍均 `PARITY OK`（`build/_parity/20261012-102129-fp-smoke`、`-102152-fp-formal`，后者 7785 个文件）；WSL 与 Windows 规划到同一目录时 `dataset.plan.json` 逐字节相同（`dataset_id` `b8da87a5…`，`build/_parity/20261012-102242-crossos2`）。两侧 67 passed / 2 skipped |
| T2.13 | 完成 | `46d6d5a`（`plan`/`run_dataset` 可注入渲染器，CLI 不变，smoke 对拍 OK）、`da78a81`（`tests/fake_renderer.py` + `tests/test_datagen_run.py`：完整运行、续跑空操作、失败重试、keep-going、产品损坏修复、未发布 attempt 恢复、中断、采集预算，共 8 项）。两侧 75 passed / 2 skipped |
| T2.14 | 完成 | `32a0a89`、`e3bfba9`：闭合审计与光谱积分审计改为经 `native_path` 取模型包。pilot `concrete_a007_e00_r00_n00/lwir/attempt-0001`：旧脚本在 WSL 下 `FileNotFoundError`（复现缺陷）；新旧脚本在 Windows 下输出逐字节相同；新脚本在 WSL 下与存档 `closure_audit.txt` 一致（`mean=-2.234 rms=6.384`）；经 `run_closure_audit` 调用同样得到 residual 行。光谱审计两侧输出一致 |
| T2.15 | 完成 | 用新包真实渲染 smoke（Quantiloom 0.7.4）：`build/env-smoke-20261012-103949`（一次跑完）与 `build/env-smoke-int2-20261012-104814`（中断后续跑，约 3 分 51 秒，`rendered=18 failed=0`，无 QA 警告）；两侧 `verify --inspect-exr` 均 `complete=6, partial=0, planned=6`；WSL 下 CLOSURE 行给出残差（grassland `mean=+0.089 rms=6.801`）；夜间 VIS rawdn 中位 0、月光 6.0，与 0.7.4 基线一致；带真实数据集两侧 `77 passed, 0 skipped`（当时计数）。正式配置跨系统规划：`dataset_id` 相同（`c2625af5…`），但 549 个任务 TOML 差 1 ULP → 发现并修正跨系统重规划缺陷（`aa07475`，两方向实测续跑不改文件，`build/_parity/20261012-110816-formal-replan-wsl2win`、`-111128-formal-replan-win2wsl`）。文档与 skills 同步（`be21fe3`…`399fd2f`） |
| T2.16 | 完成 | WSL 下 Ctrl+C（PTY 前台）两次：`Quantiloom.exe` ≤1 s 退出，rc 130，状态 `interrupted`，锁释放；无需补 Windows 侧终止方案。注意后台 `&` 进程继承被忽略的 SIGINT，`kill -INT` 无效，不能用来模拟 |
| T3.1 | 完成 | `a3a09ed` `dataset/exr.py`（`read_channels`/`read_image`/`read_header`/`write_image`） |
| T3.2 | 完成 | `edf9683`（`Product.channels`）、`937f670` `dataset/schema.py`：严格解析 index/plan 子集/sample；产品名由文件名推导（`frame_<x>.exr/json`→`<x>`，`frame.exr`→`rendered_reference`，`frame.png`→`preview`，其他报错）；四个传感器产品的 geometry 必须一致；用 FakeRenderer 产出的 sample.json 做生产端/消费端交叉检查 |
| T3.3 | 完成 | `3e593ab` `ThreeBandDataset`。pilot 24 个样本两侧全部解析；样本 0 每波段 13 个 EXR 产品可读（VIS 640×360、MWIR 320×256、LWIR 256×192），实例 ID 与材质 ID 文件同名通道 `instance_id` 但按文件区分正确。工况推导仍按 `scene_id` 后缀（`condition_of`） |
| T3.4 | 完成 | `6410cc6` `tests/dataset_fixture.py`：各波段分辨率不同的微型数据集。两侧 90 passed / 5 skipped |
| T3.5 | 完成 | `f1c0978` `leave_one_group_out`（只读 index）；pilot 24 个样本同属一个 split_group（concrete 的 glTF），白天/夜间/月光不会跨折泄漏 |
| T3.6 | 完成 | `07a00c5`。约定实测（pilot 全部 72 次采集，`build/_geom/probe_projection.py`）：相机系 = `world_to_camera`，Z 即 `truth_camera_z`（差 ≤ 2.3e-5）；投影为 OpenCV 形式的正向 Brown–Conrady，连续坐标左上像素中心在 (0.5, 0.5)，自投影误差 ≤ 0.001 px。包内 API 一律用像素索引坐标（中心为整数）。测试阈值 0.005 px / 1e-4（约 5 倍余量） |
| T3.7 | 完成 | `0ffed23`。判据：4 个双线性源邻居 validity 全为 1；双线性采样的源 `truth_world_position` 与参考点距离 ≤ `pos_rel_tol`·深度（替代 z 测试，同时用到两台相机各自的真值）；边缘带用世界坐标二阶差分（相邻深度跳变判据在掠射地面上误标 54% 的 LWIR 像素，已弃用）。默认 `edge_px=1, edge_rel_tol=0.02, pos_rel_tol=0.005`。pilot 24 样本：VIS→LWIR 有效 58%（其余主要在 LWIR 视场外），LWIR→VIS 91%，VIS→MWIR 46%，MWIR→LWIR 86%；线性重采样位置残差中位 1e-5～2e-5 深度。参数扫描与判据可视化见 `build/_geom/measure_warp.py`、`criterion-vis-to-lwir.png` |
| T3.8 | 未开始 | |
| T3.9 | 进行中 | 首个数据点：pilot 全部 72 个 rawdn（24 样本 × 3 波段）读取，WSL 0.85 s（11.8 ms/个）vs Windows 0.10 s（1.4 ms/个），经 9p 读 H 盘约慢 8 倍（`build/_refactor/pilot_reader_check.py`，热缓存，待用更大读量与冷缓存复测） |
| T4.1 | 完成 | `4a00166`（`Observation.readout`：CFA、ADC 位深、黑电平、曝光、白平衡、色彩矩阵，取自渲染器实际使用的相机配置）、`d49eb09`（BT.601 全范围 YCbCr、双线性去马赛克）、`8666dc3`（`VisDisplay`、`VisRawLinear(scale=adc|percentile)`、`IrLinear`、`IrTemperature`）。pilot 实测：只有 RGGB 相位使去马赛克 RAW 与线性化 display 在三个通道上都是 1/4095 的比值（R 相关 0.76–0.95；B 通道信号弱、相关低但比值正确）。夜间 display 亮度中位 0.0003，`percentile` 缩放可拉起（尺度 5 DN 对 4095）。注意：EXR 通道按字典序存放，读 RGB 产品必须显式给通道顺序（`normalized_input` 已处理）。两侧 127 passed / 10 skipped |
| T4.2 | 完成 | `d0cf988`：`OracleRegistrar`、`ResizeRegistrar`、`InfinityHomographyRegistrar`、`registration_error`。pilot 24 样本对 oracle 的 EPE 中位（源像素）：VIS→LWIR resize 1.86 / 无穷远单应 1.16；LWIR→VIS 3.24 / 2.02；VIS→MWIR 2.11 / 1.28；MWIR→LWIR 3.27 / 1.99。VIS→LWIR 单应误差随深度：<5 m 2.55、5–10 m 1.53、10–20 m 0.83、20–50 m 0.39、>50 m 0.20 px（与 f·b/z 一致），58% 的 oracle 有效像素误差 > 1 px（`build/_geom/measure_registrars.py`） |
| T4.3 | 完成 | 依赖按 paper-exp `uv.lock` 精确钉死（opencv-python-headless 5.0.0.93、scipy 1.18.0/1.17.1、scikit-image 0.26.0）；`fusion/align.py`（`aligned_sources`、`colourize`）与 `fusion/classical.py`（`average`、`max_abs`、`laplacian_pyramid`，移植自 `paper_exp/baselines/classical.py` ab70a3d）。与 paper-exp 双源等价：average/max_abs 逐位一致，laplacian_pyramid 最大差 1.8e-7 |
| T4.4 | 完成 | `evaluation/{no_reference,source_ref,full_ref}.py` 移植自 paper-exp（no_reference/source_ref @42024cd，full_ref @ab70a3d），源参考指标另有 N 源推广 `*_n`（双源等价已测）；`evaluation/metrics.py` 包装为 `Metric`（EN/SD/SF/AG/SCD/Qabf/MS_SSIM，version=1，评 Y，区域为各波段 support 交集） |
| T4.5 | 完成（pilot）；子集运行中 | `2f9b4b0`（`ResultStore`：JSONL、逐行 fsync、尾行撕裂自动修复、目录级单写者锁，复用 `core.locking.runner_lock`）、`151b7b3`（`reference_only` 单波段基线）、`71b1db8`（`experiments`：TOML 配置、变体展开与指纹、按样本推进的可续跑驱动、`summary.csv`、`python -m quantiloom_fusion.experiments run\|summarize`）、`45f440b`（`configs/experiments/{pilot,subset}_baselines.toml`）。pilot：24 样本 × 112 个变体（4 波段集 × display/raw_percentile × oracle/单应/resize × 3 经典方法 × grey/colour，单波段与无 IR 时折叠）= 18912 行、0 错误、1455 s（约 60.6 s/样本，全部是打分：MS_SSIM 约 24 s、Qabf 约 11 s；归一化约 0.05 s，warp ≤ 0.16 s，融合 ≤ 0.03 s）。续跑实测：第二次运行只补缺的 12 行，0.5 s。两侧 213 passed / 11 skipped |
| T4.6–T4.7 | 未开始 | |
| T5.0 子集采集 | 完成 | 96/96 COMPLETE、failed=0（`build/env-subset-0.7.4.log`）；`verify --inspect-exr` 尚未跑。以下为启动时的记录： 用户选择先跑子集：`configs/env_dataset_subset.toml`（`bbbf3e3`；与正式配置只差名称、`closure_audit_every=12` 与采样：12 个场景条目 × 8 方位 × 俯仰 12° × 半径 1.0 = 96 个三元组）。2026-10-12 起在 Windows 原生进程中运行（`.venv-win` 的 python，`Start-Process` 脱离 WSL），输出 `build/env-subset-0.7.4`，日志 `build/env-subset-0.7.4.log`/`.err`。按 pilot 吞吐估约 10 小时、约 3.7 GB。**运行期间 datagen 冻结于 `bbbf3e3`** |
| T5.1–T5.4 | 未开始 | 全量采集放在子集之后、用新目录 |
| 迁移到 Quantiloom 0.8.3 | 完成 | 同一批任务 TOML 与种子分别用 0.7.4、0.8.3 渲染对照（smoke 6 个三元组、pilot 前 3 个），结果见数据集文档"0.7.4 → 0.8.3 迁移对照"。生成器两处修复：只写一个湿度键（`f2b5a22`，0.8.3 smoke 84 个产品与修复前逐像素一致，告警 9 → 0）；`look_at` 放在相机组到目标的距离上（`fa0ea21`，ql_site 首次渲染因 float32 位姿精度被校验拒绝后发现）。文档与 skill 的版本、目录与基线数 `f0a79a9`。`build/env-smoke-0.8.3` 两侧 `verify --inspect-exr` complete=6/6；两侧 `215 passed, 11 skipped`，带真实数据集 `225 passed, 1 failed`（既有问题，见 §2.5 第 11 条） |
| ql_site 实测扫描场景 | 完成 | `10b0e68` `configs/env_dataset_ql_site.toml`：引用 `H:/LidarPointCloud/out/ql_site/site.mwir.toml`，authored 温度、NN `clear` 大气、8 方位 × 45° × 300 m。`build/env-qlsite-0.8.3-r2` 一次跑完 8/8（7 分 50 秒），`verify --inspect-exr` complete=8/8；闭合审计 rms 2.0/1.8 K。红外是模拟的（只有网格与照片纹理实测），见数据集文档对应小节 |
| 同步 Quantiloom 0.10.0 与 Qt 场景（2026-11-11） | 完成 | 夜间/低照度臂大气与白天同预设（concrete clear、desert 20 km、grassland 补 23:00 湿度与气温），四份配置同步；正式配置 `plan` 12 个条目全部通过（`build/_parity/20261111-c1-formal`，2592 三元组）。采样改为 8 方位 × 俯仰 12°/25° × 3 距离：`plan` 报 576 三元组 / 1728 次采集，每条目 48 个（`build/_parity/20261111-c2-formal`），按 0.8.3 吞吐外推约 6.9 GPU 小时（见 §2.2）。白天太阳谱按 forcing 直射缩放（`daylight_solar_scale = "forcing_direct"`，生成器从 forcing 与 LUT 算）：concrete 1.0371、desert 1.0908、grassland 1.0113、seaside 0.7381，低照度臂再乘 2e-6，记入计划 `scenes[].assumptions.solar_scale_derivation`（`build/_parity/20261111-c3-formal`）。相机固定图样与 NUC 逐相机显式配置（`[cameras.*]` 的 `apply_nuc`、`nuc_residual_fraction` 与逐像素/列/行分量，LWIR 不再依赖预设默认值），MWIR 改为两点校正残余 3 %，两类探测器启用列为主的行列条纹；生成器对新键的类型、取值范围与探测器归属给出明确报错。渲染器 0.10.0 只经 `share/quantiloom/` 找着色器，开发构建树的 exe 在 attempt 目录里启动报 `no Quantiloom resources found`（`build/env-smoke-0.10.0-pre13`，0/6），配置改用 SDK 安装版 CLI（同一二进制，SHADERSET 摘要一致），计划记录 `cli_version = "Quantiloom 0.10.0"`、`physics_sources` 为空（`build/env-smoke-0.10.0-pre13b`）。文档按 Qt 当前场景改正：车身为白色涂层实测谱（LWIR ε≈0.92），人体为给定 307 K 皮肤温度、不参与热求解，海面以 Qt 为准（纯水 n,k、0.10.0 起红外方向菲涅耳镜面瓣、热求解半球发射率 0.95），白天只有晴空，渐晕关闭；§2.5 第 9 条关闭。23:00 行重新核对：四个场景的 `t_ground_K`/`air_temperature_k`/`rh`/`atmosphere_temperature_k`/`initial_temperature_k` 与重新生成后的 forcing 一致（grassland 原缺 rh 与气温，已补） |
| 闭合审计按材质分类（2026-11-11） | 完成 | `tools/_closure_audit.py`：漫射材质用各向同性式（半球平板天空 L_down，另列天顶 ldown），镜面材质（`fresnel_complex_refractive_index`）用逐像素方向菲涅耳 F(θ,λ)（n,k 表取自 capture.toml 的 `[refractive_index]`）且只取镜面方向朝天的像素；T_air 改用 `t_ground_K`，预设先按渲染器表解析；不参与求解的材质用作者温度。pilot 0.10.0 24 个 LWIR：铬球 各向同性 −20.9/−24.7 K（夜/日）→ 方向菲涅耳 +0.45/−3.4 K，rms 3.4/5.4 K（未到 1 K）；夜间漫射混凝土 rms 0.43 K。0.10.0 对 n,k 材质导出的 `band_emissivity` 为 null（见数据集文档） |
| 大气气温与湿度跟随 forcing（2026-11-12） | 完成 | `datagen/{solar,settings,planner}.py`：新必填键 `atmosphere_air`（`forcing_row`/`none`），生成器对每个带 `[atmosphere]` 的条目从其 forcing 在 `[thermal] time_h` 处取气温与湿度，写 `t_ground_K`、`air_temperature_k`、`rh`，来源行记入计划；覆盖段手写这些键被拒绝。四份数据集配置删去夜间/低照度覆盖段的手填值（ql_site 为 `none`）。回归：四份配置改动前后逐任务比较，夜间/低照度任务数值不变；白天 concrete 308.80 K/30.4 %、desert 312.25/11.0 %、grassland 293.18/54.1 %、seaside 299.86/68.3 %。LWIR 对照（`build/_qa/atmos-forcing-{before,after}`，Ts 逐像素相同）：漫射地面 ≤0.2 K，掠射海面 +1.6 K，铬球朝天像素 desert −4.8 K、seaside +4.1 K、concrete +0.02 K（湿度同降抵消升温）。`atmosphere_temperature_k` 在 NN 大气下不被读取，不推导。测试 `test_atmosphere_air_follows_the_forcing_row_of_each_entry` |
| 融合支撑与可对应掩码分开（2026-11-12） | 完成（参考口径待用户定） | `geometry/warp.py`：`Warp.support`（融合采样与带掩码金字塔用，默认等于 `valid` 且必须包含它）；`oracle_warp` 的 `support` = 视场内 ∧ 源四邻有效 ∧ 同一表面点（含边缘带，只剔除遮挡与视场外），另加双方都是 miss 的天空（按无穷远方向对应——此前天空不在 oracle 支撑里，地平线一带的边缘带被退回可见光）；`valid` 仍是可对应层，配准误差照旧在它上面统计。`apply_warp`、`aligned_sources` 改用 `support`；两个扰动函数继承 oracle 的 `support`（逆深度偏置不移动天空）。测试：支撑 = 边缘 ∪ 可对应（∪ 天空）、天空按纯旋转映射、盒子场景完全配准时边缘带与全掩码融合的差只剩遮挡缝附近（挖掉边缘带时远离遮挡处最大差 >0.5）。pilot 0.10.0 四个参考样本（`build/_eval/run_eval.py`，带掩码基线、配准口径真值参考 PSNR 逐样本平均，修正前结果留在 `rows_before_support_fix.csv`）：幅度 0 时 oracle 可对应层 20.98 → 22.58（只修边缘带）→ 23.29 dB（再加天空），边缘层 16.45 → 18.94 → 20.59 dB；单应 23.65 / 21.98 dB；敏感度曲线随偏置单调下降（23.29→23.11 dB）。剩余差距全部来自遮挡缝附近：全掩码共视参考在遮挡区有源相机看不到的红外，oracle 退回可见光、单应用邻近内容填补。白天样本若参考也按 oracle 的支撑构造，oracle 可对应层 38.52/40.83 dB 对单应 34.41/35.71 dB、边缘层 29.00/29.29 对 24.16/25.04 dB；低照度样本噪声主导、两者持平。参考是否按此构造待用户决定 |
| 逐层噪声增益按处理链求取（2026-11-12） | 完成 | `fusion/noise_aware.py`：`chain_level_gains`（单位白噪声放在波段原生网格上，经该波段实际处理链再做同一金字塔，量各层标准差；4 次带种子抽样，离缺值边界 4·2^l px 以内的系数不计），`band_level_gains`（参考波段可给原生链，源波段再经 `aligned_sources` 用的同一个双线性 warp；无原生链时按波段与重采样比例缓存），`NoiseAwarePyramid` 默认 `gain_model="chain"`、`"white"` 保留原白噪声增益 `level_noise_gain`（仍被测试）；所用增益写进 `extras["level_gain"]`。σ 语义写明为原生像元的噪声、给在参考网格上。`signal/noise.py`：`vis_luminance_chain`（去马赛克、白平衡、颜色矩阵、BT.601 亮度）；`fpn=True` 改为按渲染器相机管线读全部分量——光子型增益偏差（PRNU、列、行）乘到达像元的光电子、偏置（DSNU 按曝光/参考曝光缩放、列、行），热型偏置（逐像素、列、行，K）与可选的增益项（需入射功率折成 K）；校正产品（corrected、tapp）乘 NUC 残余比例，rawdn 保留全部；行列条纹按逐像素方差折算（单像素边缘方差精确，跨像素不是白的，对逐层增益是近似）。光子型配 `IrTemperature` 时经 `estimate_dsdt` 的 e⁻/K 换算。实测（pilot 0.10.0 `concrete_moonlight_a000`，oracle warp，4 层，实测增益/白噪声增益）：LWIR→可见光网格（每可见光像素 0.569 个长波像素）0.50/1.35/1.66/1.69，MWIR（0.852）0.60/1.03/1.14/1.17，可见光亮度 0.26/0.78/0.98/1.03（含本相机白平衡与颜色矩阵）。测试：三种处理链下纯噪声在 κ=3 的残留比例逐层落在高斯尾 0.27 % 的 0.3–3 倍内，白噪声增益则至少一层偏出 5 倍；恒等链复现白噪声增益（3 %）；融合默认用处理链增益。低照度对照（`build/_eval/noise-gain-0.10.0`，可见光+长波，κ=2.5，平坦区 = 可对应层里两幅共视点参考高通都 <0.002 的像素，高通 σ=2 px 标准差 ×10⁻³）：moonlight_a000 基线 5.91、白噪声增益 5.64、处理链增益 5.80；moonlight_a002 10.24/9.80/10.08；白天 a000 2.27/1.98/2.08。处理链增益让可见光细层阈值降到原来的 1/4，收缩变弱，所以平坦区噪声比白噪声增益略高——白噪声增益在细层等于 κ≈10，是过度收缩，不是更好；低照度可见光噪声以亮点状为主（对照图三幅肉眼几乎相同），软阈值对它作用有限 |
| 遮挡区偏离改在高通分量上度量（2026-11-12） | 完成 | `evaluation/layer_report.py` 的 `reference_deviation`：`hp_dev_RMSE` = 融合结果与参考波段各自减去同一高斯低通（σ = `hp_sigma_px`，默认 2 px，记入每行）后的差的 RMS，并按到源足迹（所有非参考波段都有支撑的像素）的距离分档报告——`inside`（距离 0：配准方法在这里采样了源图，在遮挡层里就是把源相机看不到的内容放了进来）、(0,4]、(4,8]、(8,16]、(16,32]、>32 px，各带像素数；原来的全信号偏离改名 `fullsig_dev_*` 留作对照（它量到的主要是多分辨率样条有意延伸约 32 px 的低频过渡）。共视点参考样本上这一层的真值参考指标照常给出。测试：平坦图（可见光 0.05、红外 0.55、足迹内 3 px 遮挡缝）高通偏离在足迹外各档都 <0.005、缝里全信号 >0.1 而高通 <0.005；纹理图上带支撑覆盖遮挡缝、错位 2 px 的 warp 使缝里高通偏离大于完全配准时 5 倍以上。说明：错位本身不会把内容带进遮挡缝，只有支撑覆盖了遮挡区的方法（单应、缩放）才会——同一支撑下平移 2 px，缝里高通 rms 只从 0.0021 变到 0.0025。pilot 0.10.0（4 个参考样本，带掩码基线，遮挡+视场外层，高通 ×10⁻³）：白天 单应 总 2.79、足迹内 11.27（3430 px）、(0,4] 0.56，缩放对齐 足迹内 13.77，oracle 无足迹内像素、(0,4] 2.92、更远 0.07–0.81；低照度 单应 足迹内 6.49、oracle (0,4] 3.82。全信号偏离 0.118–0.124（白天）对所有配准几乎相同；这一层真值参考 PSNR 白天约 9.9 dB、低照度约 13.6 dB，与配准方式无关（退回可见光对有红外内容的参考） |
| 互补性分析的掩码与折法（2026-11-12） | 完成 | `evaluation/complementarity*.py`：`pixel_masks` 把"波段发射率有限"只加在对比度分解上，温度回归与条件互信息用全部有效像素（结果里记下两种像素数和按材质的无发射率像素数）；每个降幅旁并列 R²（= 1 − SSE/Σy²，y 为逐帧去均值的地表温度，参考是零预测器即各测试帧自身的均值，不是训练均值）；新增场景内划分 `view_group_folds`（同一场景内按方位角分组，每折留出一个方位组，训练与测试不共享视角组）。测试：无发射率材质进回归/互信息但不进对比度；R² 定义；视角组折法互斥。重算（`build/env-subset-0.7.4`，4 个场景组 × 8 方位/工况；跨场景 = 4 折留一场景，场景内 = 每场景 8 折共 32 折）：白天跨场景 岭回归 总方差 14.44 K²、仅可见光残差 21.31、加长波 6.14，降幅 71.2 %，但 R²(仅可见光) = −0.48、R²(加长波) = +0.57；白天场景内 仅可见光 R² +0.43（kNN +0.51）——可见光到温度在场景内可预测、跨场景不可迁移，降幅被放大。夜间/低照度跨场景 R²(仅可见光) ≈ 0（−0.00/−0.03）、加长波 +0.69/+0.67。pilot 0.10.0 只有 concrete 一个场景组，只能做场景内（8 折）：白天 R² 可见光 +0.33 → 加长波 +0.55，夜间 +0.28 → +0.87，低照度 +0.14 → +0.86。无发射率像素在现有数据集里为 0：铬球、裸铝不参与热求解、真值温度为 0，本来就被"温度 > 0"排除；海水参与求解，要等含海边场景的 0.10.0 数据集。遗留：场景内划分下岭回归（二次项）在 seaside 上发散（白天测试 MSE 51.2 K²，kNN 正常），未处理，正式采集后需要看 |
| 闭合审计铬球残差按镜面天顶角分档（2026-11-12） | 完成（残差来源部分待核实） | `tools/_closure_audit.py` 对镜面材质加一行 0–30/30–60/60–75/≥75° 分档（偏差、rms、像素数）。pilot 0.10.0 24 个 LWIR：夜间 rms 3.46/3.33/3.25/3.89 K，各档持平，残差不集中在大天顶角，未改通过判据；白天 rms 3.53/5.73/5.21/5.94 K，多出的部分从 30° 起，主要是反射瓣看到的热地面（审计没有地面项；归因用的瓣宽是假设），另有近太阳方向约 6 % 像素的太阳反射（−4…−8 K）。夜间约 3.3 K 的离散来源未查明。细节见数据集文档闭合审计段 |
| 评测口径与敏感度曲线（2026-11-11） | 完成（实现与测试；pilot 0.10.0 四个参考样本上的演示） | `evaluation/{perturb,truth_ref,layer_report,scene,protocol}.py`：指标按层分组报告（可对应、边缘带、遮挡+视场外；后者没有源信息，度量融合结果相对参考波段的偏离 `dev_RMSE/MAE/PSNR`）；无参考指标只以 `ctx_` 前缀作对照，`inflation_experiment` 专门展示错位下 AG/SF 虚高（pilot 白天场扰动 0→3.6 px：AG +17 %、SF 2.25→2.67，而真值参考 PSNR 24.98→23.25 dB）；源参考指标逐波段只在自身足迹内参与（`scd_n`/`qabf_n` 已精确满足；`ms_ssim_fusion_n` 的粗尺度会把足迹外的填充漏进来，约 5.5e-5，评测侧用 32 px 边距规避，原指标不改）；真值参考两种口径分开（`TruthMode.REGISTRATION`：同一算子作用在共视点无噪声图像上；`TruthMode.FUSION`：统一以全 1 掩码的 `laplacian_pyramid_masked` 作用在共视点无噪声图像上为公共参考，另报以无噪声源图像为参照的源参考指标；测试证明输出恒为常数的算子在后者下得低分、在自参考下会得满分）；扰动有沿视线的逆深度偏差与固定种子的平滑随机位移场，幅度 0 时逐位复现 oracle；每行记录归一化（拒绝红外显示图像，真值参考拒绝逐图百分位窗）与是否使用标定（`calibrated` 列）。发现：(1) 可对应层的分数被跨层边界的金字塔混合主导——幅度 0 时无穷远单应的真值 PSNR 30.07 dB 高于 oracle 24.98 dB，离非可对应区 32 px 以外则 oracle 48.42 > 单应 48.22；建议加一个腐蚀后的 `correspondable_core` 组（改层集合，待定）。(2) 带掩码的两种金字塔在遮挡+视场外并非干净退回参考波段（`dev_RMSE` 白天 0.108、月光 0.052），推测是热细节跨掩码边界渗透，未隔离。(3) SCD、Qabf 在错位增大时也会上升，只有真值参考指标单调。matplotlib 未安装，只出 CSV（`build/_eval/pilot-0.10.0/rows.csv`） |
| 公开深度融合模型零样本对照（2026-11-11） | 完成（0.7.4 pilot） | `fusion/zeroshot.py`（`__optional_deps__ = ("torch",)`，不经 `fusion/__init__` 导入）注册 `superfusion_msrs`、`superfusion_roadscene`：SuperFusion（IEEE/CAA JAS 2022，代码 `bee015a8938bee549d0132b80c1daf161ff80660`，作者权重 MSRS.pth `aad56e62…aa54e94`、RoadScene.pth `09337ddd…a9a0e03ae11fdf9a416`，完整 SHA-256 见 `build/_zeroshot/results.md`），只按 VIS–LWIR 双波段用。UMF-CMGR 权重只在百度网盘、MURF 是 TensorFlow、RFNet 仓库要求登录，均未选。预处理：VIS 取显示图像量化到 8 bit；LWIR 用 `IrTemperature` 270–330 K 单调线性温窗量化到 8 bit，按纯缩放放到 640×360 VIS 画幅（模型自带配准，不用我们的 oracle）。kornia/cupy 不在锁文件里，模型代码与纯 PyTorch 替身放在 `build/models/`（不入库；IRNN 替身已与原核语义逐方向对照，高斯模糊与网格未与 kornia 对照）。8 个白天 + 8 个月光样本，模型形变场换算到原生 LWIR 像素后对 oracle 的中位端点误差：MSRS 权重 白天 1.99 / 月光 5.09 px，RoadScene 权重 2.17 / 7.79 px，纯缩放基线 1.85 px。这组结果只用来定位基准的难度，不构成对这些方法的评价。测试在无 torch 或无权重时干净跳过 |
| 极线逆深度扫描配准（2026-11-11） | 完成 | `registration/epipolar.py`：`EpipolarSweepRegistrar`（逆深度 d∈[0, 1/z_min] 上 14 个假设，u′ = π_s(R π_r⁻¹(u) + d t)，含各自畸变；相似度 `grad_ncc`/`grad_orient`/`local_mi` 可切换，`registration/similarity.py`；置信度 = 峰值显著性 + 左右一致性 + 结构强度门限（参考与重采样源图在窗口内的梯度能量都要高于各自噪声底 3 倍），不置信处退回 d=0）与对照 `FreeSearchRegistrar`（同一相似度下的二维自由搜索，范围等于极线扫描的最大位移）。测试：d 全取 0 时与无穷远单应逐像素相同；已知深度的正对纹理平面（源图非单调灰度并翻转极性）落在真值所在的格；纯噪声一侧时置信像素为零、结果等于单应。pilot 0.7.4 VIS→LWIR 可对应区中位端点误差（源像素 / mrad）：单应 1.16 / 4.63，极线 `grad_ncc` 1.13 / 4.52（白天 0.98、夜间 1.15、低照度 1.20），自由搜索 1.19；VIS→MWIR 单应 1.28 → 1.19（白天 0.83）。夜间 VIS 无结构、门限把它退回单应；低照度 VIS 有真实结构通过门限，反而比单应差 0.03–0.05 px（`build/_geom/epipolar_report.md`、`epipolar_report_display.md`）。**0.10.0 pilot**（`build/env-pilot-0.10.0`，vis=raw，`build/_geom/epipolar_report_0.10.0.md`）：VIS→LWIR 可对应区中位 单应 1.16 px / 4.63 mrad → 极线 `grad_ncc` 1.12 px / 4.46 mrad（白天 1.00、夜间 1.15、低照度 1.16），边缘带 0.96 → 0.49 px，自由搜索 1.18；按深度 [10,20) m 0.83 → 0.78、<5 m 2.55 → 2.49；VIS→MWIR 1.28 → 1.14 px（白天 0.82）。默认取 `grad_ncc`、左右一致性开、结构门限 3 倍噪声底 |
| 分层掩码（2026-11-11） | 完成 | `geometry/layers.py`：参考相机像素网格按 无效 → 视场外 → 遮挡 → 边缘带 → 可对应 的顺序分成互斥五层（`layer_map`、`Layer`、`layer_fractions`、`layer_fractions_by_depth`）；遮挡用两台相机各自的世界坐标真值比较（距离 > 0.5 % 深度），边缘带用世界坐标二阶差分，与 `oracle_warp` 共用 `_oracle_masks`，测试断言五层互斥覆盖全图且 `oracle_warp(...).valid` 等于可对应层。pilot 0.7.4（24 样本）VIS→LWIR 占比：无效 0.305、视场外 0.236、遮挡 0.012、边缘 0.041、可对应 0.406；VIS→MWIR 0.305/0.339/0.008/0.027/0.321；<5 m 的近处 78–92 % 落在视场外，>50 m 边缘带升到 17–32 %（`build/_geom/epipolar_report.md`） |
| 物理互补性分析（2026-11-11） | 完成（实现与测试；数值基于 0.7.4 子集，待 0.10.0 数据重跑） | `evaluation/complementarity*.py`：对比度分解（L = εB + (1−ε)L_down 双线性，对称分解精确、无余项；L_down 真值缺失，报告 0 与场景均温黑体两种假设）、VIS→ΔT_s（单帧去均值的表面温度真值）留出残差（岭回归二次特征与 kNN，自写）、条件互信息 I(Y; X_LWIR | X_VIS)（分箱插值估计，层内置换扣偏差，Y 取表面温度或材质类别），按白天/夜间/低照度分别报告，按 split_group 留一组交叉验证并在结果里带折数与组构成；像素对只取 LWIR 参考网格上 oracle 的可对应区。`build/env-subset-0.7.4` 四折：岭回归残差降幅白天 71 %、夜间 69 %、低照度 68 %；但白天仅可见光的残差（21.3 K²）比总方差（14.4 K²）还大——VIS→温度关系不跨场景迁移，降幅因此被放大，须与总方差一起读。pilot 只有一个组，只能给组内分块的数（已标明非跨场景）。表与脚本：`build/_complementarity/` |
| 噪声感知多尺度融合（2026-11-11） | 完成（实现与测试） | `signal/noise.py`：逐波段归一化噪声 σ（光子型 √(总电子数+读出²)，总电子数含信号、背景、暗电流；热型由 NETD 经 dS/dT 或 K 窗；除以归一化窗跨度；固定图样开关默认关）。`fusion/noise_aware.py`：`laplacian_pyramid_masked`（掩码外最近有效像素延拓 + 掩码高斯金字塔加权 + 取加权系数幅值最大者，顶层按 G_top(M_b) 加权平均）与 `noise_aware_pyramid`（同上，另对每个波段含参考波段做软阈值收缩，σ_{b,l} 由同一金字塔的单位冲激能量精确折算，κ 默认 2.5）；`laplacian_pyramid` 原样保留。测试：σ→0 等于带掩码基线、全 1 掩码且 σ=0 等于标准金字塔、0.40/0.030 反例选可见光、纯噪声细节层为零、中央矩形足迹外细节层等于参考波段收缩后系数且边界无过冲，每条都做过变异检查。两点如实说明：(1) 高斯噪声下"纯噪声 → 细节层恰为零"只在 κ 足够大（测试用 6）时严格成立，κ=2.5 时 1% 量级的尾部系数会留下；(2) 取最大者的规则使足迹内（两个波段的残余可选）噪声底高于足迹外，pilot 月光样本平坦区高通 std 内/外 0.00029/0.00007（基线 0.00210/0.00045），绝对值都降了约 7 倍但比值约 4 未变（`build/_fusion/noise_aware_vs_baseline_moonlight_a000.png`）。σ 尚未接入实验驱动（`FusionInput` 不带观测与归一化器，调用方用 `band_sigma` 构造） |

## 7. 决策记录

| 日期 | 决策 | 理由 |
| --- | --- | --- |
| 2026-10-11 | 框架覆盖全链路（生成 → 读取 → 几何/配准 → 融合 → 评价） | 用户确认 |
| 2026-10-11 | 生成器立即拆成包，以行为保持对拍保证正确 | 用户确认；正式采集尚未开始，现在改指纹代价最小 |
| 2026-10-11 | PyTorch 2.9.1（cu126 索引）+ uv；生成端保持仅标准库 | 用户确认 |
| 2026-10-11 | paper-exp 按需移植，不作运行时依赖 | 用户确认 |
| 2026-10-11 | src 布局、TOML 实验配置、pytest、Conventional Commits（英文） | 与现有配置及兄弟仓库一致 |
| 2026-10-11 | WSL venv 放 `$HOME/.venvs/quantiloomfusion`，Windows venv 为 `.venv-win` | 共用工作树，两侧 venv 不能互相覆盖；9p 上建 venv 慢 |
| 2026-10-11 | 配方指纹改为 datagen 导入闭包 | 包化后单文件哈希不再完整；同时避免消费端改动误伤续跑 |
| 2026-10-12 | `build/` 改为指向 `H:\QuantiloomFusion-build` 的 junction | 用户确认并执行；机械盘空间大，代码和 venv 留在 SSD |
| 2026-10-12 | datagen 对拍基线 = `bad41d3`（生成器首次入库） | `datagen-parity` 用 `git log --diff-filter=A` 自动找到它 |
| 2026-10-12 | `.clang-tidy`、`.clangd` 加入 `.gitignore` | 用户确认；属于 Quantiloom C++ 仓库，与本仓库无关 |
| 2026-10-12 | `assets/`、`paper/` 保持被 gitignore | 用户确认 |
| 2026-10-12 | 两侧 uv 统一到 0.12.10；WSL 另装独立版到 `~/.local/bin`，不动 conda base | Windows 已是 0.12.10；新版写的 lock 旧版不一定能读 |
| 2026-10-12 | datagen 中的 `BANDS` 仍是普通字符串元组，不改用 `core.bands.Band` | 零风险保持任务 TOML 与指纹逐字节不变；类型化的 `Band` 只给消费端用 |
| 2026-10-12 | 拆包期间每步产出补丁、由 lead 逐个审阅后各自提交 | 一事一提交，同时每步都有对拍与双系统测试的证据 |
| 2026-10-12 | 审计保留为 `tools/` 下的独立脚本，以子进程调用，不迁入 `datagen/audit/` | 子进程隔离 + 300 s 超时在 11 天长跑中更稳；审计在模块级依赖 numpy/OpenEXR/safetensors，放进 datagen 会破坏"仅标准库"约束；它们本就不进配方指纹 |
| 2026-10-12 | `FakeRenderer` 放在 `tests/`，不进包 | 只给测试用；放进 datagen 会无谓地进入配方指纹 |
| 2026-10-12 | 首次规划的系统冻结计划；跨系统重规划按 1e-12 容差比对，保留已存文件 | 两系统 libm 末位不同是平台事实，逐字节一致做不到；保留首次文件让 `job_sha256` 与磁盘一致 |
| 2026-10-12 | 先跑 96 三元组子集，全量随后另开目录 | 用户确认；让阶段 3/4 尽早用上全分辨率真实数据 |
| 2026-10-12 | VIS 输入 `display` 与线性 RAW 都做、可切换；融合输出彩色与灰度都给 | 用户确认；哪种更好待实验回答。两种输出来自同一次亮度通道融合，指标在 Y 上算以保持可比 |
| 2026-10-12 | 可选 torch 模块以模块级 `__optional_deps__ = ("torch",)` 声明 | AST 可直接检查；比文件名约定灵活（深度模型可能占整个子目录） |
| 2026-10-13 | 单波段变体上 SCD/Qabf/MS_SSIM 记为 NaN 行，不进错误重试 | 这些指标至少需要两个源，永远算不出；NaN 是仓库既有的"未定义"约定，否则每次运行退出码都是 1。汇总时 NaN 不计入均值与 n |
| 2026-10-13 | 方法指纹 = 完整变体 spec（波段集、输入及归一化参数、配准、融合及参数、输出类型）的 sha256 前 16 位；实验名与指标列表不进指纹 | 同一方法在不同实验里可复用已有结果行；改任何影响结果的设置都会换指纹 |
| 2026-10-13 | 实验目录里的 `config.toml` 副本与输入配置不一致则拒绝运行 | 改配置必须换实验名，避免同一个结果文件里混入不同设置 |
| 2026-10-26 | 渲染器迁到 Quantiloom 0.8.3；0.7.4 的数据集保留、只读 | 用户要求。实测 IR 采集快 13–18 倍，近场大气与海水 IR 修正使 `tapp − T_s` 更接近发射率所定的值 |
| 2026-10-26 | 接受 0.8.x 的新缺省：`sky_distribution = "clear"`、非制冷 LWIR 预设 NETD 带宽 15 Hz；不钉旧值 | 两者都更符合物理（后者让相机真正交付配置声明的 50 mK）；铉 `uniform` 也复现不了 0.7.4 的天空像素 |
| 2026-10-26 | 保留 `lighting.atmosphere_temperature_k`（虽然 0.8.3 告警已弃用） | `atmosphere_mode = "off"` 消融臂关闭 NN 大气后，它仍是热天空回退值 |
| 2026-10-26 | ql_site 实测扫描场景单独成配置 `configs/env_dataset_ql_site.toml`，不并入正式配置 | 用户要求引用它作融合场景。它没有 `[thermal]`（authored 温度）、采样也不同（俯仰 45°、半径 300 m），并入会改变正式配方；大气从源配置的 disabled 覆盖为 NN `clear` 预设（300 m 斜距下 IR 路径项不可忽略） |
| 2026-11-12 | 全部 12 个条目的大气气温与湿度由生成器从 forcing 行推导（`atmosphere_air = "forcing_row"`），配置不再手填 | 用户决定。热求解与成像大气必须是同一团空气；白天沿用场景名义值时气温最多差 6 K，三种工况的处理方式也不一致。夜间值改为推导后数值不变，是回归检查 |
| 2026-11-11 | 同一场景的白天、夜间、低照度三个条目共用同一个大气预设（气溶胶类型与能见度），只有 23:00 forcing 行给出的气温、湿度、天空温度随时刻不同；四个场景之间不要求相同 | 用户决定。三种工况的比较才是照度的单变量对照；夜间地表温度来自晴空直射的白天热历史，成像大气要与之自洽。渲染器 0.10.0 对"地面参考太阳谱 + 非晴空"直接报错（低照度臂太阳在地平线上 5°，派生 CSV 保留 ETR 列名，同样受约束）：concrete 夜间/低照度由 urban_haze（8 km、ihaze 5）改为 clear，desert 由 18 km 改为与白天相同的 20 km，grassland 补上 23:00 的 rh 0.655 与气温 288.25 K |
| 2026-11-11 | 白天太阳谱按 forcing 直射缩放（scale = forcing 采集时刻直射 ÷ 参考谱直射列积分，由生成器计算）；不把海边采集时刻挪到太阳更高时 | 用户决定（挪时刻是备选）。只对齐幅度、不改谱形；NN 太阳散射缩放只跟随低照度 2e-6——幅度对齐不是照度工况的改变 |
| 2026-11-11 | MWIR 与 LWIR 统一为两点 NUC 残余 3 %，VIS 不校正；两类探测器启用行列条纹，总量不变、按列为主分配（LWIR 增益 0.016/0.023/0.010、偏置 1.10/1.55/0.65 K；MWIR 增益 0.0074/0.006/0.003、暗信号 5.9/4.8/2.4 e⁻） | 用户决定（取值为假设值）。三台相机校正口径不一致会污染融合比较；逐像素独立的图样在单帧上与时域噪声无异，条纹才是非制冷长波的特征 |
| 2026-11-11 | 夜间条目保留太阳表并按 `solar_scale = 1e-12` 缩放 | 渲染器只从 `lighting.solar_lut` 推导短波吸收率、否则取 0.7；夜间热求解（0→23 时）必须与白天、低照度用同一吸收率，而不缩放的太阳表在地平线下仍照亮天空。是对渲染器行为的绕行，更干净的做法是渲染器提供只用于推导吸收率的光谱入口（属 Quantiloom-dev） |
| 2026-10-12 | torch 索引由 cu126 改为 cu128，且只对 Windows 生效 | 与 paper-exp 一致（同版本、同索引，显存测量可比）；Linux 的 PyPI wheel 已带 CUDA |

## 8. 待决问题与风险

待决：

- 何时启动全量采集（576 个三元组，按 0.8.3 吞吐外推约 6.9 GPU 小时）。
- 是否在 0.8.3 下重渲 pilot（约 17 分钟）与子集（约 70 分钟）并重跑实验：`configs/experiments/*.toml` 仍指向 0.7.4 数据集，它们的 seaside 海面、近场大气与 LWIR 噪声与 0.8.3 不同（见迁移对照）。
- ql_site 是否扩大采样（更多俯仰/距离）、是否给它配专门的 VIS 曝光（沿用正式配置 5e-5 s 时 rawdn 中位只有 61/4095）。
- 工况标签是否在配置/样本清单中显式化（目前只能从 `scene_id` 后缀推导）。
- `sample.json` 是否升 v2（真值按产品名索引）。

风险：

- **已基本关闭（2026-11-12）：配准对比的评价区域不一致**。根因不在评价区域而在融合支撑：oracle 的 `valid` 不含边缘带与天空，被当成融合掩码后，红外在物体轮廓与地平线附近被挖掉。融合支撑分开后（见进度表"融合支撑与可对应掩码分开"），剩下的是遮挡缝附近 oracle 退回参考波段、而全掩码共视参考在那里有红外——是否让配准口径的参考也按 oracle 支撑构造，待用户决定；在此之前引用跨配准方式的边缘层差值要注明这一点。原记录：：评价区域是各波段 support 的交集，而 oracle 的有效区（排除边缘与对应不一致像素）比单应/resize 的小，所以三种配准方式很可能是在不同像素集合上打分的（推断，待核实：尚未量化两者 support 的差异；T3.7 只测过 oracle 有效区占比）。三波段 Qabf 出现 oracle(0.264) < 单应(0.295) 与此相符，在核实前不能据此判断配准好坏。修法：所有变体用同一公共评价区域（例如 oracle 有效区与各波段 FOV 的交集）再打分。修好之前，不要引用跨配准方式的指标差。
- SCD 与 EN 不能用来给配准排序：配准越差，融合图里不相关结构越多，二者反而偏高（pilot：vis+lwir 的 SCD resize 1.061 > oracle 1.056；EN 4.78 vs 4.04）。配准敏感度应主要看 Qabf、MS_SSIM 与 registration_epe。
- `ResultStore` 复用的锁 helper 其文件名与报错文案仍写"generator"（`.generator.lock`），只是命名不准；`core` 冻结期间不改。
- 正式运行期间改动 datagen 闭包会使目录无法续跑 → 冻结规则（`AGENTS.md`）。
- interop 中断可能残留渲染进程 → T2.16。
- WSL 9p IO 可能拖慢训练数据加载 → T3.9。
- 依赖真实数据集的测试受 ACL 限制，只能由渲染账户运行。

## 9. 交接备忘

- 当前停在（2026-11-12）：数据集配置与 Quantiloom 0.10.0、Qt 场景同步完成（夜间/低照度臂与白天同大气、576 样本、白天太阳谱按 forcing 直射缩放、夜间保留 1e-12 太阳表以推导吸收率、逐相机 NUC 与行列条纹、SDK 安装版 CLI、共视点无噪声参考子集与 sample v2、闭合审计分漫射/镜面）；`build/env-smoke-0.10.0`（6/6、参考 6/6）与 `build/env-pilot-0.10.0`（24/24、参考 12/12）已渲染并 `verify --inspect-exr`。第 6 章的分层掩码、极线扫描配准、噪声感知融合、评测口径、互补性分析、SuperFusion 零样本对照已实现入库（见进度表 2026-11-11 各行）。正式采集仍未启动：576 个三元组 + 144 次参考约 3 GPU 小时，用新目录（如 `build/env-three-band-0.10.0`），启动时冻结 datagen。
- 待用户决定：俯仰角取值（12°/25°）、参考子集视角（0/90/180/270°、12°、中间距离）、κ（默认 2.5）、相似度（默认 `grad_ncc`）、是否加腐蚀后的 `correspondable_core` 评测组、夜间 1e-12 太阳表的绕行是否接受（或请 Quantiloom-dev 提供只用于推导吸收率的光谱入口）、SDK CLI 让 `physics_sources` 为空是否接受。
- 此前状态（2026-10-13）：T4.5 完成。pilot 实验结果在 `build/experiments/pilot-baselines/`（`results.jsonl`、`summary.csv`、`variants.json`、`errors.jsonl`）。子集实验已在 Windows 上脱离启动（`configs/experiments/subset_baselines.toml`，约 96 × 60 s ≈ 1.6 小时），输出 `build/experiments/subset-baselines/`，终端日志 `build/_exp/subset-run.log`（输出有缓冲，进度看 `wc -l` 该目录的 `results.jsonl`）。中断了就用同一命令续跑：`cmd.exe /c "set UV_PROJECT_ENVIRONMENT=.venv-win&& uv run python -m quantiloom_fusion.experiments run configs/experiments/subset_baselines.toml"`（cwd 在仓库根）。子集采集已完成（0.7.4，生成器 `bbbf3e3`）。
- 下一步（按依赖顺序）：
  0. 决定是否在 0.8.3 下重渲 pilot（约 17 分钟）与子集（约 70 分钟）并据此重跑实验（见「待决」）；正式采集用新目录（如 `build/env-three-band-0.10.0`），576 个三元组外推约 6.9 GPU 小时。
  1. 修正配准对比的评价区域（见「待决问题与风险」第一条），升相关 `Metric` 的区域取法并换新的实验名重跑 pilot。
  2. 子集 `verify --inspect-exr`、汇总 CLOSURE 行与吞吐。
  3. T4.6 互补信息量化（VIS→`truth_surface_temperature` 回归残差、条件互信息 I(Y; LWIR | VIS)），T4.7 任务驱动评价（实例/材质 ID 分割）。
  4. 打分是耗时大头（MS_SSIM、Qabf 每个变体都重做 `aligned_sources`）；子集 96 样本可接受，扩大变体数之前先缓存对齐结果。
