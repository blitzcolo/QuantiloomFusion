# 干活指引与进度备忘录

本文是 QuantiloomFusion 的工作底稿：要达成什么、系统怎么切分、按什么顺序做、做到哪了。规则性的东西（git、跨平台、边界）以根目录 `AGENTS.md` 为准，本文不重复。

使用方式：

1. 开工前读「当前状态」与「交接备忘」，再到「任务分解」里领一个未完成的任务。
2. 一个任务可以对应多个 commit，但每个 commit 只干一件事；任务编号（`T2.4` 等）只在本文出现，**不要写进 commit message**。
3. 完成后更新「进度表」「决策记录」「交接备忘」（`progress-update` skill）。

最后更新：2026-10-12

---

## 1. 目标

### 1.1 研究目标

本仓库服务的研究问题：在**异构原生传感器**（不同分辨率、FOV、畸变、PSF、噪声、位姿，未配准）条件下，VIS / MWIR / LWIR 三波段融合能带来什么、代价是什么。数据集提供条件，不提供结论（见数据集文档「数据集定位」）。

| 编号 | 目标 | 交付物 |
| --- | --- | --- |
| G1 | 受控基准数据集：正式配置 2592 个三波段样本 / 7776 次采集，可断点续跑、可在 Windows 与 WSL 间交替执行、可独立校验 | `build/env-three-band-0.7.4/`（不入库）+ 发布清单 |
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
| 渲染器 | `Quantiloom 0.7.4`，RTX 4060 Ti 8 GB | `Quantiloom.exe --version`、`nvidia-smi` |
| smoke 数据集 | `build/env-smoke-0.7.4`：6/6 完整，18 次采集约 4 分钟 | `dataset.index.json`、render.log 时间戳 |
| pilot 数据集 | `build/env-pilot-0.7.4`：24/24 完整（concrete 白天/夜间/月光 × 8 方位），samples 304 MB + attempts 624 MB | 同上 |
| 场景资产 | `H:/Quantiloom-Qt/assets/configs/env_{concrete,desert,grassland,seaside}.toml` | 文件 |
| `src/`、`paper/`、`assets/` | 空目录 | `ls` |
| git | 仓库基线已入库（2026-10-12）；索引与工作树全部为 LF；工作区干净 | `git log`、`git ls-files --eol`，WSL 与 Windows git `status` 均为空 |

### 2.2 实测吞吐（决定正式采集的排期）

pilot 全分辨率（640×360 / 320×256 / 256×192，SPP 32，`wavelength_samples=12`）单次采集中位耗时：VIS 24 s、MWIR 152 s、LWIR 191 s，三元组约 6.1 分钟（24 个三元组共 2.35 小时，与之吻合）。按此外推，正式配置 2592 个三元组约 **264 GPU 小时（约 11 天连续运行）**；样本约 33 GB，attempts 约 67 GB（`build/` 在 H 盘，剩余约 1.1 TB，见 §2.4）。外推假设各场景耗时与 concrete 相近，待正式运行头几十个样本校正。

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

1. **审计工具在 WSL 下不可用**：`_closure_audit.py`、`_spectral_integration_audit.py` 把模型包写成 `Path("H:/Quantiloom-dev/assets/atmos_models")`，Linux Python 下这是相对路径。生成器在 WSL 下运行时的周期性闭合审计因此必然报 "audit unavailable"。
2. ~~**换行符风险**：`.gitattributes` 为空，而 Windows 侧 `core.autocrlf=true`。若由 Windows git 检出，源文件变成 CRLF，`generator_sha256` 随之改变，"同一数据集目录跨系统交替执行"失效。~~ 已由 `1fae36e`（`* text=auto eol=lf`）解决。
3. **生成器指纹只覆盖单文件**：包化后必须改成覆盖 datagen 的导入闭包，否则拆出去的模块改了也不会触发拒绝续跑。
4. **interop 下中断渲染**：WSL 下 `child.terminate()` 作用于 Linux 侧 interop 代理进程，Windows 侧 `Quantiloom.exe` 是否随之退出未验证（待核实）；`alive(child_pid)` 在 WSL 下检查的也是代理进程。
5. ~~数据集文档与现状不符：测试数写作"10 项"（实为 11）；证据文件写作 `docs/env-dataset-generation-evidence-2026-10-06.json`（实际在 `build/`，被 gitignore）；使用示例目录为 `build/env-smoke`、`build/env-three-band`（现有为 `*-0.7.4`）；pilot 配置未被文档提及。~~ 已由 `61341e0`、`81d5286`、`f36ed47`、`0d29c43` 修正（T0.4）。
6. `sample.json` 的 `extra_products` 以文件路径为键而非产品名，消费端只能按文件名反查产品名（可在 sample schema v2 中补 `truth` 映射，需单独决策）。另外 `frame_truth_material_id.exr` 的通道名是 `instance_id`（与实例 ID 文件同名），读取器必须按文件而不是按通道名区分两者（pilot 实测）。
7. ~~`.clang-tidy`、`.clangd` 是 Quantiloom C++ 仓库的配置，与本仓库无关。~~ 已加入 `.gitignore`（`460a57a`）。
8. 物理待核实：pilot LWIR 元数据 diagnostics 含 "material 'Car_Car_body' legacy scalar IR fractions are constant assumptions extended across camera response [4000, 12000] nm"，而数据集文档称车身绑定 Al n,k。可能只是热求解边界参数的提示，也可能说明 n,k 绑定未生效——需在 Quantiloom 侧核对，不在本仓库修。

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
    def version(self) -> str: ...                      # "Quantiloom 0.7.4"，配方判定用
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
| `dataset.index.json` | `quantiloomfusion.env_dataset` v1 | 完整样本列表、`split_group`、完成度 |
| `samples/<id>/sample.json` | `quantiloomfusion.sample` v1 | 每波段 `observations`：产品路径、`signal`（kind/unit/transfer/cfa）、`provenance.geometry`（内参、畸变 `k1_k2_p1_p2_k3`、`camera_to_world`、`right_down_forward`、像素中心偏移 0.5、左上原点）、`extra_products`（真值） |
| `dataset.plan.json` | `quantiloomfusion.env_dataset` v1 | 配方、场景假设、相机固定配置（只读，用于溯源与消融配对） |

`dataset_id` 标识配方；消融臂之间按 `sample_id` 配对（同场景/方位/俯仰/距离的 `sample_id` 在不同配方中相同）。schema 版本变化必须同时改生产端、消费端解析器与本表。

**B4 信号归一化**（`signal/`）

```python
class Normalizer(Protocol):
    name: str
    def params(self) -> dict: ...                              # 写入结果，参与方法指纹
    def __call__(self, image: np.ndarray, semantics: SignalSemantics) -> np.ndarray: ...  # float32
```

`measurement`/`rawdn`/`corrected` 量纲不同；红外 `display` 含场景相关拉伸，不当线性测量用。归一化策略必须显式选择并记录。

**B5 几何**（`geometry/`）

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

def oracle_warp(ref: Observation, src: Observation, *, edge_px: int, z_rel_tol: float) -> Warp
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

**B8 评价**（`evaluation/`）

```python
class Metric(ABC):
    name: str; version: int
    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]: ...

class ResultStore:   # 追加写 JSONL；键 = (dataset_id, sample_id, method 指纹, metric 名, metric 版本)；已有则跳过
    def has(self, key) -> bool: ...
    def append(self, key, values: dict) -> None: ...
```

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
- **T2.14** 审计工具迁入 `datagen/audit/`：模型包路径从计划的 `atmosphere_model_pack` 经 `native_path` 解析；生成器改为进程内调用审计。判据：WSL 下对 pilot 的一个 LWIR attempt 跑通闭合审计。
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

- **T4.1** 归一化策略；VIS 用 `display` RGB 还是 RAW 去马赛克（待决，两者都应可选）。
- **T4.2** `OracleRegistrar`、`ResizeRegistrar`、`InfinityHomographyRegistrar` 与 `registration_error`。
- **T4.3** 从 paper-exp 移植经典基线并推广到 N 波段（`port-from-paper-exp`）。
- **T4.4** 移植指标（EN、SD、SF、AG、SCD、Qabf、MS-SSIM、PSNR、SSIM），源参考类指标推广到多源。
- **T4.5** `ResultStore` + `experiments` 驱动 + 实验 TOML；第一次端到端：pilot 上「波段子集 × 经典基线 × oracle/naive 配准」。
- **T4.6** 互补信息量化：VIS → `truth_surface_temperature` 回归残差（留一场景组，报告折数与组构成）；条件互信息 I(Y; LWIR | VIS)。口径按数据集文档「评价协议建议」。
- **T4.7** 任务驱动评价（基于实例/材质 ID 的分割）——先写设计再实现。

### 阶段 5：正式采集与消融渲染（GPU 长任务）

- **T5.1** 冻结 datagen（记录冻结时的 commit），正式运行 `configs/env_dataset.toml` → `build/env-three-band-0.7.4`；定期 `verify`；每 50 个三元组的闭合审计结果汇总进本文。
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
| T2.13–T2.16 | 未开始 | |
| T3.1–T3.9 | 未开始 | |
| T4.1–T4.7 | 未开始 | |
| T5.1–T5.4 | 未开始 | T5.1 依赖 T2.15 完成并冻结 |

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
| 2026-10-12 | 可选 torch 模块以模块级 `__optional_deps__ = ("torch",)` 声明 | AST 可直接检查；比文件名约定灵活（深度模型可能占整个子目录） |
| 2026-10-12 | torch 索引由 cu126 改为 cu128，且只对 Windows 生效 | 与 paper-exp 一致（同版本、同索引，显存测量可比）；Linux 的 PyPI wheel 已带 CUDA |

## 8. 待决问题与风险

待决：

- 正式采集规模：按实测约 264 GPU 小时；是否先跑一个子集（如每场景 8 方位）以尽早开始阶段 4 的真实数据实验。
- 工况标签是否在配置/样本清单中显式化（目前只能从 `scene_id` 后缀推导）。
- `sample.json` 是否升 v2（真值按产品名索引）。
- VIS 融合输入：`display` RGB 还是 RAW 去马赛克；输出保留颜色还是灰度。

风险：

- 正式运行期间改动 datagen 闭包会使目录无法续跑 → 冻结规则（`AGENTS.md`）。
- interop 中断可能残留渲染进程 → T2.16。
- WSL 9p IO 可能拖慢训练数据加载 → T3.9。
- 依赖真实数据集的测试受 ACL 限制，只能由渲染账户运行。

## 9. 交接备忘

- 当前停在：T2.1–T2.12 完成：生成器已拆包，配方指纹覆盖 datagen 导入闭包（`recipe.generator_files`）。
- 下一步：T2.13 `FakeRenderer` + 无 GPU 的 run/publish/恢复/损坏/verify 测试；然后 T2.14 审计迁入、T2.15 收尾验证（含真实 smoke 渲染）、T2.16 interop 中断验证。从 WSL 调 Windows `python -c` 时 interop 会弄坏带空格的双引号参数，改用脚本文件。
