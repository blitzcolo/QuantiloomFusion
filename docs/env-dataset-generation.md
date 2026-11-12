# Quantiloom 三波段环境数据集生成

`tools/generate_env_dataset.py` 使用 Quantiloom 的普通 TOML 渲染入口，为 QuantiloomFusion 生成未配准的 VIS、MWIR、LWIR 数据集。场景来自 Quantiloom-Qt 的 `env_concrete`、`env_desert`、`env_grassland`、`env_seaside`，及其 23:00 夜间变体与低照度（月光级）变体。每个相机组姿态包含三次采集，三款相机的位置、朝向、分辨率、FOV、镜头畸变和退化不同；同一款相机的配置在整个数据集内固定。

## 数据集定位

本数据集是一个**受控基准**：在异构原生传感器（不同分辨率、FOV、畸变、PSF、噪声模型和相机位姿）下做未配准融合、配准鲁棒性与退化处理的定量研究。真值产品随每个相机一同导出：逐像素深度/距离、世界坐标、法线、覆盖有效性、实例 ID、材质 ID，以及**求解表面温度**（K）和每材质的波段等效发射率清单——对比度可以分解为"温度驱动"与"发射率驱动"两部分，也直接支持温度–发射率分离（TES）类评价。

它**不能**单独证明"三波段融合在物理上是必要的"。三波段互补的物理前提是存在 MWIR/LWIR 能提供而 VIS 不能提供的信息：热辐射差异、材料光谱差异、VIS 失效工况。本版本相对此前 `thermal_mode="authored"` + 常数发射率 + 共享噪声流 + 无几何真值的设置做了实质修订（见"物理建模"一节），但结论性断言仍需配套的消融条件（单波段、双波段、`ir_materials="constant"`、夜间臂）在同一协议下对比。把本数据集读成"未配准异构传感器基准 + 初步光谱互补证据"是恰当的；把它读成"三波段必要性已被证明"则不是。

渲染器仍在 0.x 开发期，二进制更新频繁：配方校验按 `Quantiloom.exe --version` 的版本字符串判定，不按可执行文件、SPIR-V 或 DLL 的内容哈希。版本号变化（例如 0.7.3 → 0.7.4）视为新配方，需要新的输出目录；CLI 与其同目录资源仍逐字记录到计划的 `renderer_observed` 中供溯源，但不参与判定。**规则：凡是改变数值结果的修复都必须升版本号**——0.7.3 时代曾把 LWIR 采样缺陷的修复落在同一版本号下，若不升版，按"同版本可续跑"规则修复前后的产品会混入同一数据集；本条自 0.7.4 起强制执行。在此基础上计划指纹另加 `physics_sources`：当 CLI 位于开发构建树（旁边有 `src/`）时，`src/shaders`、`src/libQuantiloom/atmos`、`src/libQuantiloom/postprocess` 三个物理相关目录的逐文件内容哈希参与配方判定——同版本号下改了着色器/大气烘焙/传感器链源码也不能续跑（编译出的 .spv/DLL 本身不门禁，否则每次链接都换指纹）。**自 0.10.0 起配置改用 SDK 安装版 CLI**（`H:/Quantiloom-SDK/windows_amd64/bin/Quantiloom.exe`）：0.10.0 只经 `share/quantiloom/`（带 SHADERSET 摘要校验）查找着色器，开发构建树的可执行文件旁边没有这个目录，在 attempt 目录里启动时找不到资源（`no Quantiloom resources found`）。SDK 由 `./build_wsl.sh` 在全部门禁通过后安装，与开发构建树是同一个二进制（2026-11-11 核对：大小、时间戳、`--version` 相同，SHADERSET 摘要 `cb0b2770…` 一致）。代价是安装目录旁没有 `src/`，`physics_sources` 为空，同版本号下的重编译只能靠"改数值必升版本号"的规则拦住。已验证版本为 **Quantiloom 0.10.0**（2026-11-11 起，SDK 安装版 CLI；此前 0.8.3，2026-10-26 迁移；与 0.7.4 的同任务对照见"验证记录"的迁移节，0.7.4 渲染的 `build/env-*-0.7.4` 是另一个配方，只能 `verify`）。最低要求仍是 **0.7.4**（在 0.7.3 的几何/温度/材质真值、`dataset.acquisition_index`、LWIR 采样修复之上，本数据集进一步依赖其相机网格 Voronoi 胞盒平均的大气积分语义——0.7.2 及更早版本的 LWIR 相机产品系统性偏低约 10%，0.7.3 的路径项仍有最高约 −23% 的光谱混叠偏差，均见"大气"节；旧版 CLI 无法产出真值产品）。

生成端需要 CLI 和场景资产。融合端只需分发数据集里的 EXR、JSON 元数据和索引，不需要 Quantiloom 的源码、SDK 或可执行程序。数据集自身的 `verify` 命令同样不读取原始场景，也不调用渲染器。

本文当前同时承担四种角色：使用说明（运行环境/使用方法/断点恢复）、技术报告（物理建模/验证记录）、变更与证据历史（修复叙事）、Quantiloom 能力清单（附录）。正式发布时建议拆分：面向使用者的 README 保留运行与产品说明，数据集说明参照 *Datasheets for Datasets*（Gebru 等，CACM 2021）组织——"物理建模"与"实验边界与限制"两节可直接平移进 datasheet 的 motivation/composition/collection-process/preprocessing/maintenance 各段，修复叙事与版本演变收进仓库的变更记录。

## 运行环境：Windows 与 WSL2

脚本可在 Windows Python 下运行，也可在 WSL2 的 Linux Python 下运行；渲染器始终是 Windows 版 `Quantiloom.exe`，WSL 下通过 interop 调用。

- 配置中的 `qt_root`、`cli` 以及场景 TOML 内的资产路径写 `H:/...` 或 `/mnt/h/...` 均可，脚本按当前系统翻译，并在计划与任务 TOML 中统一记录 Windows 形式。因此**同一数据集目录可以在两个系统间交替执行 `plan`/`run`/`verify`**，配方指纹与任务哈希保持一致。
- WSL 下 `--out` 必须位于 `/mnt/<盘符>/` 之下（即 Windows CLI 能读写的位置），否则脚本提前报错。
- 场景与模型资产同样需要位于 `/mnt/<盘符>/` 之下；计划阶段会对 CLI 无法触及的路径给出警告。
- 依赖：Python ≥ 3.11。计划生成与文件哈希校验只用标准库；默认的像素校验还需 `numpy` 与 `OpenEXR`（两个系统下都是 `python -m pip install numpy OpenEXR`）。没有它们时可把配置改为 `verification = "hashes"`。
- 数据集目录不能整体搬迁后继续 `run`：红外任务 TOML 内嵌了 `--out` 的绝对路径（派生 `solar_lut`），换路径会触发 `planned file changed`；只跑 `verify` 不受影响。
- 不要从两个系统（或两个进程）对同一输出目录同时执行 `plan`/`run`：Windows 与 WSL 的文件锁机制不同，同目录写互斥只在同一系统内成立。

## 配置和覆盖范围

正式配置为 `configs/env_dataset.toml`，小样本配置为 `configs/env_dataset_smoke.toml`。小样本配置用于验证采集链路，采用较少采样和较低分辨率，不适合作为论文中的质量基准。另有全分辨率 pilot 配置 `configs/env_dataset_pilot.toml`：相机分辨率、SPP 32、`wavelength_samples = 12` 与物理开关同正式配置，只取 concrete 的白天/夜间/月光三个条目 × 8 个方位角（俯仰 12°、单一距离），共 24 个三波段样本，闭合审计每 8 个三元组一次；用于在正式采集前实测吞吐和检查全分辨率产品（本机 `build/env-pilot-0.7.4` 已在 0.7.4 下完整渲染；`build/env-pilot-0.8.3-asis` 是同一批任务 TOML 在 0.8.3 下的前 3 个三元组，只用于版本对照）。

正式配置覆盖 **12 个场景条目**（4 个白天 + 4 个 `*_night` 夜间变体 + 4 个 `*_moonlight` 低照度变体）、8 个方位角、2 个俯仰角和 3 个距离，每个条目 48 个视角，共 **576 个三波段样本、1728 次相机采集**（2026-11-11 由 24 × 3 × 3 = 2592 个样本缩减：相邻 15° 方位角的视图对评测高度冗余；去掉 5° 俯仰——那是对地面 85° 的观测天顶角，漫射材质的各向同性发射率近似在这里最不可靠；小规模使配置修改后的整体重渲可以承受）。这是计划规模，脚本不会在创建计划时开始完整渲染；夜间与月光臂共享同一条 23 时热求解轨迹（热缓存使每场景只需两条求解：白天与 23 时）。

| 参数 | 正式配置 |
| --- | --- |
| 方位角 | 0° 至 315°，间隔 45° |
| 俯仰角 | 12°、25° |
| 距离 | 原场景相机到目标的距离乘以 0.8、1.0、1.25 |
| 目标点 | 原场景的 `camera.look_at`；可由每个 `scenes` 条目的 `target` 覆盖 |
| 重复次数 | `sampling.repeats`，正式配置为 1（必填键，无代码默认值） |
| 主渲染 SPP | 32 |
| 物理相机 `wavelength_samples` 参数 | 12 |

方位角从世界坐标的 +Z 方向起算，朝 +X 增加。俯仰角为目标上方的角度。`radius_m` 可以覆盖一个场景的基础轨道半径，单位为米；`target` 使用场景世界坐标。脚本会按 `world_units_to_meters` 换算相机偏移与轨道距离。

每个样本遍历 `scene × azimuth × elevation × radius_scale × repeat`，随后固定生成三个波段。相机偏移在整个相机组的局部坐标系中定义，随相机组一起转动，因此不同视角下的基线长度保持一致。

| 相机 | 原生分辨率 | 名义垂直 FOV | 局部位置偏移，米，右／下／前 | 朝向偏移，度，yaw／pitch／roll | PSF σ |
| --- | --- | --- | --- | --- | --- |
| VIS | 640 × 360 | 45° | [0, 0, 0] | [0, 0, 0] | 0.45 px |
| MWIR | 320 × 256 | 38° | [0.035, 0.002, 0.008] | [0.12, −0.08, 0.03] | 0.8 px |
| LWIR | 256 × 192 | 42° | [−0.045, −0.003, 0.012] | [−0.15, 0.10, −0.04] | 1.1 px |

三款相机都是建模假设的通用传感器预设（`generic_cmos`、`generic_cooled_photon_ir`、`generic_uncooled_thermal_ir`），不是实测标定设备。VIS 用 RGGB CFA 的 CMOS 模型，响应 400–700 nm；MWIR 用冷却光子探测器的 3–5 µm 响应；LWIR 的任务光谱范围是 8–12 µm，落在热探测器预设的 8–14 µm 响应曲线内。VIS 与 MWIR 的读出噪声、PRNU、DSNU、满阱、电荷到 DN 的转换和位深分别配置；LWIR 使用 NETD、响应度、时间常数和预热设置。三款相机采用不同 Brown–Conrady 系数。具体值以配置文件及导出的实际相机配置为准。

### 固定图样、行列条纹与 NUC

每台相机在 `[cameras.*]` 里显式给出 NUC 开关（`apply_nuc`）与残余比例（`nuc_residual_fraction`），以及固定图样的逐像素、列、行三个分量——不依赖渲染器预设的默认值，预设改了数据集也不跟着变。三个分量的方差相加，总量保持原值，按"列为主"分配（非制冷长波的残余非均匀以列条纹为主：每列一个放大器与 ADC；逐像素独立的图样在单帧上与略大的时域噪声没有区别，条纹才是融合算法真正难处理的部分）：

| 相机 | 量 | 总量 | 逐像素 | 列 | 行 | NUC |
| --- | --- | --- | --- | --- | --- | --- |
| LWIR | 增益 | 3 % | 0.016 | 0.023 | 0.010 | 两点校正，残余 3 % |
| LWIR | 偏置 | 2 K | 1.10 K | 1.55 K | 0.65 K | 同上 |
| MWIR | 增益 | 1 % | 0.0074（`prnu_sigma`） | 0.006 | 0.003 | 两点校正，残余 3 % |
| MWIR | 暗信号 | 8 e⁻ | 5.9 e⁻（`dsnu_e_rms`） | 4.8 e⁻ | 2.4 e⁻ | 同上 |
| VIS | 增益/暗信号 | 0.5 % / 0.5 e⁻ | 不变 | 0 | 0 | 不校正 |

此前 LWIR 走预设自带的逐像素图样（3 %、2 K、残余 3 %），MWIR 只设了 `enable_fpn`、没有校正——中波的校正信号带着完整的 1 % 响应非均匀（约 0.3 K，是它时域噪声的数倍），长波只留约 0.06 K，三台相机的口径不一致。现在 MWIR 与 LWIR 同为残余 3 %（长波约 0.06 K，中波约 0.01 K）。VIS 不校正：0.5 % 的响应非均匀在满阱以内始终低于散粒噪声（满阱 20000 e⁻ 时散粒噪声 0.7 %），`nuc_residual_fraction = 1.0` 只表示"未校正"。渲染器 0.10.0 修正了光子型 NUC 的语义（`b1f0051`：没有校正图时 `apply_nuc` 留下 r 倍的固定图样，与热型一致；此前是 1 + r 倍）；行列分量是 0.10.0 新增的（`01b8ba1`），种子键不含采集序号，所以和逐像素图样一样在全部视角间保持同一张。数值都是假设值，不是器件标定。

平场实测（2026-11-11，`build/_qa/flatfield-0.10.0`：Quantiloom 的 ε=1、300 K 黑体炉腔，换上本数据集 pilot 的 MWIR/LWIR 传感器段，各渲染两次、`acquisition_index` 不同）：LWIR 原始 DN 的列均值离散为信号的 3.31 %（行均值 21.9 DN 对列 52.0 DN，列为主），校正信号的列离散是原始的 0.030 倍，`tapp` 列均值离散 0.061 K、逐像素 0.095 K（含 NETD 0.05 K）；MWIR 原始列均值离散 0.59 %，`tapp` 列均值离散 0.008 K、逐像素 0.113 K（以时域噪声为主；校正/原始列离散比 0.050，因为时域噪声也进入了列均值）。两次采集的原始列均值相关系数 LWIR 0.999997、MWIR 0.9986——固定图样跨采集保持同一张。

渐晕关闭：渲染器的 `sensor.optics.cos_fourth_vignetting` 缺省为 false，生成器不设置它（0.10.0 起镜头自身发射随视场角的衰减也只在开启渐晕时生效）。三款相机的画幅内辐射响应因此是均匀的，画面边缘的亮度/温度变化全部来自场景。

名义 FOV 指畸变前的垂直视角。脚本由分辨率、像元尺寸和 FOV 计算焦距，并验证导出内参。畸变后的边界视场由原生投影模型决定。

改动采样列表可扩大或缩小覆盖范围；改动任一相机配置会产生另一个实验数据集，应该使用新的输出目录。脚本不在单个数据集内随机改变相机 FOV、分辨率或退化强度。

### 场景变体（`overrides`）

`[[scenes]]` 条目可带 `overrides` 表、`solar_scale` 和 `vis_exposure_s` 三个可选键。覆盖按**整个命名段**替换：写一个 `[scenes.overrides.lighting]` 等于整体替换场景 TOML 的 `[lighting]` 表，不做字段级合并——被省略的键回到场景文件或加载器默认值，不会残留原值。`*_night` 变体以此完整重写 `[lighting]`（太阳/天空辐亮度置零、太阳改为 23 时真实的地平线下方向、太阳表按 `solar_scale = 1e-12` 缩放后保留——见"夜间臂"）、`[thermal]`（同一 forcing 文件，求解到 23 时）与 `[atmosphere]`（同一大气预设，但 `t_ground_K`/`air_temperature_k`/湿度取 forcing 文件 23 时那一行，不沿用白天值）。`*_moonlight` 变体额外使用 `solar_scale`（作用于派生太阳表，乘在白天的幅度对齐之上，见"低照度臂"与"白天太阳谱的幅度"）与 `vis_exposure_s`（VIS 长曝光）。删除变体条目即回到 192 样本的纯白天集。

## 物理建模

本节列出本版本相对早期"等温 + 常数发射率"设置的全部修订，以及仍然是假设、不能当实测的部分。

### 温度场：`thermal_mode = "solved"`

四个环境场景本身携带完整的 `[thermal]` 配置（逐材质热导率、密度、比热、厚度、对流系数、短波吸收率、湿度因子）与逐日 forcing 文件（`assets/forcing/*_day.csv`）。正式配置启用求解模式：渲染前按场景做一次表面能量平衡求解，同一求解结果由 Quantiloom 的热求解缓存复用于该场景全部视角。生成器会拒绝为没有启用 `[thermal]` 的场景指定 `"solved"`。

smoke 验证中的实测数值（0.7.4，LWIR 表观温度产品，64×36）：混凝土白天表面温度真值均值 318.5 K、`tapp` 均值 315.5 K；23:00 夜间真值 300.0 K、`tapp` 298.2 K。逐材质看，混凝土（ε_LWIR≈0.91，JHU 绑定）的 `tapp − T_s` 夜间 −1.7 K、白天 −3.0 K——与 ε≈0.91 加天空反射的理论量级吻合；Al n,k 车身（ε≈0.05，近乎冷镜；这是 0.7.4 时的场景版本，现车身为白色涂层，见材料节）夜间 −5.9 K、白天 −6.0 K，`tapp`≈294 K 逼近反射天空的温度——镜面效应而非缺陷；ε≈0.10 的铬球反射体 `tapp` 294–306 K，同样跟随反射环境。白天存在约 20 K 的日照驱动分布、夜间趋向平衡——这正是 MWIR/LWIR 能提供而 VIS 不能的那类信息。`"authored"` 仍保留为消融开关（等温对照组）。

### 材料光谱：`ir_materials = "spectral"`

四个场景的材质已重绑到 Quantiloom merged 光谱库（`quantiloom_basis_v3_merged.qlbin`，3524 种材料、16 维 NMF 基，由 USGS splib07a、ECOSTRESS 系测量与 RefractiveIndex.info 数据合并烘焙）。同一材质名在 VIS 与 IR 中使用同一条测量光谱：**渲染端**的发射率由测量反射率（或 n,k 的 Fresnel）按基尔霍夫关系 ε = 1 − ρ 决定（不透明表面）；`[[materials]]` 块内的 `ir_emissivity` 只作为热求解的辐射边界参数保留（车身为白色涂层、0.92，见下），不参与渲染光谱。未绑定材质（玻璃、金属球、车内饰、人物等）保留场景作者的常数发射率，缺省 0.9。`"constant"` 模式则回到旧的纯作者发射率行为，供消融对照。

绑定清单（库内名即 measurement ID）：

| 场景材质 | 库条目 | VIS | MWIR | LWIR [6.5–15] / [8–12] | NMF RMSE (LWIR/VIS) |
| --- | --- | --- | --- | --- | --- |
| ConcreteGround（混凝土场景） | Construction Concrete (0598UUUCNC) + Asphaltic concrete (0425UUUASP) | 1.00 | 1.00 | 1.00 / 1.00 | 0.0012–0.0015 / 0.0021–0.0027 |
| ConcreteBlock | Construction Concrete (0598UUUCNC) | 1.00 | 1.00 | 1.00 / 1.00 | 0.0012/0.0021 |
| HouseWall | White rubberized coating (0795UUURBR) | 1.00 | 1.00 | 0.88 / 1.00 | — |
| HouseRoof | Asphalt roofing shingle (0597UUUASP) | 1.00 | 1.00 | 1.00 / 1.00 | — |
| HouseDoor | Pine Wood (0404UUUWOD) | 1.00 | 1.00 | 0.71 / 1.00 | 0.0026/0.0062 |
| Car_Car_body | White rubberized coating (0795UUURBR)（白色涂层代替车漆，见下） | 1.00 | 1.00 | 0.88 / 1.00 | — |
| Car_M_Wheel_ALL.002 | Black unspecified rubber (0833UUURBR) | 1.00 | 1.00 | 0.88 / 1.00 | 0.0012/0.0004 |
| Car_M_BODY_black.004 | Black gloss paint (0403UUUPNT) | 0.84 | 1.00 | 0.88 / 1.00 | 0.0013/0.0005 |
| DesertSand | Brown loamy fine sand (87P3468) + Basalt (Basalt_7) | 0.88 | 1.00 | 0.88 / 1.00 | 0.0007/0.0016 |
| Rock（各场景） | Basalt (Basalt_7) | 0.88 | 1.00 | 0.88 / 1.00 | 0.0013/0.0016 |
| GrassGround | Grass (grass) + Dry grass (drygrass) | 1.00/0.93 | 1.00 | 0.88 / 1.00 | 0.0003/0.0027 |
| BeachSand | Brown fine sandy loam (87P3665) + Brown loamy fine sand (87P3468) | 0.88 | 1.00 | 0.88 / 1.00 | 0.0006/0.0022 |
| SeaWater | —（见下，复折射率绑定） | — | — | — | — |

coverage 列的口径是**测量区间并集**：光谱烘焙器（`spectral_processor.py`）按源数据相邻采样点间距超过 `hole_um = 0.25 µm` 处断开为多个实测区间，逐段与波段网格求交后求和，再按网格宽度归一。此前版本按源数据最小/最大波长取凸包重叠，把内部空洞计成已覆盖——SLUM（Kotthaus 等，ISPRS J. 2014）的条目正属此类：它只有便携 FTIR 实测的 VIS–SWIR 与 8–14 µm 两段，**2.5–8 µm 没有数据**，凸包算法会将其库字段 MWIR 误计为 1.00。**因此本版本不再绑定任何 SLUM 条目**：此前 ConcreteGround（C004+A001）与 Car_Car_body（Z002）的 MWIR 光谱都是 NMF 基跨 5.5 µm 空洞的插值，而 MWIR 反射太阳项约占地物热辐射的 10%（见下），不能让面积最大的材质靠插值给出。替换为全波段连续测量的 JHU 条目（混凝土/沥青）后，任务波段内不再存在跨空洞插值的材质（车身现绑定同样全波段连续的白色涂层实测谱，见下）。LWIR 列 0.71/0.88 对应源数据约止于 12.5/14 µm，任务波段 [8–12] 均完整包含；波段边缘为外推/截断行为，不是新测量。多组分材质（`spectral_material_refs` 列表）由 `spectral_unmix="auto"` 在逐像素端元混合中展开，绑定条目总数不超过配置上限。

NMF RMSE 的物理量级换算：典型 ~0.006 的反射率误差按 ε=1−ρ 进入发射率，在 10 µm、300 K 下约对应 0.4 K 的系统性表观温度偏差（(0.006/0.95)/(1.6 %/K)）——远小于白天 ~6 K 的空间结构，但与非制冷探测器 NETD 同量级。

复折射率（n,k）绑定按源数据波段覆盖决定去留：Al（0.124 nm–200 µm，裸铝部件 `Car_M_IRON.004`）、Cr（248 nm–62 µm，铬球）与纯水（0.2–200 µm，海面）的测量覆盖全部任务波段而保留；玻璃用 Treharne 数据仅 301–1498 nm，在 MWIR/LWIR 任务中被剥离——与其用不覆盖波段的数据驱动透射光学，不如让绑定材质回落到测量反射率、未绑定材质回落到作者发射率。**这与真实物理方向一致**：钠钙玻璃在 LWIR 本来就不透明（ε≈0.88 的吸收体），数据集不会产生"红外里透明玻璃"这类错误互补性。VIS 中 n,k 绑定全部保留，玻璃保持真实透射。

**Car_Car_body 绑定白色涂层的实测谱**（以 Quantiloom-Qt 当前 `assets/configs/env_*.toml` 为准，2026-11-11 核对）：四个场景的车身都是 `White rubberized coating (0795UUURBR)`（白色涂层代替车漆），渲染发射率按基尔霍夫由实测反射率给出，LWIR 约 0.92；热求解按 3 mm 涂漆铝板建模（`ir_emissivity = 0.92`，热导率 200 W/m·K，绝热内边界），热容约为地面的千分之一，几分钟内就跟上日照与天空。此前文档写的"车身绑定 Al 的 n,k（Rakic 1998），LWIR 发射率约 0.03–0.05 的冷镜"已经过期——那是 0.7.4/0.8.3 时期的场景版本，那些版本的数值（例如下文 smoke 记录里 Al 车身 `tapp − T_s` 约 −6 K）只描述当时的数据集。场景里仍保留的有光学常数的光滑表面是铬球（Cr，Rakic-LD）、裸铝部件 `Car_M_IRON.004`（Al，Rakic）与海面（纯水，Hale & Querry）：渲染器 0.10.0 起，它们在红外按方向菲涅耳的镜面瓣渲染（`b7091d7`：反射率 F(θ, λ)、发射率 1 − F(θ, λ)，反射镜面方向上的入射辐亮度）；此前的版本把法向反射率 R0 当成各向同性的总反射率用，而且材质上的 `ir_emissivity` 标量会盖过 n,k，所以铬球、铝件和玻璃在红外其实从未用上 n,k。玻璃的 n,k 只覆盖 301–1498 nm，在 MWIR/LWIR 按作者给的半球发射率 0.84（EN 673 的修正发射率）作不透明漫射面渲染。

**SeaWater 不绑光谱条目，绑复折射率**（以 Qt `env_seaside.toml` 为准）：海水是界面加体吸收的问题。`[refractive_index]` 绑定 Hale & Querry（1973）纯水 n,k（0.2–200 µm，覆盖全部任务波段），它描述的就是界面本身，不与体吸收重复计数；可见光里 n（1.333–1.339）取代 GLB 的 IOR 1.33，体衰减仍由 GLB 的 attenuation 给出（k 不额外叠加吸收）。渲染器 0.10.0 起红外按方向菲涅耳镜面瓣渲染：水在 10 µm 法向只反射约 1 %，85° 入射时约 55 %，所以掠射视角下远处海面的发射率明显下降、更多反射天空——此前的版本按法向反射率各向同性处理，把掠射视角的海面渲染成水温的黑体。热求解用的是**半球**发射率 `ir_emissivity = 0.95`（Hale & Querry n,k 的余弦加权半球菲涅耳反射率按 300 K、8–12 µm 普朗克加权：0.9488；3–5 µm 为 0.932），而不是此前的法向值 0.99；渲染不读这个数。海水按 1.5 m 水层、内边界固定 291 K、自由蒸发（`wetness_factor = 1`）求解，一天之内只变化一度左右，而旁边的沙滩摆动约 15 K。粗糙度没有 Cox–Munk 坡度分布（见"实验边界与限制"）。

**人物是给定温度的热源，不参与热求解**：`Man_Cesium_Man-effect` 在四个场景中均以 `ir_temperature_k = 307`（皮肤温度）、`ir_emissivity = 0.98` 给定——人有血液循环与代谢，表面能量平衡里没有这两项，参与求解的人体会冷却到地面温度，这恰恰是人永远不会出现的情况。此前文档写的"310 K 核心温度 + 5 cm 组织层的固定温度内核、皮肤求解到 305–309 K"已经过期。合成库里没有皮肤光谱，人物在可见光保持 RGB 上采样（贴图是风格化贴花）。

### 大气：`atmosphere_mode = "source"`

保留场景绑定的大气配置，并新增数据集级 `atmosphere_model_pack` 键把**四个场景统一到同一 NN 模型包**（`H:/Quantiloom-dev/assets/atmos_models`）——大气模型是配方变量，不允许逐场景解析到不同实现；场景之间只保留预设参数差异（concrete、grassland、seaside 为 `clear`，desert 为 `turbulent_clear` 且 `vis_km = 20`，及各自的 `t_ground_K`、湿度）。同一场景的白天、夜间、低照度三个条目共用同一个大气预设（气溶胶类型与能见度），只有 23:00 forcing 行给出的地表/气温、湿度与天空温度随时刻不同——三种工况之间的比较因此是照度的单变量对照，夜间地表温度所依据的热历史（晴空直射的白天）与成像大气自洽。渲染器 0.10.0 另有一条硬规则（`ConfigResolve.cpp` 的 `ReferenceSunAtmosphereConflict`）：太阳在地平线以上且 `solar_lut` 是地面参考谱（列名含 ETR/extraterrestrial，ASTM G-173 即是）时，大气必须是晴空（`vis_km ≥ 20`、`ihaze = 1`、无云无雨），否则直接报错。生成器派生的太阳谱 CSV 保留原列名，所以低照度臂（太阳仰角 5°）同样受这条规则约束——此前 concrete 夜间/低照度臂沿用的 8 km 城市霾（ihaze 5）与 desert 的 18 km 在 0.10.0 下都会被拒绝。按 `{lwir,mwir,nir,swir,vis} × {ground,slant,sky} × {tau,lpath,ldown}` 组织：路径透射、路径辐射与**天空下行辐射**都由网络给出，LWIR 中 L = εB(T) + (1−ε)L_down 的下行项因此是有模型的，不是常数背景。

**湿度只写一个键**：Quantiloom 0.8 把 `atmosphere.rh`（分数，NN 输入）与 `atmosphere.relative_humidity`（百分数）解析为同一个湿度，两者都给且不一致时告警，并用 `rh` 同时驱动 NN 与热平衡。Qt 白天场景两键并存且不一致（concrete 0.75 对 50 %、desert 0.15 对 50 %、seaside 0.70 对 50 %），生成器在两键并存时只保留 `rh`，任务 TOML 写的就是渲染器实际使用的湿度。数值不变：NN 自 0.7.4 起就读 `rh`，热求解逐行读 forcing CSV 的湿度列（四份 `*_day.csv` 都是含湿度的 8 列格式）。

与渲染的太阳项的组成关系：`L_pixel(λ) = τ_view(λ) × L_surface(λ) + L_path(λ)`——表面照明（太阳直射 + 天空漫射）完全走 `solar_lut` 绑定表，NN 只提供视线透射、路径辐射和热下行。因此 ASTM G173 地表谱**不会被 NN 二次衰减**；代价是 LUT 的直射/漫射配比与 NN 大气状态不严格自洽，属已声明的近似。

NN 包在 CPU 烘焙时以太阳天顶角为输入：天顶角 >85° 判定夜间，太阳散射项整体归零（这正是夜间臂不依赖 `[lighting]` 置零就天然正确的原因——零星空 RGB `sky_radiance` 只作用于解析天空回退，碰不到 NN 的 `ldown`）；天顶角恰在 85° 的掠射太阳求出黄昏级弱路径辐射——月光臂利用的正是这一档。预设表见文末能力清单；`"off"` 移除 `[atmosphere]` 供消融。已知限制：NN 模型包是 MODTRAN 系数据的回归近似，不是在线逐线计算；`sky_model` 标量设置在 NN 激活时被忽略。

**训练域边界**：NN 网络的 `cos_sun_zenith` 输入域是 [0.087, 1]（天顶角 0–85°，太阳仰角 ≥5°），**没有夜间/黄昏域**。0.7.3 起，非热波段（VIS/NIR/SWIR）的 `lpath` 在天顶角 >85° 时直接归零而不是沿用钳制到 85° 的网络输出——钳制值会产出物理上没有依据的黄昏级"雾幕"。LWIR 的 `lpath` 是纯热路径发射，不受此限制。

**单波长采样语义（0.7.3 修复、0.7.4 改胞盒平均）**：相机波长积分按单波长逐点渲染再梯形求和，每个单波长的大气量（τ/lpath/ldown）由 NN 通道网格给出。NN 输出本身是**每个波数通道的盒平均量**：LWIR 通道在 ν∈[833,1250] cm⁻¹ 上以 1 cm⁻¹ 等距（λ 间距 6.4–14.4 nm），MWIR 约 0.9–2.5 nm、VIS <0.7 nm。

- **0.7.2 及更早（缺陷）**：单波长烘焙要求 NN 通道落在请求 λ ±2 nm 窗口内，LWIR 下绝大多数 λ 捕获不到任何通道——τ、lpath、ldown **全部静默归零，该样本整帧只有自发光像素非零**。相机采样网格的 12000 nm 端点距最近通道（12004.8 nm）4.8 nm 恰好落空，该样本全黑；5 点网格（{8000,9000,10000,11000,12000}，smoke 的 `wavelength_samples=4`）中端点的梯形权重为 (500/4000)=**12.5%**，造成 LWIR `corrected`/`tapp` 系统性偏低约 10–12%（`tapp` 比真值低 9–14 K，远超发射率不足 1 应有的 1–4 K）。
- **0.7.3（最近通道回退）**：窗口内无通道时取波数最近通道，消灭了归零帧；但逐点最近通道采样在 1 cm⁻¹ 谱线结构上是**混叠**——离线全通道积分对比（`tools/_spectral_integration_audit.py`）显示 τ ±2%、lpath −23%、ldown 等效约 −1.7 K 的残差，且**不随采样点数收敛**（N=49 仍在 ±5–17%）。
- **0.7.4（Voronoi 胞盒平均）**：渲染器从相机管线取得确定性的波长积分网格，每个样本的大气烘焙改为其 Voronoi 胞（相邻采样中点为界）内全部 NN 通道的盒平均，胞宽的显式参数同时计入烘焙缓存哈希。离线复核：N=13（正式 `wavelength_samples=12`）下 τ/lpath/ldown 的带均偏差全部 <0.15%（LWIR ldown +0.019%）；N=5 的 smoke 网格在 MWIR 上仍有 −2.7%/+6.8%（τ/lpath）的胞均化残差——smoke 数值按此口径解读，正式采集不受影响。这是 `wavelength_samples=12` 的定量依据。

修复后夜间 LWIR 采集的 ConcreteGround `tapp − T_s` 为 −1.7 K（ε=0.91 的 JHU 混凝土）、白天 −3.0 K，与发射率加天空反射的理论量级一致。**使用 0.7.2 或更早版本生成的 LWIR 相机产品在定量上不可信，0.7.3 产品的路径项（lpath）有最高约 −23% 的带内混叠偏差，均需要重渲**。

**太阳散射缩放（0.7.3 新增）**：`atmosphere.solar_scatter_scale`（缺省 1.0，负值钳 0）按一阶线性缩放 NN 路径辐射的太阳散射分量，供月光臂使用。MWIR 的 `lpath` 网络输出显式 `SOL_SCAT` 分量，只缩放它；VIS/NIR/SWIR 的 `lpath` 网络只有单一 `TOTAL_RAD` 输出（这些波段的大气热发射可忽略，整项实质上就是太阳散射），故缩放整项、夜间归零；LWIR `lpath` 为热发射不缩放。该字段计入大气烘焙缓存哈希，改动即重烘。这是"散射 ∝ 源辐照度"的一阶近似，不做完整月球辐射传输。

### 太阳谱：`solar_ir_policy = "planck_tail"`

源 ASTM G173 表只覆盖到 4 µm。`"planck_tail"`（必填键，无代码默认值）以 4 µm 末行锚定 5778 K 普朗克形状延伸到 15 µm，各辐照度列分别延伸；派生表按 `%.12e` 固定格式写出，保证两个系统下逐字节一致。**该尾部穿过 4.2–4.4 µm CO₂ 吸收带与 5–8 µm 水汽带，会高估 MWIR 上段的太阳反射分量**。量级估算：3–5 µm 地表太阳辐照度约 10 W/m²，漫反射率 ρ≈0.15 的地物反射辐亮度约 0.4 W/m²·sr，而 318 K 的带内热辐亮度约 3 W/m²·sr——受照区域的反射太阳项约占热发射的 **10% 量级**，4 µm 附近两者甚至可比，并非先前估计的不足 1%。这意味着普朗克尾的近似误差与"日间 MWIR 是否有显著太阳分量"这个消融变量同量级，MWIR 波段消融结论不应建立在它之上；补齐 MODTRAN DATA 或构建 libRadtran 太阳表的优先级相应提高（见下方 QLTrans 段）。

**QLTrans（仓库自带的 MODTRAN 封装）本应是自洽的替代路径**——它内嵌 MODTRAN4 可执行体，且 NN 大气本身就用 MODTRAN 系数据训练，两者同源才一致。实测结论是**当前检出缺 MODTRAN 分子带模型 DATA 文件**（`MOD4v1r1.EXE` 在，但 DATA 目录不含所需波段模型文件；MODTRAN 数据文件不可再分发），QLTrans 磁带能生成、辐射计算无法执行。`reference-only/libRadtran-2.0.6` 在仓库内但只有源码未构建。所以本版本保留 `planck_tail` 并把它标注为显式近似；补齐 MODTRAN DATA 或构建 libRadtran 后按场景各生成一张 0.28–15 µm 太阳表即可替换，属明确后续工作。夜间变体无 `solar_lut`（无太阳项），月光臂的太阳表是同一份派生表再乘 `solar_scale`。选择 `"source"` 则保持原表不延伸（>4 µm 无太阳项）。

### 白天太阳谱的幅度：`daylight_solar_scale = "forcing_direct"`

渲染用的太阳谱是 ASTM G-173（太阳天顶角 48.2°、AM1.5 的地面参考谱，直射+环日列在 280–4000 nm 上的梯形积分 900.14 W/m²），幅度不随场景变化；热求解用的是 Quantiloom-Qt 的 forcing 文件，其直射辐照度随太阳高度变化（`scripts/gen_env_forcing.py`：1158·exp(−浑浊度·大气质量)）。不处理的话，海边场景（16 时、太阳高度 21.1°、forcing 直射 664.4 W/m²）里照亮画面的太阳比加热地表的太阳强三分之一。

`"forcing_direct"`（必填键，无代码默认值；`"none"` 不缩放）让生成器对每个带 `solar_lut` 的条目计算 scale = forcing 文件在场景采集时刻的直射辐照度 ÷ 参考谱直射列（`solar_lut_columns` 的第一项）在原表上的积分，乘进派生太阳谱 CSV 的每一列；采集时刻与 forcing 文件取**源场景 TOML** 的 `[thermal] time_h`/`forcing_file`（低照度臂的覆盖项把 `[thermal]` 换成了 23 时，取的仍是场景构图的那个白天时刻）。低照度臂的 `solar_scale = 2e-6` 乘在这个之上；夜间条目的太阳表只为让渲染器推导短波吸收率而保留，`solar_scale = 1e-12` 同样乘在这个之上（见"夜间臂"）。NN 路径辐射的太阳散射缩放（`atmosphere.solar_scatter_scale`）只跟随低照度的 2e-6，不跟随这个幅度对齐。派生 CSV 保留原列名（渲染器靠列名识别地面参考谱并执行晴空规则）。计划的 `scenes[].assumptions` 记录 `solar_scale` 与 `solar_scale_derivation`（两侧直射值、时刻、参考列、对齐比、臂系数）。

正式配置实测（`build/_parity/20261111-c3-formal`）：

| 场景 | 采集时刻 | forcing 直射 W/m² | scale |
| --- | --- | --- | --- |
| concrete | 14 h | 933.5 | 1.0371 |
| desert | 11 h | 981.9 | 1.0908 |
| grassland | 10 h | 910.3 | 1.0113 |
| seaside | 16 h | 664.4 | 0.7381 |

这只对齐幅度，不改谱形：G-173 的谱形对应 AM1.5，海边 21° 太阳（AM≈2.8）的真实直射谱更偏红，这部分没有建模。渲染器推导的短波吸收率是反射率按太阳谱加权的比值，不受幅度影响。

### 噪声与随机种子：`noise_seed_policy = "per_acquisition"`

相机噪声流按 `pixel / acquisition / noiseClass / counter` 分键，其中**固定图样类（PRNU/DSNU）的键里本就不含采集序号**——这一实现事实让"FPN 跨帧稳定"与"时域噪声逐样本独立"可以同时成立，不需要取舍：

- `per_acquisition`（正式配置）：每个波段一个固定 `noise_seed`——PRNU/DSNU 图样跨全部视角保持同一张，和真实相机一致；同时生成器把 `fingerprint(seed, "acquisition", sample_id, band)` 写入任务 TOML 的 `dataset.acquisition_index`，渲染器 `camera::CaptureState` 以它为时域噪声类的计数器，每次采集得到独立的散粒/读出实现。
- `per_sample`：每次采集一个种子，FPN 也随样本变化（上一版的行为，保留作消融）。
- `fixed_detector`：同波段所有视角共享同一随机流——FPN 一致但时域噪声也逐像素相同，仅供对照。

由此确认的三条结论：

- 相机产品仍不吃 `renderer.seed`（该种子只影响 `frame.exr`）；其随机性由 `sensor.noise_seed` + `dataset.acquisition_index` 驱动，每次 CLI 调用缺省从序号 0 开始；
- `per_acquisition` 下同波段跨视角的 FPN 图样严格一致（可逐像素验证），时域噪声独立；`repeats > 1` 的相机产品是同一姿态下的独立时域噪声抽样；
- 波长积分网格仍是确定的（响应曲线节点加 `wavelength_samples+1` 个均匀点），随机种子不进入网格构造。

每次采集的 `acquisition_index` 同时记录在样本清单 `cameras.<band>.acquisition_index` 与任务 TOML `[dataset]` 段中，供噪声溯源。

动态场景、模型运动和带动画的 glTF 会被拒绝，避免三次先后执行被误当成同一时刻的动态三相机观测。

### 夜间臂（VIS 失效工况）

四个 `*_night` 变体在相同相机组几何下提供 VIS 失效对照：太阳/天空 RGB 辐亮度置零、热求解到 23 时，**大气段沿用白天条目的预设（气溶胶类型与能见度），只把随时刻变化的量换成 23 时 forcing 行**（地表/气温、湿度、以及普朗克天空回退用的 `atmosphere_temperature_k` 取 forcing 的天空辐射温度而非白天值），太阳方向取 23 时真实的地平线下位置（NN 烘焙由此判定夜间、太阳散射项归零）。红外天空下行辐射不受影响：MWIR/LWIR 的天空项走 NN `ldown`（再回退解析晴空、再回退大气温度普朗克），零星的 RGB `sky_radiance` 只管 VIS 解析天空——这是源码级确认，不是推断。**注意 `atmosphere_temperature_k` 的语义分叉**：NN 大气激活时配置解析会把 `T_air` 覆写为大气段的 `t_ground_K`（发射率归一化需要真实气温），该键只在普朗克回退路径上生效——热求解的宽频下行辐照度与渲染的天空辐射因此来自不同口径，对账见"验证记录"。

smoke 实测（64×36，0.7.4）：夜间 VIS rawdn 均值 2.1、中位 0 对白天中位 118（均值 108，灯具/灯牌高亮尾巴抬升）；夜间 LWIR 有效像素 `tapp` 均值 298.2 K（真值 300.0）。两点口径修正：**(1)** 天空像素的表观温度约 **295 K**——这不是 NN `ldown` 本身，而是着色器把天顶 NN `ldown`（8–12 µm 带内等效亮温约 264 K）经平流层模型 `L_sky(θ)=B_air·[1−(1−ε0)^sec θ]` 向地平线延拓的结果，近地平线天空逼近 `T_air`=300 K（即 `t_ground_K`）的厚层极限。**(2)** 早版"最低表观温度 284.6 K ≈ forcing 天空温度 284.68 K"的记录是 0.7.2 采样缺陷积分后的巧合，不是 `ldown` 生效的证据；`ldown` 为 NN 输出的结论是**由闭合检验证实**的（见"验证记录"：对高发射率漫反射面，τ·[εB(T_s)+(1−ε)L_down]+L_path 与 corrected 信号闭合到约 0.3–0.7 K（镜面路径变体），NETD 量级）。这提供了"融合增益必须来自 IR"的可检验条件。雾/霾条件可通过大气预设以同样的 `overrides` 机制追加；当前正式配置未含。

**夜间臂保留一张按 1e-12 缩放的太阳表**（2026-11-11）：Qt `406deb8` 起，场景里有实测光谱的求解材质不再手填 `shortwave_absorptivity`，渲染器从 `lighting.solar_lut` 与材质反射率推导（混凝土 0.73、白墙 0.47……）；没有可加载的太阳表时退回 0.7 并告警（`none derivable ... assuming 0.7`）。夜间条目的热求解从 0 时算到 23 时、中间经过 forcing 文件里的整个白天，此前覆盖项移除了太阳表，于是夜间臂的白天热历史用的是 0.7，而白天与低照度臂用推导值——同一场景 23 时的温度场在夜间与低照度臂之间差到 0.69 K（smoke concrete，`build/env-smoke-0.10.0-pre13b`）。保留太阳表但不缩放也不行：太阳在地平线下时，渲染器仍按太阳表的天空项照亮场景，夜间 VIS rawdn 中位从 0 变成 23 DN（`build/_qa/night-lut-0.10.0`）。所以夜间覆盖项带上太阳表并设 `solar_scale = 1e-12`：吸收率是比值、不受幅度影响，夜间与低照度臂的表面温度真值逐像素相同（QA 实测最大差 0.0 K），而照度比低照度臂的 2e-6 再低六个量级。

### 低照度臂（`*_moonlight`，SNR 受限而非全零）

夜间臂的二值性（VIS 几乎纯噪声）会让"融合增益来自 IR"变成同义反复。`*_moonlight` 变体给出中间工况：**太阳光谱表按 `solar_scale = 2e-6` 缩放**（约满月 0.2 lux 对直射日 ~1e5 lux），**同步向 job 注入 `atmosphere.solar_scatter_scale = 2e-6`**——NN 路径辐射的太阳散射分量按同一因子缩放，否则未缩放的黄昏级路径辐射会以"雾幕"形式叠加在月光地表上（0.7.3 新增键，见"大气"节）；**太阳位于 5° 仰角**——NN 训练域的下边界（天顶 85°），再低则触发夜间分支把散射项整体归零、贴地直射光也消失，白天仰角则会烘出日光级天幕；**VIS 曝光改为 1 s**（50 µs 白天曝光下物理缩放的月光是光子饥饿而不是 SNR 受限）。MWIR/LWIR 是热发射，与 23 时夜间臂同一条求解轨迹、同样不变。

smoke 实测（64×36，0.7.4）：月光 VIS rawdn **中位 6.0 DN**（白天中位 118、夜间中位 0），与曝光/照度线性预算 170×(1 s/50 µs)×2e-6≈6.8 DN 吻合——月光地表与阴影区进入散粒噪声主导区间，这正是研究融合权重分配想要的工况。均值 53.2 DN 由场景灯具（车头/尾灯，不随太阳缩放）在长曝光下的高亮尾巴拉高，属均值-中位差伪影而非大气信号；`atmosphere off` 对照与 1 s 暗帧分解确认残余项 = 灯具饱和尾 + 暗电流（~2 DN）+ 未缩放前的 NN 散射（已修）。热与材质与夜间臂一致，三波段对照干净。

需要注意：月光臂的太阳表仍是 `planck_tail` 近似再乘 2e-6，NN 散射缩放是"散射 ∝ 源辐照度"的一阶近似——VIS 的量级是对的，但不应被解释成"有月光的精确光谱"。

### 实测扫描场景（`configs/env_dataset_ql_site.toml`）

融合用的真实地形场景：terra2ql（`H:/LidarPointCloud`）把 DJI Terra 5.3.5 的 LiDAR + 摄影测量网格转成 Quantiloom 场景，放在 `H:/LidarPointCloud/out/ql_site`。场地在河北围场附近，约 1 km × 1 km、11,843,651 个三角形，航飞时间 2026-08-27 14:19 +08。配置直接引用 `site.mwir.toml`。

**哪些是实测、哪些是模拟**：只有网格几何与 RGB 照片纹理是实测的，而本数据集只渲染场景的 `spectral` 变体（逐类材质、无纹理），照片纹理不进入渲染。地物类别约 88 % 面积来自自动规则、约 12 % 手工标注；每类光谱取自 ECOSTRESS/SLUM 合并库（玉米、车漆等是替代或占位条目）；每类 `ir_temperature_k` 由 terra2ql 的 `scripts/band_tomls.py` 按稳态能量平衡算出，气象取气候平均（气温 26 °C、RH 55 %、风速 2 m/s、河水 23 °C）而不是实测。红外图像全部由 Quantiloom 渲染；Terra 工程只含 LiDAR 与 RGB 航摄，没有实测红外数据。

与 env 场景的差异：

- `thermal_mode = "authored"`：场景没有 `[thermal]`（千万级三角形做求解元太慢），每类一个温度，阴影与受照面同温，IR 中的阴影只来自被挡住的反射太阳光。
- 大气：源配置 `preset = "disabled"`。300 m 斜距下 IR 路径项不可忽略，配置用 `overrides.atmosphere` 改为数据集 NN 包的 `clear` 预设（乡村气溶、能见度 23 km），气温/地温 299.15 K、`rh = 0.55` 取场地气象。
- 太阳谱：`planck_tail` 把 ASTM G173 延伸到 4 µm 以外，取代源配置在 4–5 µm 的端点钳制。
- VIS 用合并库曲线：源 `site.vis_hero.toml` 的 USGS 曲线只到 2.5 µm，不能跨三个波段共用一份材质表。河水的 VIS 因此是清水曲线（Tap water），比浑浊的真实河水暗。
- 采样：环绕源相机的注视点 `[-508.624, -117.589, -406.428]`，8 个方位 × 俯仰 45° × 半径 300 m。俯仰更低时 VIS 画面上沿会越过网格边界。三款相机配置与正式配置相同，包括 VIS 曝光 5e-5 s。
- glTF 的 74 个 `terra_photo_*` 材质在 IR 任务里得到缺省 `ir_emissivity = 0.9` 条目，但 `spectral` 变体不使用它们。只有一个 glTF，全部样本同属一个 `split_group`。

实测（Quantiloom 0.8.3，`build/env-qlsite-0.8.3-r2`）：8/8 个三元组一次跑完，单次采集中位 VIS 21.6 s、MWIR 14.4 s、LWIR 13.3 s（含场景加载）。`truth_validity` 未命中像素最多 7/230400（VIS）。有效像素 `truth_surface_temperature` 296.1–326.5 K，`tapp` 1–99 分位 MWIR 302.4–320.3 K、LWIR 300.7–322.1 K。300 m 的大气路径把表观温度拉向气温：例如 a000 的 LWIR `tapp − T_s` 河水（296 K）+3.2 K、涂层钢板屋顶（327 K）−4.9 K、玉米（304 K）−0.9 K。VIS rawdn 中位 57–61 DN（与 concrete 白天同量级），无饱和。闭合审计 a003：mean −0.107 K、rms 2.01 K；a007：mean −0.051 K、rms 1.81 K（env 场景 rms 约 6.5 K）。首次尝试 `build/env-qlsite-0.8.3` 规划于 `look_at` 修复之前，MWIR 被位姿校验拒绝（见迁移节），只作证据保留。

## 使用方法

在 `D:/QuantiloomFusion`（Windows）或 `/mnt/d/QuantiloomFusion`（WSL）目录运行。命令在两系统下相同。输出目录以渲染器版本为后缀（如 `build/env-smoke-0.8.3`）；改了相机配置、配方开关或渲染器版本就换一个新目录，旧目录只 `verify`。本机现有的 `build/env-smoke-0.7.4`、`build/env-pilot-0.7.4`、`build/env-subset-0.7.4` 由 0.7.4 渲染（前两个还规划于 `build/` 迁到 H 盘之前），`build/env-smoke-0.8.3` 已在 0.8.3 下完整渲染，都只能 `verify`，重新试跑请用新目录名：

```powershell
python -m pip install numpy OpenEXR
```

先生成计划，查看相机 TOML 和组合数量：

```powershell
python tools/generate_env_dataset.py plan `
  --config configs/env_dataset.toml `
  --out build/env-three-band-0.8.3
```

小样本试跑生成六个场景条目各一组 VIS／MWIR／LWIR（含一个夜间变体和一个低照度变体）：

```powershell
python tools/generate_env_dataset.py run `
  --config configs/env_dataset_smoke.toml `
  --out build/env-smoke-0.8.3-try1
```

正式运行及断点续生使用同一条命令：

```powershell
python tools/generate_env_dataset.py run `
  --config configs/env_dataset.toml `
  --out build/env-three-band-0.8.3
```

限制本次新增采集数、仅选择场景，或失败后继续其他采集：

```powershell
python tools/generate_env_dataset.py run `
  --config configs/env_dataset.toml `
  --out build/env-three-band-0.8.3 `
  --scene concrete --scene seaside `
  --max-captures 6 --keep-going
```

`--max-captures` 按单款相机计数，6 次采集通常对应两个三波段样本；每次尝试（包括失败的）都计一次，已通过验证而跳过的采集不占额度。预算允许停在 VIS 与红外之间，未完成的三波段组合不会进入完整样本索引。`--scene` 仅筛选执行范围，原计划仍保留全部组合。

无需渲染器和源资产的检查：

```powershell
python tools/generate_env_dataset.py verify `
  --out build/env-three-band-0.8.3 --inspect-exr
```

不加 `--inspect-exr` 时，仍检查文件哈希、输出完整性和实际相机参数；加上后进一步检查 EXR 尺寸、通道、有限像素及内嵌事务标识。`verify` 只对目录中已存在的样本做一致性校验：**没有渲染的计划目录也能通过验证**（输出 `complete=0`）。判断数据集是否完成须看输出的 `complete` 是否等于 `planned`，或以 `dataset.index.json` 为准。

辐射自检另有独立工具 `tools/_closure_audit.py <attempt_dir>`（配套：`tools/_mask_audit.py` 按 validity×材质分桶统计 `tapp` 与真值，定位偏差归属材质；`tools/_spectral_integration_audit.py` 以全通道精确积分定标采样网格残差）：离线复算同一 NN 模型包的大气推理（numpy 复刻 AtmosNet/ResMLP 前向），按 `L_pred = τ·[εB(T_s)+(1−ε)L_down]+L_path` 逐像素对照 corrected 信号（经 tapp 反演口径），按材质报告偏差——渲染端与导出端的大气不一致会以系统性偏差暴露（0.7.2 的 LWIR 采样缺陷即由此类检查定位）。低发射率材质用**镜面路径变体**：沿法线-视线反射方向代入着色器同款角向天空模型 `L_down(θ)=B_air·[1−(1−ε0)^sec θ]`，检验"冷镜面反射暖天空"这条 LWIR 物理链的其余部分（掠射角发射率、遮挡修正之外的项）。CLI 不导出波段积分 τ/lpath/ldown，该脚本是闭合检验的替代路径；依赖 `numpy`/`OpenEXR`/`safetensors`。

**2026-11-11 起审计按材质分两类**（渲染器 0.10.0 之后，有光学常数的光滑表面在红外按方向菲涅耳镜面瓣渲染）：漫射材质（绑定实测谱或只有标量发射率）用各向同性式 τ·[εB(T_s)+(1−ε)L_down]+L_path，L_down 取渲染器平板天空律 B(T_air)·[1−(1−ε₀)^secθ] 的余弦加权半球平均（同时给出天顶 NN ldown 的对照列）；真值清单里每个材质只有一个波段等效发射率（`band_emissivity` = 1 − 波段内反射率的不加权平均 − τ），所以发射率是在波段积分之后才乘上的，残差里包含这个带平均近似。镜面材质（真值清单 `emissivity_model = "fresnel_complex_refractive_index"`，0.10.0 对它们导出的 `band_emissivity` 是 null——渲染器的注释原话是"角度相关，一个数会是谎言"；与 capture.toml 的 `[refractive_index]` 交叉核对，本数据集是铬球 Cr 与裸铝 `Car_M_IRON.004`，玻璃在中长波没有 n,k、按 0.84 的漫射面处理）逐像素由法线真值求镜面方向，**只取镜面方向指向天空的像素**（镜面方向的遮挡没有检验），反射率取该入射角下由 n,k 表计算的非偏振菲涅耳 F(θ, λ)，发射率 1 − F，入射辐亮度取镜面方向的平板天空；镜面方向落到地面或物体上的像素不纳入。两点口径修正：NN 大气激活时渲染器把热天空的 T_air 换成 `atmosphere.t_ground_K`（旧审计误用 `lighting.atmosphere_temperature_k`，夜间 284.68 K 对 300 K）；大气预设先按渲染器的预设表取值再被显式键覆盖。不参与热求解的材质（铬球、铝件）没有表面温度真值（导出 0），按真值清单里的 `authored_ir_temperature_k` 取。

实测（`build/env-pilot-0.10.0` 全部 24 个 LWIR 采集，中位数）：铬球镜面方向朝天的像素，各向同性式偏差 −20.9 K（rms 24.0 K，夜间与低照度）、−24.7 K（rms 28.6 K，白天）；方向菲涅耳式 +0.45 K（rms 3.4 K）、−3.4 K（rms 5.4 K）；离开球缘的内部像素（cos θᵢ > 0.5）几乎一样，所以剩余离散主要不是 PSF 在球缘混入背景。偏差降到 1 K 以内，rms 没有：剩余部分来自镜面瓣的有限宽度（铬球粗糙度 0.12，近地平线处平板天空随天顶角变化很陡）、带平均近似，以及镜面方向上可能被房屋等遮挡（未检验）。漫射面夜间/低照度混凝土地面 rms 0.43 K、白墙 0.20 K、车身 0.86 K；白天混凝土 rms 3.3 K（中位偏差 −1.3 K），离散集中在强温度梯度处——预测用逐像素真值温度，而相机 `tapp` 经过 1.1 px PSF 模糊。正式运行中 `dataset.closure_audit_every = 50` 让生成器每完成 50 个三元组自动在最新 LWIR attempt 上跑一次并归档 `closure_audit.txt` 到该样本目录，中途回归不必等到全部渲完才暴露。

## 断点恢复和复现

计划保存配置、`--version` 版本字符串、场景及直接引用资产的内容哈希、生成器指纹（`recipe.generator_sha256`：`quantiloom_fusion.datagen` 在包内的导入闭包——datagen 各模块及其用到的 `core` 模块，换行规范为 LF 后逐文件哈希，清单见 `recipe.generator_files`；消费端模块与 `datagen/audit/` 不计入）、派生太阳光谱哈希和 `physics_sources`（开发构建树下 `src/shaders`、`src/libQuantiloom/atmos`、`src/libQuantiloom/postprocess` 的逐文件哈希，见"版本兼容"节）；CLI 可执行文件与同目录 SPIR-V／DLL／USD 资源记入 `renderer_observed`，仅供溯源。恢复时重新计算并比对：场景资产、配置、生成器闭包内任一模块、CLI 版本号或任一物理源文件变化都会拒绝复用旧目录（提示换用新输出目录）；同版本号下仅重新编译（无源码改动）不触发拒绝。已生成的任务 TOML 若被手动修改，也会报错。

首次规划的那个系统冻结计划：Windows 与 Linux 的 libm 在部分三角函数结果上差最后一位（ULP），正式配置 7776 个任务 TOML 中有 549 个在另一系统上重算后逐字节不同（2026-10-12 实测，`build/_parity/20261012-105513-formal-crossos`）。因此在另一系统上重新 `plan`/`run` 同一目录时，生成器保留已存的 `dataset.plan.json` 与任务 TOML，只要求重算结果在 1e-12 相对/绝对容差内一致、派生太阳光谱逐字节一致、每个任务文件仍与记录的 `job_sha256` 相符；正式配置在 WSL→Windows 与 Windows→WSL 两个方向上都实测可续跑且不改动任何已存文件。

每款相机有独立状态记录。恢复时先核验元数据哈希、全部渲染产品哈希、实际分辨率、镜头参数、内参和位姿；只有验证通过才跳过。一个波段失败或损坏不会要求重新生成另外两个已验证波段。若渲染已结束而发布步骤中断，脚本尝试接纳已经完成的输出。

每次重试使用新 `attempt-NNNN` 目录，保留之前的日志和不完整输出；不会清理或复用 Quantiloom 的导出事务目录。成功输出复制到 `samples` 中的独立版本目录，验证后以原子操作发布元数据和完成状态。Windows 下复制内容而不复制渲染器的仅所有者访问权限，便于后续分发。

同一输出目录只允许同一系统内的一个计划或生成进程写入。操作系统文件锁会随进程退出释放；状态文件还记录活动渲染子进程，避免生成器意外退出后启动两个并行写入者。正常按 Ctrl+C 会停止当前子进程并记录中断状态，之后运行原命令即可继续（WSL 下经 interop 运行时，Windows 侧的 `Quantiloom.exe` 在 Ctrl+C 后约 1 秒内退出，2026-10-12 两次实测，不会残留）。若生成器被强制结束而渲染子进程仍运行，应先让该进程结束，再恢复。

## 输出结构和融合输入

```text
dataset.plan.json                 冻结的实验计划与来源（含 renderer_observed 溯源表）
dataset.index.json                已完成的三波段样本索引
provenance/solar_<scene>.csv       明确标注假设的派生太阳光谱
jobs/<sample>/<band>.toml         可审阅的 CLI 配置
state/<sample>/<band>.json        单款相机的恢复与验证记录
attempts/<sample>/<band>/         原始尝试、实际配置与渲染日志
samples/<sample>/sample.json      三波段组合元数据
samples/<sample>/<band>/vNNNN/    完整 EXR、预览和原生导出元数据
jobs|state|attempts|samples/<sample>/ref_<band>...   共视点无噪声参考（仅参考子集里的样本）
```

### 共视点无噪声参考子集（`[reference_subset]`）

融合质量缺一个真正的参考图像；合成数据能给出实拍给不了的东西。配置里的 `[reference_subset]`（`azimuths_deg`、`elevations_deg`、`radius_scales`，取值必须出现在 `[sampling]` 里；不写这个表则计划与任务文件逐字节不变）选出一部分样本，每个追加三次渲染 `ref_vis`、`ref_mwir`、`ref_lwir`：**几何全部取可见光相机**（位姿、分辨率、名义 FOV、畸变、偏置/旋转，光学模糊统一取可见光的 `psf_sigma_px = 0.45`），探测器类型、光谱响应与光谱模式、像元尺寸、F 数、曝光与读出取各自波段的相机（毫米焦距随像元尺寸变，像素内参与可见光相同；低照度臂的 1 s VIS 曝光同样作用于 `ref_vis`），`sensor.quality.noise_free = true`。场景各段（温度场/热求解、大气、材质、太阳表）与该样本对应波段的正式任务完全相同，只有 `[camera]`、`[renderer]` 的分辨率与种子、`[sensor]` 以及 `[dataset].acquisition_index`（按 `ref_<band>` 取键）不同；同一配方、同一指纹体系。三幅参考与可见光的无噪声图像严格共视点、逐像素对齐，没有视差，也没有传感器噪声。

正式配置取方位 0°/90°/180°/270°、俯仰 12°、距离系数 1.0：12 个条目 × 4 个视角 = 48 个参考样本、144 次渲染（`plan` 打印 `(+144 reference renders)`）；pilot 12 个、smoke 6 个（全部样本）。参考在该样本三个正式波段之后渲染，与相机采集一样计入 `--max-captures`、走同一套 attempt/receipt/发布/恢复（`state|attempts|samples/<id>/ref_<band>`）；样本是否完整仍只看三个正式波段，参考是独立的一组产品。发布：`sample.json` 升为 v2，`references` 在三幅都发布后才非空，结构与 `observations` 相同；`dataset.index.json` 每条带 `references` 布尔值；`verify` 打印 `references=<已发布>/<计划>`。消费端只经契约读取（`Sample.references`、`ThreeBandDataset.references(sample_id)`），不按文件名找。

实测（Quantiloom 0.10.0，`build/env-pilot-0.10.0`，12 个参考样本 × 3 = 36 幅）：用真值世界坐标经可见光相机模型把每幅参考投回自身，最大偏差 0.0010 px（float32 精度），三幅参考的世界坐标真值与可见光正式采集逐像素相同（差 0.0 m）。真实 MWIR/LWIR 相机的 `tapp` 经 oracle 对应重采样到可见光网格后与共视点参考 `tapp` 相减（只取可对应区，每个工况 4 个样本，像素数 MWIR 294728、LWIR 373358）：夜间与低照度 LWIR rms 0.173 K、MWIR 0.103 K，白天 LWIR 0.794 K、MWIR 0.609 K，中位数都在 ±0.01 K 以内，P95 绝对值 0.18–0.71 K——差异来自分辨率（真实相机像素更粗、重采样插值）、模糊（PSF 0.8/1.1 px 对参考的 0.45 px）与噪声（NETD 与残余固定图样），白天对比度高、前两项更大。耗时：参考一组中位 17.5 s，正式配置 144 次渲染约 14 分钟（`render.log` 时间戳）。

融合端从 `dataset.index.json` 遍历完整样本，再读取 `sample.json` 中 `observations.vis`、`observations.mwir`、`observations.lwir`。产品路径相对于数据集根目录，原生导出元数据的产品路径相对于它所在的目录。分发时保留 `dataset.plan.json`、`dataset.index.json`、`state` 和 `samples` 即可使用完整的独立验证命令；融合读取只需索引、样本清单及清单指向的产品与元数据。`jobs`、`attempts` 和原始资产用于生成端复现，不是融合端依赖。

每次采集还在 attempt 目录留下 CLI 生成的 `pipeline_cache.bin`、BRDF LUT 与日志（约 5 MB 并含数秒 LUT 生成时间；1728 次采集估算约多 8.6 GB 与约一小时附加开销）。attempts 属生成端复现材料，可在验证通过后分场景清理。热求解另有磁盘缓存，同一场景全部视角只解一次。

### 产品清单

| 清单产品名 | 文件 | 使用意义 |
| --- | --- | --- |
| `measurement` | `frame_measurement.exr` | 原生物理相机波段测量，未加入后续读出与 ISP 处理 |
| `rawdn` | `frame_rawdn.exr` | 经探测器和读出退化后的原生 RAW DN |
| `corrected` | `frame_corrected.exr` | 校正后的设备信号，按元数据解释单位 |
| `display` | `frame_display.exr` | 面向观看的相机输出；VIS 是 RGB，红外文件内为 R/G/B 三通道复制的灰度（含百分位显示拉伸） |
| `tapp`（仅 MWIR/LWIR） | `frame_tapp.exr` | 逐像素表观温度（K），辐射传输反演；验证热场与做温度图分析的直接输入 |
| `rendered_reference` | `frame.exr` | 另一次理想针孔主渲染，用作参考；没有相机退化 |
| `truth_camera_z` | `frame_truth_camera_z.exr` | 相机系深度（m），像素中心瞬时采样，与相机产品逐像素对齐 |
| `truth_ray_distance` | `frame_truth_ray_distance.exr` | 光线传播距离（m），斜距 |
| `truth_world_position` | `frame_truth_world_position.exr` | 命中点世界坐标 |
| `truth_geometric_normal` | `frame_truth_geometric_normal.exr` | 几何法线（单位向量） |
| `truth_validity` | `frame_truth_validity.exr` | 覆盖/有效编码：0=未命中 1=不透明可对应 2=镜头无效 3=部分覆盖 4=透射表面 |
| `truth_instance_id` | `frame_truth_instance_id.exr` | 逐像素实例 ID（UINT），配 `frame_truth_instances.json` 的 ID→glTF 节点名索引 |
| `truth_instances` | `frame_truth_instances.json` | 实例索引：`{instance_id, node_index, name}` |
| `truth_surface_temperature` | `frame_truth_surface_temperature.exr` | 逐像素求解表面温度（K）；未命中为 −1。与 tapp 不同，这是**物理真值**而非传感器反演 |
| `truth_material_id` | `frame_truth_material_id.exr` | 逐像素材质 ID（UINT），配 `frame_truth_materials.json` |
| `truth_materials` | `frame_truth_materials.json` | 材质清单：`{material_id, name, emissivity_model, band_emissivity, band_nm, thermal_participates, authored_ir_temperature_k, interior_bc, interior_temperature_k}`——其中 `band_emissivity` 是该采集波段的等效发射率（绑定材质为测量谱的波段积分，未绑定为作者常数），`emissivity_model` 区分 `kirchhoff_spectral`/`authored_curve`/`derived_from_metallic` 等来源 |

`tapp`（仅 IR）的语义需要写清：它由**带噪校正信号**逐像素反演黑体温度——等效于假设 ε=1 的亮温，包含反射下行项、路径辐射与探测器噪声；它**不是**表面温度。表面温度真值用 `truth_surface_temperature`；波段等效发射率真值用 `truth_materials` 的 `band_emissivity`。三者合起来可以直接做温度–发射率分离（TES）类评价（Gillespie 等 1998 的 ASTER TES 是这类方法的基准），也可以把图像对比度分解为温度驱动与发射率驱动两部分。

真值产品来自 `OfflineRenderer::CaptureGeometry`：传感器自身投影的一次确定性光线投射，与 SPP 无关，因此与相机产品严格同网格同畸变。配准评价的推荐做法：

- 双向 z 测试：用两台相机各自的 `truth_camera_z` 做重投影—回测双向遮挡掩码，而不是只信单向重投影；
- 像素中心采样与相机产品的 PSF+像素积分在深度不连续处定义不同——边界应排除一圈边缘带，`validity=3`（部分覆盖）与 `validity=4`（透射）像素同样排除；
- 天空（0）与玻璃（4）按 validity 掩码排除。

本数据集相机基线 **3.5 cm（VIS–MWIR）、4.5 cm（VIS–LWIR）、约 8.0 cm（MWIR–LWIR）**——10 m 深度下 LWIR↔VIS 视差约 2 px（fy≈435），5 m 约 4 px，近处不能用无穷远单应近似。实例 ID 与材质 ID 同时提供语义/分割级评价锚点。

`measurement`、`rawdn`、`corrected` 的量纲不一定相同。VIS 的 CFA 单通道也不能直接等同于灰度亮度；可使用 `display` 的 RGB 结果，或在融合端明确实现 RAW 解码与去马赛克。应读取每个产品的 `signal.kind`、`signal.unit`、传递函数和 CFA 信息（`camera_cfa` 记录枚举字符串，`"0"` 为单色、`"1"` 为 RGGB），随后分别处理动态范围。红外显示产品包含场景相关的显示拉伸，即使自动曝光已关闭，也不应把它当成线性测量。

VIS 主渲染显式使用 `vis_hero` 采样波长；物理相机产品通过各通道响应曲线和 `sensor.quality.wavelength_samples` 积分采样。两条渲染路径的采样、投影和产品语义不同，因此 **`frame.exr` 不能当成与原生畸变相机逐像素对应的干净真值**——干净的对应真值用 `truth_*` 产品。三波段融合应一致选择相机产品，而不要混用 VIS 主渲染与 IR RAW。

每个相机产品保留导出内参、畸变、实际 camera-to-world/world-to-camera、像素中心、坐标约定及完整生效相机配置。三波段尺寸和视场均保持原样，脚本不重采样、配准、裁剪到公共视场或强制转换为共同灰度。最终融合为灰度还是保留 VIS 颜色，由后续算法决定。

像素 QA 记录最小值、最大值、标准差、平坦标记、RAW 达到 ADC 上限比例，以及光子相机达到满阱或 ADC 上限的比例。这些是检查信息；非平坦和无饱和不能证明视角有效或物体可辨识。其中满阱比例按整像素阈值判定，叠加负读出噪声后饱和像素可能略低于阈值，统计值会稍微偏低。正式生成前仍应查看代表性视角，尤其是极端距离、背面遮挡和相机进入建筑的组合。

## 实验边界与限制

- **传感器是通用预设**，不是三款实测相机的标定。要主张真实相机等价，需要 EMVA 1288 类实测响应、噪声参数与镜头标定替换通用模型。
- **材质语义已修订但仍为代理**：混凝土/沥青为全波段连续测量的 JHU 条目，车身为白色涂层实测谱（代替车漆，LWIR ε≈0.92，接近真实漆面的上限），任务波段内没有跨空洞插值的材质；部分绑定条目在 LWIR 库段覆盖 0.71–0.88（任务段 8–12 µm 满覆盖，见材料表）；VIS 中少数条目覆盖 0.84–0.93；材质 NMF RMSE 在 300 K 下约对应 0.4 K 的系统性表观偏差（材料节有换算）。低发射率的强镜面目标现在只有铬球与裸铝部件，面积都很小。
- **热求解假设**：单层热壳 + forcing 驱动的表面能量平衡；forcing 文件是合成气象日，不是实测。人体是给定的 307 K 皮肤温度、不参与求解（见"物理建模"），其他内部热源/相变仍无。
- **大气是 NN 回归**（MODTRAN 系数据训练），非在线逐线辐射传输；太阳直射/天空漫射走 solar LUT 不被 NN 二次衰减，但 LUT 的直射/漫射配比与 NN 大气状态不严格自洽。NN 训练域的太阳天顶角上限是 85°：非热波段 `lpath` 超出即归零（夜间真黑），黄昏过渡档（5–0° 仰角）不可用网络外推——月光臂固定用仰角 5° 的边界工况就是这个原因。
- **海面粗糙度未建模**：SeaWater 是纯水 n,k + Fresnel 平面界面，没有 Cox–Munk 坡度分布。Masuda（1988）的方向发射率模型假设有风海面的坡度统计——本数据集的镜面海在掠射角下发射率下降会比真实粗糙海面更剧烈，远处海面的 LWIR 表观温度因此系统性偏低天空反射方向。VIS 端同理：纯水 n,k 无体散射，海面是一面"黑镜"（只反射天空/岸线），不是真实海色的绿蓝体散射色——融合算法如果学到"海面=暗镜面"的先验，在真实数据上会失败，做 VIS 融合结论时须注意。
- **固定图样与 NUC 是假设值**：MWIR 与 LWIR 都按"两点校正后残余 3 %"建模，行列条纹按"列为主"分配（见"固定图样、行列条纹与 NUC"），这些比例不是实测器件的标定；真实微测辐射热计的残余还随环境温度漂移（快门间隔内），本数据集没有建模这种时间变化。
- **太阳谱 >4 µm 是普朗克尾近似**；QLTrans/MODTRAN 的分子带 DATA 文件本检出缺失，自洽太阳表是明确的后续工作（见"solar_ir_policy"）。日间 MWIR 受照区反射太阳项约占热发射 10% 量级，量级与此近似误差相当。
- **月光臂**是太阳谱 ×2e-6 加 NN 散射线性缩放的近似（`solar_scatter_scale`，见"大气"节），不是"有月光的精确光谱"；场景灯具在长曝光下局部饱和（真实夜间相机行为）。**1 s VIS 曝光只对固定监控/三脚架场景合理**：车载、手持或无人机平台的实际曝光在数十毫秒量级，那些部署形态下月光 VIS 的 SNR 会显著更差、更接近夜间臂的失效工况。
- **自发光物体**（车灯等）在 IR 波段没有绑定发射谱时不贡献辐射——渲染器会列出这些材质；本场景车辆处于熄火状态。车身的 ~0.05 灰度 `emissiveFactor` 是 glTF 导出残留（漆面不是灯），已在场景级置零——在长曝光下否则会向像素注入编造的光子。
- **repeats 语义**：`per_acquisition` 下重复是同一姿态的独立时域噪声抽样（FPN 不变），不是新几何观测。
- **四个真实场景 + 变体**的规模不足以证明现实场景泛化；训练／验证／测试划分应按 `split_group`（场景模型哈希）分组，避免同一资产的相邻视角泄漏。
- **"三波段优于 VIS+LWIR"** 需要在本数据集上做波段消融后才能主张；当前数据提供条件，不提供结论。

## 评价协议建议

数据集提供条件，以下协议是把"互补性"变成数字的建议路径。

**互补信息量化。** "互补"的定义应是可测的：训练一个 VIS → `truth_surface_temperature` 的回归器（在按 `split_group` 分组的训练折上），其残差方差是 IR 独有信息的**上界**——回归器越强残差越小，它给出的是"VIS 不可解释部分"的乐观估计，不是下界。白天约 20 K 的表面温度分布中有相当部分（阴影、反照率）可由 VIS 预测，不可预测的部分（热惯性滞后、发射率差异）才是 IR 真正独有的。三点口径须遵守：回归目标用 `truth_surface_temperature` 而非 `tapp`（后者含反射下行项与探测器噪声；若坚持以传感器量为目标，应先从残差中扣除 NETD² 噪声底，否则残差地板由探测器而非信息决定）；条件互信息的写法是 I(Y; LWIR | VIS)，其中 Y 是任务目标（T_s 或检测/分割标签），不能写 I(tapp; LWIR | VIS)——`tapp` 本身就是 LWIR 的导出量，该写法恒非零、不提供信息。`truth_surface_temperature` 与 `truth_materials.band_emissivity` 允许把残差进一步分解为温度驱动与发射率驱动两项。另外只有 4 个场景组，留一折验证时回归器泛化差会进一步放大这个上界——报告的残差数字必须注明折数和场景组构成，跨场景外推结论不能从同场景划分的数字里读出。

**消融矩阵。** 建议按四轴组织，每轴一个对照：

| 轴 | 对照条件 | 需要的额外渲染 |
| --- | --- | --- |
| 波段 | VIS-only / VIS+MWIR / VIS+LWIR / 三波段 | 无（同一份数据的不同输入组合） |
| 材料光谱 | `ir_materials="spectral"` vs `"constant"` | IR 重渲；VIS 任务 TOML 与光谱模式逐字节一致（`[thermal]` 与 `ir_*` 字段已从 VIS job 移除），可直接复用 |
| 热模式 | `thermal_mode="solved"` vs `"authored"` | IR 重渲（authored 等温对照）；VIS 同理可复用 |
| 工况 | 白天 / 夜间 / 月光 | 已在数据集中 |

材料×热模式的 2×2 不需要全网格：在一个俯仰角、8 个方位角的子集上足以估计主效应。波段消融完全不需要重新渲染——从既有采集里选波段组合即可。

**配准评价**。双向 z 测试生成跨波段遮挡掩码（用两台相机各自的 `truth_camera_z`）；深度不连续处排除一圈边缘带，`validity=3/4` 像素同样排除；基线按 3.5/4.5/8.0 cm 计。指标建议直接报告配准误差分布（像素/米）对下游融合指标的敏感度，而不是只报融合指标。

## 相关工作

- **多波段数据集**：TRICLOBS（Toet 等，PLoS ONE 2016，16 段已配准 VIS/NIR/LWIR 序列，第三波段是 NIR 而非 MWIR）；TNO Image Fusion Dataset（微光可见、NIR、LWIR 夜间数据）；常用 RGB–LWIR 基准 M3FD、MSRS、RoadScene、LLVIP；SWIR-LightFusion（2025）指出公开基准只有 RGB–LWIR，三波段含 MWIR 的公开数据稀缺——这是本工作的直接动机之一。
- **光谱数据源**：ECOSTRESS/JHU 光谱库（Beck 等；混凝土、沥青等连续测量条目，本数据集主体绑定来源）；ASTER 光谱库（Baldridge 等，2009）；USGS splib07a；Hale & Querry（1973，水的光学常数，SeaWater 的 n,k 来源）；Rakic（1998，Al 与 Cr 光学常数，裸铝部件与铬球的 n,k 来源）。SLUM（Kotthaus 等，ISPRS J. 2014）仍在合并库中但**本数据集不再绑定**——其 2.5–8 µm 实测空洞在 MWIR 消融下不可接受（见材料节）。
- **水面方向发射率**：Masuda 等（1988，海面方向发射率模型，约 50° 观测角内与实测吻合）；Niclòs 等（SPIE 2004；RSE 2005，角向发射率实测与模型比较）。
- **温度–发射率分离**：Gillespie 等（1998，IEEE TGRS，ASTER TES——5 通道对 6 未知数需经验约束，是本数据集 `truth_surface_temperature`/`band_emissivity` 真值可支撑的评价基准）。
- **MWIR/LWIR 融合证据**：Chari 等（SPIE 2005）用多指标量化了 MWIR+LWIR 融合收益。
- **未配准融合方法**（可作 baseline）：RFNet（CVPR 2022）、MURF（TPAMI 2023）、UMF-CMGR、AUNet、FusionRegister、Diff-RF（指出退化会严重影响配准质量，支撑本数据集"原生退化 + 未配准"的设计动机）。
- **评价**：VIFB 指出无单一无参考指标占优；趋势是任务驱动评价（检测/分割精度），本数据集的实例 ID 与表观温度正为此准备。
- **物理仿真同行**：DIRSIG（RIT，第一性原理辐射传输）；MuSES/CoTherm（部件级传热 + EO/IR 预测）。本方案相对它们的差异：Quantiloom 是光谱路径追踪 + 传感器退化链 + 数据集事务封装的开源实现；审稿意义上的交叉验证（例如平板材料表观温度对比）尚未做。
- **物理数据与标准**：ASTM G173-03（太阳谱）、libRadtran（Emde 等，GMD 2016）、EMVA 1288（传感器特征化）。

## 验证记录

代码测试位于 `tests/test_generate_env_dataset.py`，共 17 项，两个系统下都能跑；其中 2 项需要真实数据集，不设环境变量时自动跳过：

```powershell
python -m unittest discover -s tests -v
```

设置 `QUANTILOOM_TEST_DATASET` 后，会进一步使用真实导出测试发布恢复、产品损坏检测、位姿和 FOV 拒绝路径。两点要求：该数据集必须由**运行测试的那个 Windows 账户**渲染（渲染器给 `attempts/` 文件设置仅创建者访问的 ACL，别人渲染的目录会在恢复测试里报 PermissionError）；从 WSL 调 Windows Python 时要经 `WSLENV` 传变量，否则变量到不了 Windows 侧、测试被静默跳过：

```powershell
$env:QUANTILOOM_TEST_DATASET = "D:/QuantiloomFusion/build/env-smoke-0.8.3"
python -m unittest discover -s tests -v
```

```bash
WSLENV=QUANTILOOM_TEST_DATASET/p \
QUANTILOOM_TEST_DATASET=/mnt/d/QuantiloomFusion/build/env-smoke-0.8.3 \
<windows-python> -m unittest discover -s tests -v
```

WSL 的 Linux Python 下同一环境变量直接生效（值为 `/mnt/d/...` 形式）。

本版本验证记录：Quantiloom 0.7.4 编译与单元测试通过（1894/1894，8 skip 为基线），`./build_wsl.sh` 全量门（测试、ABI、furnace 空腔、照明套件）完成后 SDK 安装至 `H:/Quantiloom-SDK/windows_amd64`；smoke 配置 6 组三波段共 18 次采集在 WSL 下全部渲染成功（solved 热求解、夜间、月光臂），`verify` complete=6/6。关键数值（64×36 smoke 分辨率，0.7.4 + JHU/Al 材质 + 胞盒平均）：

- 表面温度真值：concrete 白天有效像素均值 318.5 K，23 时夜间/月光 300.0 K——同一 23 时热轨迹，月光与夜间逐像素一致；
- 表观温度（带噪反演，ε=1 亮温语义，不透明有效像素均值）：夜间 LWIR `tapp` 均值 298.2 K（真值 300.0）。逐材质：JHU 混凝土（ε=0.911）夜间 −1.7 K、白天 −3.0 K；Al 车身（ε≈0.05）夜间 −5.9 K、白天 −6.0 K（冷镜反射暖环境，见闭合检验）；铬球（ε≈0.10）294–306 K 跟随反射天空；天空像素 ~295 K（NN 天顶 ldown 带内等效 ~264 K 经平流层模型向地平线延拓、逼近 `t_ground_K`=300 K 的厚层极限）；
- **大气积分审计的数值证据**：0.7.2 缺陷的定量定位见"大气"节（12000 nm 端点样本整帧归零 → 5 点梯形丢 12.5% 权重 ≈ 10% 辐亮度缺口）；0.7.3 最近通道回退后离线审计（`tools/_spectral_integration_audit.py`：NN 前向全通道精确积分 vs 采样网格积分）测得 τ ±2%/lpath −23%/ldown −1.7 K 的不收敛混叠；0.7.4 胞盒平均在 N=13 下把三分量带均偏差压到 <0.15%（LWIR ldown +0.019%），N=5 smoke 网格 MWIR 仍有 −2.7%/+6.8% 残差——`wavelength_samples=12` 由此定标；
- **辐射闭合检验**（`tools/_closure_audit.py`，离线复算 NN 大气推理）：夜间 LWIR `L_pred = τ·[εB(T_s)+(1−ε)L_down]+L_path` 对 corrected 信号逐像素比较——各向同性变体：混凝土 bias −1.0 K；**镜面路径变体**（沿反射方向的角向天空）把漫反射面收敛到混凝土 bias +0.4 K / 墙面 −0.25 K 的 NETD 量级；Al 车身从 −33 K 收敛到 −16 K——残余来自掠射角方向发射率与遮挡项（角向模型只覆盖天空部分），镜面路径的主项已闭合。CLI 不导出波段积分 τ/lpath/ldown，该脚本用同一 NN 模型包离线复算，是闭合检验的替代路径；正式运行中 `closure_audit_every` 让审计周期性自动执行；
- **天空对账**：渲染器 NN 天空的半球积分下行辐照度在 8–12 µm 带内约 81.6 W/m²，带内等效亮温约 278 K；与热求解 forcing 的宽频天空辐射温度 284.68 K（宽频 ~372 W/m²，8–12 µm 带内等效约 93 W/m²）相差约 12% / 带内等效约 7 K——带内大气窗口比宽频有效温度冷属正常物理（臭氧带、水汽线在窗口内不填），但两者确实来自不同口径：热求解的能量平衡用宽频 forcing，渲染反射项用 NN 带内天空，已声明为已知近似（`atmosphere_temperature_k` 在 NN 激活时被 `t_ground_K` 覆写，两个键不互为替代）；
- 低照度臂：月光 VIS rawdn 中位 6.0 DN（线性预算 ≈6.8）、均值 53.2（灯具饱和尾主导）——早期"均值 63 vs 预期 7"的 9 倍差异分解为：未缩放的 NN 黄昏散射（已由 `solar_scatter_scale` 修复，中位 +2.5 DN）+ 灯具尾 + 暗电流 ~2 DN；夜间 VIS 均值 2.1、中位 0；
- `dataset.acquisition_index` 逐采集不同并计入 manifest；`per_acquisition` 下同波段 `noise_seed` 恒定（FPN 图样跨帧一致、时域噪声独立，源码级保证由测试锁定）；
- 新真值产品 `truth_surface_temperature`/`truth_material_id`/`truth_materials.json` 逐采集产出，材质清单含波段等效发射率与发射率模型来源。

早期 QA 结果另存于 `build/env-dataset-generation-evidence-2026-10-06.json`（`build/` 不入库，仅本机可见），记录的是 Quantiloom 0.7.0 时代的计划与渲染，属历史证据，不代表当前二进制；修复前 0.7.3 构建下的 smoke 目录以及更早的 `env-three-band-ready`、`env-smoke-ready` 等旧目录已封存于 `build/_archive-0.7.3-buggy/`，其配方与现版本不兼容，不能直接续跑。

### 0.7.4 → 0.8.3 迁移对照（2026-10-26）

同一份任务 TOML、同一组种子分别由两个版本渲染（用迁移前的生成器规划，生成器改动不混入）：smoke `build/env-smoke-0.7.4` 对 `build/env-smoke-0.8.3-asis`（6 个三元组），pilot `build/env-pilot-0.7.4` 对 `build/env-pilot-0.8.3-asis`（concrete a000–a002 三个三元组）。耗时取各采集 `render.log` 首末时间戳（只计渲染器）；像素差按 `truth_validity` 分地物/天空，按 `truth_materials.json` 的 `material_id` 分材质。

| 单次采集耗时 | 0.7.4 | 0.8.3 |
| --- | --- | --- |
| pilot VIS（640×360，SPP 32，12 个波长样本） | 25.1–25.5 s | 24.7–25.2 s |
| pilot MWIR（320×256） | 132–133 s | 9.7–9.9 s |
| pilot LWIR（256×192） | 155 s | 8.6–8.9 s |
| pilot 三元组合计 | 约 5.2 分钟 | 约 43 s |
| smoke 中位 VIS / MWIR / LWIR（热缓存未命中） | 11.6 / 17.1 / 11.5 s | 11.3 / 20.1 / 8.2 s |

- **速度**：全分辨率 IR 采集快 13–18 倍，VIS 不变。0.7.4 的 IR 日志在主渲染 EXR 写出后有一段约 144 s 无输出的相机仿真阶段，0.8.3 中消失；归属哪个提交待核实（候选 `0b58383` 只动太阳方位角时不重烘大气 LUT、`8df3da8` 换波长时就地刷新材质缓冲、`0d11799` 查找表就地追加、`98aca37` 相机大气缓冲只分配一次）。按 pilot 外推，当时 2592 个三元组的正式配置约 31 GPU 小时（此前估算 264 小时；同样假设各场景耗时与 concrete 相近，待核实）。
- **几何真值不变**：smoke 18 次采集的 `truth_validity` 逐像素相同。`truth_surface_temperature`（热求解）smoke 中位变化 ≤0.2 K；pilot 混凝土 ≤0.03 K，车/房/人材质 −0.25…−0.56 K（99 分位最大 5.7 K）——长波半隐式、灰体辐射度交换、互易视角因子三处求解器修正的合力。
- **近场大气**（`21f5849`）：50 m 以内的视线不再按 NN LUT 的 50 m 下限计算透射与路径辐射。本数据集相机距目标 5–40 m，0.7.4 多算了大气遮蔽：MWIR 白天 `tapp − T_s` 混凝土地面 −2.87 → −0.67 K、混凝土块 −2.91 → −0.01 K、墙 −0.46 → 0.00 K（pilot concrete_a001）；LWIR 地物 `tapp` 中位 +0.06…+0.33 K。
- **海水**：SeaWater（纯水 n,k，LWIR ε≈0.99）的 LWIR `tapp − T_s` −7.56 → −0.67 K，MWIR −4.98 → −0.59 K（smoke seaside）。0.7.4 的海面表观温度系统性偏低约 7 K；归因于 `03da7eb` 的 IR 透射修复是推断，待核实。
- **天空**：VIS 天空像素 rawdn 下降 45–92 DN（concrete/desert/grassland 与 pilot），seaside +16 DN，地物只变 +1…+5 DN。原因是 `lighting.sky_distribution` 缺省改为 `clear`（`867f488`），以及主光线未命中改为经 NN 大气合成（`35b716f`、`e660d84`）。没有配置键能复现 0.7.4 的天空像素；`sky_distribution = "uniform"` 只恢复穹顶分布。本数据集接受 `clear`。
- **低发射率镜面**（Al 车身，ε≈0.05）反射的环境变冷：LWIR `tapp − T_s` 白天 −5.7 → −7.2 K，夜间 −5.7 → −8.8 K（推断来自下视未命中改取地平线天空 `0fb3e03` 与天空合成变化，待核实）。
- **LWIR 噪声**：`generic_uncooled_thermal_ir` 预设的 NETD 参考带宽 30 → 15 Hz（`d264a71`；生效相机配置 `netd_noise_bandwidth_hz` 实测 30 → 15）。读出带宽是 1/(2×1/30 s) = 15 Hz，0.7.4 实际交付约 35 mK 而不是配置声明的 50 mK；0.8.3 交付 50 mK，LWIR 时域噪声 σ 约 ×1.41。本数据集接受新值。
- **闭合审计**：grassland 残差 mean +0.089 → +0.027 K，rms 6.80 → 6.49 K。
- **契约不变**：导出记录 schema v1、EXR 通道名、真值产品、`replay.toml`、`--version` 格式与 CLI 参数均不变；产品溯源多了 `sky_distribution` 与 `sky_aerosol_optical_depth_500nm` 两个键。两个版本的 smoke 都通过 `verify --inspect-exr`（complete=6/6），消费端读取器在 0.8.3 smoke 上的真实数据集测试通过。
- **热缓存**：键 schema 4 → 7 且含版本号，0.7.4 的缓存条目不会再命中，可以删除。
- **生成器随之改动的两处**：只写一个湿度键（见"大气"节；修改前后的 0.8.3 smoke 有 84 个产品逐像素一致）；`look_at` 放在相机组到目标的距离上。渲染器用 float32 从 `look_at − position` 重建视线基，前视点只在 1 m 外时，离原点数百米的场景（ql_site）会丢失约 2e-5 的方向精度，被位姿校验拒绝（`build/env-qlsite-0.8.3` 即此失败）；env 场景的视线方向变化 ≤1.5e-8 rad。
- **0.8.x 可用而未用的新键**：`camera.view_up`（正下视时指定画面朝向）、`lighting.sky_aerosol_optical_depth`、`thermal.sky_temperature_k`。
- 0.8.3 smoke（`build/env-smoke-0.8.3`，生成器 `f2b5a22`）：约 3 分钟（热缓存已热），`verify --inspect-exr` complete=6/6；月光 VIS rawdn 中位 5.0 DN（0.7.4：6.0），夜间 0。

## 附：Quantiloom 能力清单

面向无源码访问的读者。状态标记：**使用中**=本数据集用到；**可用**=CLI/配置可启用而本数据集未用；**条件**=需特定输入/实验特性；**无**=该路径下不产出。版本基准：Quantiloom 0.10.0。

### 光谱与渲染

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| 光谱模式 | `rgb` / `single` / `vis_hero`（400–780 nm 采样 4λ/路径）/ `vis_fused`（32λ 确定性参考）/ `nir_fused`（930–1200）/ `swir_fused`（1400–2400）/ `mwir_fused`（3000–5000）/ `lwir_fused`（8000–12000）/ `multispectral`（高光谱立方体） | VIS=`vis_hero`，MWIR=`mwir_fused`，LWIR=`lwir_fused` |
| 波长积分 | 物理相机按响应曲线节点 + `wavelength_samples`+1 均匀点确定网格（非随机） | 使用中，12 |
| 路径追踪 | Vulkan RT 光谱路径追踪；逐波长 Monte Carlo；SPP 可调 | 使用中，SPP 32 |
| 荧光 | rank-1 激发/发射/量子产额（`fluorescence_*` 或 glTF 扩展） | 可用未用 |
| 色散 | n(λ) 穿越色散界面；hero 模式按采样λ跟踪 | 可用未用 |
| 环境贴图 | RGB 预览用；光谱测量场景不用（不会编造天空光谱） | 显式关闭 |
| 自发光谱 | `emissive_curve` 绑定内置曲线或测量表；未绑定材质的 RGB 自发光只活在 380–780 nm | 使用中（车灯 VIS 发光；IR 不贡献） |
| 高光谱立方体 | `multispectral` 模式 + ENVI/spectral-EXR/TIFF 导出 | 可用未用 |

### 材质

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| 光谱库 | USGS splib07a / ECOSTRESS 系 / merged（3524 条目，16 维 NMF 基）/ RefractiveIndex.info n,k | merged（spectral 模式） |
| 材质绑定 | `spectral_material_ref(s)` 单条或多端元 + `spectral_unmix`（auto/texture/off） | 使用中 |
| 能量守恒检查 | strict 相机下 ρ+ε(+τ)≤1、波段覆盖检查为致命错误 | 使用中（约束了绑定/发射率的组合方式） |
| 常数 IR 参数 | `ir_emissivity`、`ir_transmittance`、作者温度 | 未绑定材质的回退 |
| n,k 光学 | `[refractive_index]` 材料→YAML；源数据须覆盖任务波段（可 `allow_partial_nk` 预览降级） | 覆盖者保留（Al、Cr），不覆盖者剥离 |
| 温度纹理 | `temperature_texture` 逐纹素温度场 | 可用未用 |

### 热

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| authored 温度 | 场景/材质标量温度，直接渲染 | 消融开关 |
| solved 热场 | 表面能量平衡求解：逐材质热导率/密度/比热/厚度/对流/短波吸收/湿度因子，8 层壳，时间步进 + 视角因子辐射交换 | 使用中 |
| forcing | CSV 气象日（直射/散射/气温/风/湿度）驱动 | 使用中（4 份场景文件） |
| 内边界 | `interior_bc = "fixed"` + `interior_temperature_k`：固定温度内核（代谢/热源近似） | 使用中（墙、屋顶、门的室内侧，海水 291 K 深层水）；人物不求解，给定 307 K |
| 热缓存 | 求解按输入哈希落盘复用（`QUANTILOOM_THERMAL_CACHE=0` 关闭）；0.8.3 的键含版本号（schema 7），跨版本不复用 | 使用中 |
| 几何历元 | 运动场景按分段静态区间分别求解 | 本数据集禁用运动场景 |
| timeline | `sequence` 命令在单一渲染器上推进时钟 | 可用未用 |

### 大气与光照

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| 大气预设 | clear / turbulent_clear / urban_haze / fog / light_rain / heavy_rain / snow / haze / disabled | 场景自带（clear、turbulent_clear；白天只有晴空），数据集级统一到同一 NN 包 |
| NN 大气包 | 按 波段×几何×量（tau/lpath/ldown）组织的 safetensors 网络；CPU 烘焙以太阳天顶角为输入，训练域天顶 ≤85°：>85° 非热波段 lpath 归零（0.7.3） | 使用中（全部四场景，含夜间/月光变体） |
| 单波长烘焙 | 相机网格单 λ 大气量取该波长 Voronoi 胞（相邻采样中点为界）内全部 NN 通道的盒平均，胞宽计入烘焙缓存哈希（0.7.4）；非网格单 λ 取 ±2 nm 窗盒平均、无通道时取波数最近通道（0.7.3 修复，此前静默归零——LWIR 通道间距 6.4–14.4 nm 会大面积落空） | 使用中（相机波长积分路径） |
| 太阳散射缩放 | `atmosphere.solar_scatter_scale`：MWIR 只缩放显式 SOL_SCAT 分量，VIS/NIR/SWIR 缩放整项合并 lpath（无独立散射输出、热发射可忽略），LWIR 热路径发射不缩放（0.7.3 新增） | 使用中（月光臂 2e-6） |
| 天空下行辐射 | `ldown` 网络/解析模型给出 LWIR 反射项；天空像素经平流层模型 `B_air·[1−(1−ε0)^sec θ]` 方向延拓；夜间不受 `[lighting]` 置零影响 | 使用中 |
| 太阳谱 LUT | `solar_lut` 绑定测量表（ASTM G173 等），含直射/全球列；直射与漫射不被 NN 二次衰减 | 使用中 + planck_tail 派生 + `solar_scale` 缩放（月光臂） |
| QLTrans | MODTRAN 封装工具：太阳/天空/透射表、波长范围与步长、大气模型与天气、太阳天顶角扫描 | **条件**：内嵌 MODTRAN4 但缺分子带 DATA 文件，当前检出不可算 |
| sky_model | isotropic / clear_sky 等解析天空 | NN 激活时被忽略 |
| 天空分布 | `lighting.sky_distribution`：`clear` 按晴空大气分布漫射天空（半球辐照度不变），`uniform` 是 0.8.2 之前的均匀穹顶；只作用于 VIS/单波长天空（0.8.2 新增） | 使用中（缺省 `clear`） |
| 天底视角 | `camera.view_up`：`look_at` 与 `up` 共线（正下视）时指定画面朝向（0.8.2 新增） | 可用未用（俯仰 <80°，不会共线） |

### 传感器与相机

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| 探测器模型 | 光子探测器（QE、光子散粒、PRNU、DSNU、读出噪声、满阱、ADC）；热探测器（NETD、响应度、时间常数、预热、增益/偏置固定图样）；两类都有逐像素、列、行三个固定图样分量（0.10.0 起，默认 0）与理想两点 NUC（`apply_nuc` 留下 `nuc_residual_fraction` 倍的图样；0.10.0 修正了光子型的语义），也可经 `nuc_gain_map`/`nuc_offset_*_map` 外部标定图注入 | 使用中（VIS/MWIR 光子，LWIR 热；固定图样与 NUC 逐相机显式配置，未用标定图） |
| CFA | mono / RGGB Bayer + 去马赛克 | 使用中 |
| 镜头 | FOV/焦距内参导出；Brown–Conrady 畸变；pincushion/fisheye 等变体 | 使用中 |
| PSF | 逐相机 σ | 使用中 |
| 噪声种子 | `noise_seed` 驱动 counter-based 流（pixel/acquisition/noiseClass/counter 分离）；FPN 类（PRNU/DSNU）键中不含采集序号 | `per_acquisition`：每波段固定种子 + 逐采集序号 |
| 采集序号 | `dataset.acquisition_index` 由任务 TOML 传入 `camera::CaptureState`（0.7.3 新增）；缺省从 0 开始，预热计入同一序列 | 使用中（逐采集独立指纹） |
| 曝光/ISP | 自动曝光/白平衡（热探测器忽略 AE）、CLAHE、显示拉伸 | 使用中（AE 关闭） |

### 数据产品与导出

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| 相机产品 | `band_measurement`、`raw_dn`、`corrected_device_signal`、`display`、`apparent_temperature`、`traced_radiance`、`cie_linear_srgb` | 前 5 项使用（tapp 仅 IR） |
| 几何真值 | `dataset.geometry_truth`：camera_z、ray_distance、world_position、geometric_normal、validity(0–4)、instance_id + instances.json | 使用中 |
| 温度/材质真值 | `truth_surface_temperature`（逐像素求解表面温度 K）、`truth_material_id` + `truth_materials.json`（材质 ID、发射率模型、波段等效发射率、热参与标志）（0.7.3 新增） | 使用中 |
| 主渲染参考 | `frame.exr` 理想针孔 + `frame.png` 预览 + `frame.replay.toml` 回放配置 | 使用中 |
| 元数据 | `frame.metadata.json`：产品清单、SHA-256、完整生效相机配置、信号种类/单位、事务标识 | 使用中 |
| 导出事务 | ExportSession 暂存→提交、逐产品哈希、`dataset-verify` 校验 | 使用中 |
| fusion-export | 多相机联合导出：共享场景、rig 合并、光路对应真值、contribution、整流对应 | 可用未用（要求共享场景配置，与本数据集按波段定制流程不相容） |
| 回放 | `frame.replay.toml` 可重放采集 | 使用中（留存） |
| 私有 ACL | attempt 输出仅创建者可读（Windows ACL） | 使用中（发布时剥离） |

### 运行形态

| 能力 | 说明 | 本数据集 |
| --- | --- | --- |
| CLI | 单配置 / `batch` 列表（行内覆盖）/ `sequence` 时间序列 / `fusion-export` / `dataset-verify` / `serve`（MCP） | 单配置逐次调用 |
| GUI | Quantiloom-Qt：交互渲染、热面板、序列/高光谱/融合导出对话框 | 未用（仅取场景资产） |
| 平台 | Windows 原生（MSVC），SDK 另有 linux_amd64；WSL2 经 interop 驱动 | 双系统交替使用 |
| GPU | 需 Vulkan RT（NVIDIA RTX）；无 GPU 时渲染类 gate 跳过 | 必需 |
| USD | 可选 OpenUSD 场景加载 | 未用（glTF 场景） |
