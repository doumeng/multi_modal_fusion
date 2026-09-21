# 倾斜下视距离直方图模拟器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task if inline execution is selected; use superpowers:subagent-driven-development only if the user chooses delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可复现的倾斜下视几何、200 次回波统计模拟及检查图，为后续特征融合提供可审计的代理输入。

**Architecture:** 几何、分布构造、采样与导出各自独立。纯 NumPy 核心接受统一车辆框，命令行和绘图层只负责配置与展示；真实雷达数据适配器在完成样本审计后另行实施，首阶段不猜测其二进制格式。

**Tech Stack:** Python >=3.11，NumPy，Matplotlib，Pillow；pytest 用于数值及数据契约测试。使用项目 .venv 隔离依赖，不修改全局环境。2026-09-20 只读检查发现系统 Python 3.13.5 及 NumPy，可视化和测试依赖尚未安装；安装后记录实际版本。

**Spec:** ../specs/2026-09-20-oblique-radar-simulation-design.md。补充研究背景：../../paired-radar-feasibility.md。

## Global Constraints

- 高度 60 m、相对水平面向下 45°、水平视场角 60°：均为仿真假设。
- 距离区间 [0,256) m；256 桶时 1 m/桶，128 桶时 2 m/桶。
- 有效模拟样本每周期 200 个回波；128 桶严格由同一次 256 桶采样合并。
- 初始无偏航、无横滚，雷达与相机近似共址共轴，地面 Z=0。
- 标签条件化数据必须带来源标记，不宣称真实雷达性能。
- 不训练模型、不下载完整数据集、不处理跟踪；不自动创建外部项目。
- 用户于 2026-09-21 要求将本项目初始化为 Git 仓库并上传 GitHub；后续任务仅在本项目仓库内进行对应提交，不向无关父目录提交文件。

## Review Focus

以下五类容易遗漏的输入已分别加入任务测试：

1. 半开区间上界：256 m 不得落到末桶；任务 2。
2. 看向地平线以上、地面可见但全超量程：前者返回无效，后者拒绝模拟；任务 1、3。
3. 无车辆与不合法车辆框不同：空列表合法，负宽高或 NaN 框报错；任务 4。
4. PNG 和 JSON 中图像尺寸不一致：报错，不能默默缩放；任务 5。
5. 重复运行输出覆盖和来源误标：非空输出目录拒绝覆盖；合成图必须标注 schematic；任务 5。

## 文件与接口

```text
pyproject.toml
README.md
.gitignore
configs/oblique_default.json
examples/schematic.json
src/rangefusion/__init__.py
src/rangefusion/config.py       # 参数数据类与输入检查
src/rangefusion/geometry.py     # 射线、地面交点与雷达斜距
src/rangefusion/histogram.py    # 半开区间统计、200 次采样、二合一
src/rangefusion/distribution.py # 车辆、地面、杂波分量
src/rangefusion/simulation.py   # 从框到样本的纯数值流程
src/rangefusion/export.py       # npz/json 写入与读回检查
src/rangefusion/plotting.py     # 三面板检查图
src/rangefusion/cli.py          # 命令行入口
tests/test_geometry.py
tests/test_histogram.py
tests/test_distribution.py
tests/test_simulation.py
tests/test_cli.py
```

所有步骤命令在项目根目录 PowerShell 下执行。测试和代码导入均使用包名 `rangefusion`。

## Task 1: 几何与配置

**Files:** 创建 pyproject.toml、.gitignore、config.py、geometry.py、__init__.py、tests/test_geometry.py。

**Interfaces:**

- `SimulationConfig`：冻结 dataclass，字段 `width=640, height=512, altitude_m=60.0, depression_deg=45.0, hfov_deg=60.0, max_range_m=256.0, ground_rays=4096, target_sigma_m=2.0, clutter_sigma_m=4.0, weights=(0.3,0.6,0.1), target_dropout=0.0`。256 桶及 200 次为本阶段固定常量。
- `intersect_ground(uv: ndarray[N,2], cfg: SimulationConfig) -> tuple[ndarray[N,3], ndarray[N], ndarray[N]]`，依次返回世界交点、斜距、bool 有效掩码。无交点位置填 NaN；非有限输入 uv 直接报错。
- Config 创建时检查正整数图像尺寸和射线数、有限正高度/量程/标准差、`0<depression_deg<=90`、`0<hfov_deg<180`、三个有限非负权重且总和>0，以及 `0<=target_dropout<=1`。

- [ ] 创建包元数据，声明 `numpy, matplotlib, pillow` 及测试 extra `pytest`；控制台入口 `rangefusion-sim = rangefusion.cli:main`。创建 .venv 并安装本项目：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

.gitignore 包含 `.venv/`、`__pycache__/`、`.pytest_cache/`、`*.egg-info/`、`outputs/` 和 `data/`。不忽略 configs、examples、tests。

- [ ] 添加解析几何测试，运行确认尚未实现时失败：

```python
import numpy as np
import pytest
from rangefusion.config import SimulationConfig
from rangefusion.geometry import intersect_ground

def test_center_and_left_right_symmetry():
    cfg = SimulationConfig()
    uv = np.array([[319.5,255.5], [219.5,255.5], [419.5,255.5]])
    points, ranges, valid = intersect_ground(uv, cfg)
    assert valid.all()
    np.testing.assert_allclose(points[0], [0,60,0], atol=1e-10)
    assert ranges[0] == pytest.approx(60*np.sqrt(2))
    assert ranges[1] == pytest.approx(ranges[2])

def test_upward_ray_is_invalid():
    cfg = SimulationConfig(depression_deg=5)
    _, ranges, valid = intersect_ground(np.array([[319.5,0.0]]), cfg)
    assert not valid[0]
    assert np.isnan(ranges[0])

@pytest.mark.parametrize('value', [0, -1, float('nan')])
def test_invalid_height(value):
    with pytest.raises(ValueError):
        SimulationConfig(altitude_m=value)
```

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_geometry.py -q`。

- [ ] 实现 dataclass 和射线求交，核心数值路径：

```python
theta = np.deg2rad(cfg.depression_deg)
f = cfg.width / (2*np.tan(np.deg2rad(cfg.hfov_deg)/2))
x = (uv[:,0]-(cfg.width-1)/2)/f
y = (uv[:,1]-(cfg.height-1)/2)/f
d = np.column_stack((x, np.cos(theta)-y*np.sin(theta),
                     -np.sin(theta)-y*np.cos(theta)))
valid = d[:,2] < -1e-8
t = np.full(len(uv), np.nan)
t[valid] = -cfg.altitude_m/d[valid,2]
points = np.array([0,0,cfg.altitude_m]) + t[:,None]*d
ranges = t*np.linalg.norm(d,axis=1)
```

- [ ] 增加相同中心像素但非方形图像、空 uv 及非有限 uv 检查；重复运行本任务测试至通过。空 uv 用 shape `(0,2)` 返回三个空数组。

## Task 2: 距离直方图与采样

**Files:** 创建 histogram.py、tests/test_histogram.py。

**Interfaces:** `histogram_ranges(ranges_m, max_range_m=256.0) -> ndarray[256]`；`sample_counts(probabilities, seed) -> ndarray[256]`；`merge_adjacent(counts) -> ndarray[128]`。内部一维数组，最终导出再添加通道维。

- [ ] 添加边界、计数守恒和可复现测试：

```python
import numpy as np
import pytest
from rangefusion.histogram import histogram_ranges, sample_counts, merge_adjacent

def test_half_open_range():
    c = histogram_ranges(np.array([-1, 0, 1, 255.9, 256, np.nan]))
    assert c.sum() == 3
    assert c[0] == c[1] == c[255] == 1

def test_reproducible_and_conservative():
    p = np.zeros(256); p[10:12] = [0.4,0.6]
    a = sample_counts(p, seed=7)
    np.testing.assert_array_equal(a, sample_counts(p, seed=7))
    assert a.sum() == 200
    assert np.issubdtype(a.dtype, np.integer)
    assert (a >= 0).all()
    merged = merge_adjacent(a)
    assert merged[5] == 200
    assert merged.sum() == 200

def test_empty_distribution_rejected():
    with pytest.raises(ValueError):
        sample_counts(np.zeros(256), seed=7)
```

- [ ] 运行 `.\.venv\Scripts\python.exe -m pytest tests/test_histogram.py -q` 确认失败。
- [ ] 实现半开区间筛选，不依赖 np.histogram 将最右边界纳入最后一个桶的默认行为：

```python
valid = np.isfinite(ranges_m) & (ranges_m >= 0) & (ranges_m < max_range_m)
counts = np.histogram(ranges_m[valid], bins=np.linspace(0,max_range_m,257))[0]
```

采样前验证长度为 256、值有限且非负、总和>0，再执行 `np.random.default_rng(seed).multinomial(200, p/p.sum())`。合并时验证一维偶数长度为 256、非负整数，再执行 `counts.reshape(128,2).sum(axis=1)`。

- [ ] 增加负概率、无限概率和非整数计数拒绝测试并运行本任务全部测试。

## Task 3: 目标、地面和杂波分布

**Files:** 创建 distribution.py、tests/test_distribution.py。

**Interfaces:** `build_components(target_ranges_m, cfg, seed) -> Components`。`Components` dataclass 包含长度 256 的 `target, ground, clutter, mixture`，`effective_weights` 三元组，`ground_xy` 二维坐标数组及 `target_kept` 布尔数组。

- [ ] 添加无车辆、确定性分布与无地面交集测试：

```python
import numpy as np
import pytest
from rangefusion.config import SimulationConfig
from rangefusion.distribution import build_components

def test_no_targets_preserves_background():
    c = build_components(np.empty(0), SimulationConfig(), seed=12)
    assert c.target.sum() == 0
    assert c.effective_weights[0] == 0
    assert c.mixture.sum() == pytest.approx(1)
    assert c.ground.sum() == pytest.approx(1)
    assert c.clutter.sum() == pytest.approx(1)

def test_ground_outside_range_rejected():
    with pytest.raises(ValueError, match='ground'):
        build_components(np.empty(0), SimulationConfig(altitude_m=300), seed=12)

def test_same_distance_targets_overlap():
    cfg = SimulationConfig()
    a = build_components(np.array([80.]), cfg, seed=12)
    b = build_components(np.array([80.,80.]), cfg, seed=12)
    np.testing.assert_allclose(a.target, b.target)
```

- [ ] 运行 `.\.venv\Scripts\python.exe -m pytest tests/test_distribution.py -q` 确认失败。
- [ ] 用 `SeedSequence(seed).spawn(3)` 为地面射线、杂波、目标丢失创建独立随机流。地面抽样 `(u,v)` 位于 `[0,W-1] × [0,H-1]`；调用任务 1 和 2，过滤无效/超量程点。无有效地面点时抛出包含 `ground` 的 ValueError。
- [ ] 用桶中心 `np.linspace(0, max_range_m, 257)` 的相邻均值计算高斯核。对每个有效目标先构造核并独立归一化，再等权混合；避免直接指数下溢，先减去核最大 log 值：

```python
log_kernel = -0.5*((centers-range_m)/sigma_m)**2
kernel = np.exp(log_kernel-log_kernel.max())
kernel /= kernel.sum()
```

杂波使用 `0.5*uniform + 0.25*peak1 + 0.25*peak2`；两峰位置在量程内均匀抽样，标准差取配置。目标丢失用独立随机流，全部丢失按无目标处理。按设计权重混合，移除不存在的分量后权重总和为零时抛出 ValueError。

- [ ] 增加 dropout=1、背景主导权重、配置相同可复现测试，运行本任务测试。检查每个非空分量有限、非负且和为 1。

## Task 4: 从车辆框到完整模拟样本

**Files:** 创建 simulation.py、configs/oblique_default.json、examples/schematic.json、tests/test_simulation.py。

**Interfaces:** `simulate(boxes_xyxy, cfg, seed) -> SimulationResult`。结果 dataclass 包含 `counts_256` shape `(1,256)`、`counts_128` shape `(1,128)`、`bin_edges_256_m`、`components`、`target_points_world`、`target_ranges_m`、`target_status`、`metadata`。状态为 `valid / invalid_ray / out_of_range / dropped`。

- [ ] 添加端到端数值测试：

```python
import numpy as np
import pytest
from rangefusion.config import SimulationConfig
from rangefusion.simulation import simulate

def test_empty_scene_is_valid_simulation():
    r = simulate(np.empty((0,4)), SimulationConfig(), seed=42)
    assert r.counts_256.shape == (1,256)
    assert r.counts_128.shape == (1,128)
    assert r.counts_256.sum() == r.counts_128.sum() == 200
    assert r.metadata['source_type'] == 'label_conditioned_proxy'

def test_bad_box_rejected():
    with pytest.raises(ValueError):
        simulate(np.array([[50,50,10,10]]), SimulationConfig(), seed=42)

def test_channel_counts_are_exact_merge():
    r = simulate(np.array([[300,240,339,271]]), SimulationConfig(), seed=42)
    np.testing.assert_array_equal(r.counts_128[0],
                                 r.counts_256[0].reshape(128,2).sum(axis=1))
```

- [ ] 运行 `.\.venv\Scripts\python.exe -m pytest tests/test_simulation.py -q` 确认失败。
- [ ] 检查框 shape `(N,4)`、有限值、正宽高及图像范围 `[0,W]×[0,H]`；以框中心调用几何函数。独立保留无效车辆标签及状态，将有效量程内斜距传给分量生成器；外层再派生独立 seed 用于最终计数采样。
- [ ] 构造以下明确格式的示例 JSON，路径相对于 JSON 文件解析；省略 `image_path` 表示合成示意：

```json
{"sample_id":"schematic-001","width":640,"height":512,
 "source_type":"schematic","boxes_xyxy":[[180,200,220,230],[400,200,440,230]]}
```

默认配置 JSON 与 `SimulationConfig` 字段逐一对应；读取时拒绝未知字段，防止拼写错误被静默忽略。

- [ ] 增加超量程车辆仍保留状态、配置/种子完整记录、NaN 框拒绝测试，运行全部数值测试。

## Task 5: 导出、命令行与检查图

**Files:** 创建 export.py、plotting.py、cli.py、tests/test_cli.py、README.md。

**Interfaces:** `save_result(result, sample, output_dir) -> None`；`render_result(result, sample, output_path) -> None`；`main(argv=None) -> int`。`sample` 为示例 JSON 格式字典，可选解析后的绝对 `image_path`。

- [ ] 添加临时目录的命令行验收测试，实际调用 `main`，不 mock 数值核心：

```python
import json
import numpy as np
import pytest
from rangefusion.cli import main

def test_cli_exports_reproducible_contract(tmp_path):
    out = tmp_path/'result'
    assert main(['--sample','examples/schematic.json',
                 '--config','configs/oblique_default.json',
                 '--seed','42','--output',str(out)]) == 0
    with np.load(out/'histogram.npz', allow_pickle=False) as data:
        assert data['counts_256'].shape == (1,256)
        assert data['counts_256'].sum() == 200
    meta = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    assert meta['image_source'] == 'schematic'
    assert meta['source_type'] == 'label_conditioned_proxy'
    assert (out/'inspection.png').is_file()
    assert main(['--sample','examples/schematic.json',
                 '--config','configs/oblique_default.json',
                 '--seed','42','--output',str(out)]) == 2
```

- [ ] 运行 `.\.venv\Scripts\python.exe -m pytest tests/test_cli.py -q` 确认失败。
- [ ] `argparse` 实现上述必需参数。配置尺寸必须与样本尺寸一致；如果提供真实图像，使用 Pillow 检查尺寸一致。显式错误打印 stderr 并返回 2；不自动拉伸或覆盖已有非空输出目录。
- [ ] 用 `np.savez_compressed` 保存整数计数、距离边界和三个期望分量；用 `json.dump(..., ensure_ascii=False, allow_nan=False)` 保存配置、种子、版本、样本 ID、来源和状态。无效数值在诊断 JSON 中转为 null，不写非标准 NaN。
- [ ] Matplotlib 使用 Agg 后端绘制三面板：图像和框、世界 XY 地面投影、实际计数与三个 `200*effective_weight*component` 期望曲线。合成图标题必须含 `Schematic - not thermal data`；总图标注 `Simulated radar proxy`、角度定义、量程和种子。
- [ ] 增加 Pillow 生成尺寸不匹配图像的测试，要求 main 返回 2 且不产生可误用的样本；增加空样本可视化测试。运行本任务测试。
- [ ] README 写明来源与证据边界、安装、运行命令及输出字段：

```powershell
.\.venv\Scripts\python.exe -m rangefusion.cli --sample examples/schematic.json --config configs/oblique_default.json --seed 42 --output outputs/demo-42
```

`cli.py` 用 `if __name__ == '__main__': raise SystemExit(main())` 支持模块调用。

## Task 6: 验收报告与真实数据衔接

**Files:** 创建 docs/simulation-validation.md；仅在执行后记录结果，不预填“通过”。

**Interfaces:** 消费任务 1–5 的完整包，输出测试结果、检查图路径、依赖版本和限制说明。

- [ ] 运行 `.\.venv\Scripts\python.exe -m pytest -q`；失败时修复后只重跑受影响测试和最终全套。
- [ ] 运行 README 的示例命令，检查保存的 PNG，确认文字清晰、框和峰可辨认、计数与期望曲线没有混淆。
- [ ] 增加无车辆、同距离双车辆、背景主导、杂波伪峰、超量程车辆和目标明显六个独立输出案例，复用固定种子、明确记录各配置。所有合法输出必须总和 200；无地面交集案例必须失败。
- [ ] 用 `python -m pip freeze` 记录项目环境版本到报告的环境部分；不声称未运行的性能指标。
- [ ] 将模拟验收与配对数据审计分开。FlexSense 后续审计需核实：雷达 bin 字段与单位、实际标定方向、时间匹配、图像大小、类别和 ID、原始/裁取后点数、量程占用及空帧比例。此时才能制定真实数据适配器的字段映射和测试。

## Self-review 与执行交接

- 设计中的几何、无效输入、回波模型、计数约束、种子、元数据和可视化均已映射到上述任务。
- 五项 Review Focus 已归属具体任务；首阶段不依赖未下载的真实数据。
- 此文件是实施计划，示例代码尚未写入产品源文件，也未运行上述测试。
- 建议由当前会话逐项实施：任务存在明确的接口依赖，规模较小，暂不需要子代理。用户已批准模拟参数；本轮新增点云路线为分析建议，未替代已确认模拟器设计。
