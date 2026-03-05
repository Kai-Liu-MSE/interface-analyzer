# Interface Analyzer Manual

`interface-analyzer` 用于分子动力学（MD）中固液界面识别，以及 Capillary Fluctuation Method (CFM) 后处理。

## 1. 功能概览

- 固液界面识别（`x-z` 平面）：
  - `CSP`：Centrosymmetry Parameter
  - `PTM`：Polyhedral Template Matching
  - `LOP`（距离型）
  - `OrientationPhi`（向量匹配型，需指定晶体取向）
- Brown maximize-difference 方法提取 `h_upper(x)` 与 `h_lower(x)`
- CFM 频谱分析：`<|A(k)|^2>`、`k^2` 线性拟合、拟合敏感性分析
- 支持自定义 `PhaseModifierBase` 扩展

## 2. 安装

```bash
pip install .
```

依赖（由 `setup.py` 管理）：

- `numpy`
- `scipy`
- `matplotlib`
- `ovito`
- `tqdm`

建议 Python `>=3.8`。

## 3. 包结构与公开 API

主模块：`interface_analyzer`

### 3.1 接口识别函数

- `analyze_by_CSP(...)`
- `analyze_by_PTM(...)`
- `analyze_by_LOP(...)`
- `analyze_by_OrientationPhi(...)`
- `analyze_by_custom_modifier(...)`
- `LOP_analysis(...)`（网格版距离型）
- `Orientation_analysis(...)`（网格版取向向量型）

### 3.2 CFM 分析函数

- `analyze_cfm(...)`
- `plot_cfm_k2_single(...)`
- `analyze_cfm_fit_sensitivity(...)`

### 3.3 可扩展类/工具

- `PhaseModifierBase`
- `CSPModifier`
- `PTMModifier`
- `LOPModifier`
- `OrientationPhiModifier`
- `get_orientation_matrix(...)`

## 4. 两套 LOP 思路（重点）

## 4.1 距离型 LOP（不需要取向）

- 典型入口：`analyze_by_LOP`、`LOP_analysis`
- 思路：仅比较近邻键长与理想键长 `r_fcc` 的偏差
- 适合：你不想显式指定晶体朝向，或体系取向变化较复杂的情况

## 4.2 取向向量型 LOP / OrientationPhi（需要取向）

- 典型入口：`analyze_by_OrientationPhi`、`Orientation_analysis`
- 思路：把实际近邻向量与旋转后的 FCC 理想向量集做匹配，计算偏差
- 需要输入：`miller_x`, `miller_y`, `miller_z`
- 要求：三组 Miller 方向两两正交（`get_orientation_matrix` 会检查）

## 5. 最小工作流（推荐）

## 5.1 单帧界面识别

```python
from interface_analyzer import analyze_by_PTM

res = analyze_by_PTM(
    cfg_path="cfg.100000",
    binsx=150,
    binsz=300,
    n=15,
    rmsd_max=0.10,
)

print(res.keys())
# dict_keys(['phase', 'M', 'x', 'z', 'h_upper', 'h_lower', 'cell'])
```

## 5.2 多帧批处理并保存 `pkl`

可参考仓库中的 `Process.py`。核心是把每一帧输出存成：

```python
results_all[step_id] = {
    "phase": ...,
    "M": ...,
    "x": ...,
    "z": ...,
    "h_upper": ...,
    "h_lower": ...,
    "cell": ...,
}
```

然后：

```python
import pickle
with open("cfg_post.pkl", "wb") as f:
    pickle.dump(results_all, f)
```

## 5.3 CFM 主分析

```python
from interface_analyzer import analyze_cfm

cfm = analyze_cfm(
    pickle_path="cfg_post.pkl",
    T=933.0,          # K
    a=4.05,           # 当前版本仅记录到输出中
    use_pchip=True,
    pchipres=1000,
    show_plot=True,
)
```

返回结果包含：

- `k`, `k2`
- `Smax_mean`, `Smin_mean`（上下界面谱平均）
- `Ak_max`, `Ak_min`（反演量）
- `Lx`, `Ly`, `Lz`

## 5.4 `k^2` 线性拟合（单数据文件）

```python
from interface_analyzer import plot_cfm_k2_single

fit_res = plot_cfm_k2_single(
    filename="cfm_k2_data.txt",
    k2_min=4.0e-4,
    min_points=5,
    a_lattice=4.05,
    L_min_interface=10,
    through_origin=False,
)
print(fit_res)
```

输入文件默认三列：

1. `k2`
2. 上界面数据
3. 下界面数据

函数内部使用第 2、3 列均值拟合。

## 5.5 拟合敏感性分析

```python
from interface_analyzer import analyze_cfm_fit_sensitivity

sense = analyze_cfm_fit_sensitivity(
    filename="cfm_k2_data.txt",
    k2_min=4.0e-4,
    min_points=5,
    a_lattice=4.05,
    L_min_interface=10,
    through_origin=False,
)
```

## 6. 关键参数建议

- `binsx`, `binsz`：网格分辨率。`binsz` 常设为 `2*binsx` 或按盒子纵横比调整。
- `n`：Brown 窗口半宽。你当前实践里常用 `box_size/20` 量级。
- `rmsd_max`：PTM 判定严格度，越小越严格。
- `r_fcc`：距离型 LOP 的理想键长，建议与材料与温度状态匹配。
- `d`：平滑半径。过小噪声大，过大过度平滑。
- `pchipres`：CFM 插值分辨率，越高越平滑但计算更慢。

## 7. 自定义 Modifier（扩展）

你可以继承 `PhaseModifierBase`，实现两件事：

1. `apply_modifier(self, node)`：向 OVITO pipeline 注入你的处理逻辑
2. `get_property_name(self)`：返回用于后续 binning 的粒子属性名

然后调用：

```python
from interface_analyzer import analyze_by_custom_modifier

res = analyze_by_custom_modifier("cfg.100000", custom_modifier_instance=my_modifier)
```

## 8. 输出数据约定

界面识别函数统一返回字典：

- `M`: `(binsz, binsx)` 的序参量场
- `x`: `x` 网格中心坐标
- `z`: `z` 网格中心坐标
- `h_upper`: 上界面高度函数
- `h_lower`: 下界面高度函数
- `phase`: 相区分类矩阵（默认固相 `1`，液相 `2`）
- `cell`: 模拟盒矩阵

## 9. 引用与致谢

如果你在论文中使用本包，建议在方法部分说明：

- 使用了 OVITO 的结构分析工具（CSP/PTM）
- 使用 Brown maximize-difference 提取界面
- 使用 CFM 的 Fourier 频谱关系进行刚度拟合

---

如需我帮你再补一版英文文档（`MANUAL_EN.md`）和 PyPI 风格 README，我可以直接继续生成并整理成发布版本。
