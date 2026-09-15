# 备选方案：YOLOv8n 自训练钢球检测（评估后未采用）

这是本项目初期并行评估的深度学习路线。**从板载采图到量化转换的链路是通的**，但最终没有选它作为正式方案——当时的取舍考虑见主仓库 README 的「关于技术路线的说明」一节。

保留在这里的原因有两个：一是它是一段相对完整的工程过程，二是技术选型的判断过程本身值得记录。

> **说明**：本目录代码同样在 AI 编程助手辅助下生成（见主 README「开发方式说明」）。本节描述的是**脚本实际做的事**，不代表作者能给出对应的实验数据。

---

## 一、为什么需要自己训练

COCO 预训练权重里没有「小钢球」这个类别。其中相近的 `sports ball` 在本场景的综合光照与金属反光下误检、漏检严重，无法作为闭环控制的输入。所以必须用**真实场景的光照、背景、距离和相机角度**自行采集并标注。

---

## 二、完整链路

```text
板载采图 ──→ 标注 ──→ 训练 ──→ ONNX ──→ nncase 量化 ──→ .kmodel ──→ 板端推理
01_capture  02_dataset  03_train  04_convert  04_convert   05_deploy
```

### 1. 采集（`01_capture/capture_steel_ball.py`）

程序跑在开发板上，按板载 USER 键（GPIO53，Lite K230D 需改为 GPIO64）触发保存一张照片到 `/sdcard/steel_ball_dataset/images/raw/`。

### 2. 数据集规模

| 项目 | 数量 |
|---|---|
| 原始采集 | 235 张 |
| 标注后进入数据集 | 220 张 |
| train / val / test | 142 / 51 / 27 |
| 类别 | 单一类别 `steel_ball` |

**场景覆盖设计**（决定模型能不能用）：

- 真实场景的背景、灯光与相机安装高度
- 近 / 中 / 远距离，尤其是远处只有十几个像素的小球
- 单球 / 多球 / 部分遮挡 / 相互接触 / 位于画面边缘
- 强反光、阴影、过曝、较暗环境、不同拍摄角度
- 硬负样本：螺丝、轴承、灰色圆点、灯光反射等与钢球相似的干扰物

**标注规则**：框紧贴球可见外缘；有 5 个球就画 5 个框，重叠的球不能用一个大框包起来；清晰可见的球必须全部标出；负样本不画框但必须保留空标签文件。

标注工具是自研的 `tools/steel_ball_annotator.py`（框选 + 快捷键 + 直接导出 YOLO txt），也可以用 labelme / AnyLabeling / CVAT。

### 3. 训练（`03_train/train_yolov8.py`）

```python
MODEL_NAME   = "yolov8n.pt"   # nano 版，K230 算力下的现实选择
IMG_SIZE     = 320            # 必须与部署端 MODEL_INPUT_SIZE 完全一致
EPOCHS       = 100
BATCH_SIZE   = 16
patience     = 20             # 早停
close_mosaic = 10             # 末段关闭 mosaic，提升小目标定位精度
lr0 / lrf    = 0.01 / 0.01
multi_scale  = False          # K230 输入尺寸固定，多尺度训练无意义
```

实际超参存档见 `03_train/args.yaml`。

### 4. 导出与量化（`04_convert/`）

```python
model.export(format="onnx", imgsz=320, opset=12, simplify=True, dynamic=False)
```

`opset=12` + `simplify` + 固定输入，是 nncase 兼容性最好的组合。随后经 `onnx2kmodel.py` 量化，得到约 6.4 MB 的 `.kmodel`。

> **量化校准集的图片必须来自最终摄像头与真实场景环境**，否则反光金属球在 PTQ 后漏检会非常严重。

### 5. 板端推理（`05_deploy/`）

两个版本：

- **`main.py`** —— 基于固件自带 `libs.PipeLine` + `libs.YOLO`，CSI2 输入，AI 推理帧 640×360，置信度 0.65，NMS 0.45，最多 50 框。LCD OSD 绘制检测框，并用十字标出置信度最高目标的中心；终端每 10 帧打印一次数量、坐标、置信度。
- **`detect_ball_raw_nms.py`** —— 不依赖固件 YOLO 库，自行实现图像预处理、置信度筛选、坐标变换、IoU 与 NMS。因为 CanMV 的 `ulab.numpy` 算子受限，全部后处理用显式循环实现，用于理解并验证固件库内部到底做了什么。

---

## 三、验证状态

**这个方案没有被选为正式方案，因此也没有走完最后的精度验证。** 诚实记录如下：

| 阶段 | 状态 |
|---|---|
| 采集脚本 / 标注工具 / 训练脚本 / 导出与量化脚本 | 齐全，可复现 |
| `.kmodel` 转换 | 已完成（约 6.4 MB） |
| 板端部署代码 | 已编写（`05_deploy/` 两个版本） |
| 训练权重 `best.pt` | **未保留**（`weights/` 为空） |
| mAP@50 / Precision / Recall | **无实测数据** |
| 板端帧率 | 未系统测量 |
| 换场地泛化 | 未验证，数据集以单一场地为主 |

> 本表只标注**文档与代码层面的完成度**。具体每一步在实机上验证到什么程度，以作者本人的调试记录为准。

如果要把这条路线当作独立作品展示，需要先补做一次训练并记录 `results.csv` 的真实指标——**不要把估计值写进任何文档**。

---

## 四、踩过的坑

> 下表是从调试过程和代码注释中整理出来的，**不是逐条都有完整记录**。其中"原因"一列有些是当时的推断，属于事后解释，仅供参考。

| 问题 | 原因 | 解决 |
|---|---|---|
| 量化后金属球漏检严重 | 钢球强反光，PTQ 校准图分布与实际场景差异大 | 校准集改用同一摄像头与实际光照下的图片，必要时补采强反光样本重训 |
| 固件升级后程序起不来 | v1.2.2 起 Sensor 需显式导入 | `from media.sensor import Sensor`，并在 `PipeLine.create()` 前手动实例化 |
| 长时间运行内存耗尽 | 每帧产生新的图像与推理缓冲区 | 主循环内显式 `gc.collect()`；`finally` 中 `detector.deinit()` + `pipeline.destroy()` |
| 推理结果与训练效果不符 | 三处输入尺寸不一致 | 训练 `imgsz`、ONNX 导出 `imgsz`、`.kmodel` 转换 `--input_width/--input_height` 必须三处相同 |
| 板端后处理写不动 | `ulab.numpy` 缺少完整向量化算子 | 改用显式循环实现筛选、坐标变换与 NMS |
| 远处小球检不到 | 小目标在 320 输入下只剩十几个像素 | 训练集补充远距离样本；或提高输入分辨率并接受帧率下降 |
| 板端库与固件不匹配 | `libs/PipeLine.py`、`libs/YOLO.py` 与固件强绑定 | 必须使用与烧录固件配套的官方示例库，不要混用旧版 |

---

## 五、目录结构

```text
├─ 01_capture/capture_steel_ball.py   板端采图：USER 键触发
├─ 02_dataset/labelme2yolo.py         labelme JSON → YOLO txt，划分 train/val/test
├─ 02_dataset/steel_ball.yaml         数据集配置（单类别）
├─ 03_train/train_yolov8.py           训练 / 验证 / 预测 / 导出 ONNX
├─ 03_train/args.yaml                 实际训练超参存档
├─ 04_convert/export_onnx.py          导出 ONNX（opset 12、固定输入）
├─ 04_convert/onnx2kmodel.py          nncase 量化转换
├─ 05_deploy/main.py                  板端主程序（PipeLine + libs.YOLO）
├─ 05_deploy/detect_ball_raw_nms.py   手写后处理版：自行实现 IoU / NMS
├─ tools/steel_ball_annotator.py      自研标注工具
├─ dataset/samples/                   12 张样本图 + 对应标签
├─ dataset/annotations/               完整 labelme 原始标注
└─ docs/                              数据采集与训练流程、K230 部署说明
```

---

## 六、复现步骤

```bash
# 1. 采集：把 01_capture/capture_steel_ball.py 拷到开发板运行，按 USER 键逐张采图
# 2. 标注：python tools/steel_ball_annotator.py
# 3. 转换数据集格式并划分
python 02_dataset/labelme2yolo.py
# 4. 训练（需 ultralytics，建议有 N 卡）
python 03_train/train_yolov8.py
# 5. 导出 ONNX
python 04_convert/export_onnx.py
# 6. 转 kmodel（需与本机固件匹配的 nncase）
python 04_convert/onnx2kmodel.py
# 7. 部署：把 .kmodel 与 05_deploy/main.py 放到 SD 卡同一目录，运行 main.py
```
