"""
YOLOv8 小铁球检测训练脚本
功能：训练 YOLOv8 模型识别小铁球
使用方法：
  1. 确保数据集已准备好（运行 02_数据集转换 后）
  2. 修改下面的配置参数
  3. 运行: python train_yolov8.py

依赖安装：
  pip install ultralytics
"""

from ultralytics import YOLO
import os

# ==================== 配置参数 ====================
# 数据集配置文件路径
DATA_YAML = r"D:\k230\ball_yolo\data.yaml"

# 模型选择（K230 部署推荐用 nano 版，速度快）
# yolov8n.pt - 最小最快，适合 K230
# yolov8s.pt - 稍大一点，精度更高
MODEL_NAME = "yolov8n.pt"

# 输入图像尺寸（必须和 K230 部署时一致，320 适合小铁球）
IMG_SIZE = 320

# 训练轮次
EPOCHS = 100

# 批次大小（根据显存调整，GPU 显存小就调小）
BATCH_SIZE = 16

# 设备：0=GPU，cpu=CPU
DEVICE = 0  # 有 N 卡用 0，没有就改 "cpu"

# 项目名称（输出文件夹名）
PROJECT_NAME = "ball_detector"

# 实验名称
EXP_NAME = "yolov8n_320"

# 预训练权重（首次训练用官方预训练模型）
PRETRAINED_WEIGHTS = "yolov8n.pt"

# 是否开启数据增强（小数据集建议开启）
AUGMENT = True
# ==================================================


def train():
    """训练模型"""
    print("=" * 50)
    print("  YOLOv8 小铁球检测训练")
    print("=" * 50)
    print(f"数据集: {DATA_YAML}")
    print(f"模型: {MODEL_NAME}")
    print(f"输入尺寸: {IMG_SIZE}")
    print(f"训练轮次: {EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"设备: {DEVICE}")
    print("-" * 50)

    # 检查数据集
    if not os.path.exists(DATA_YAML):
        print(f"[错误] 数据集配置文件不存在: {DATA_YAML}")
        print("请先运行 02_数据集转换/labelme2yolo.py 生成数据集")
        return

    # 加载模型
    print("[加载] 加载模型...")
    model = YOLO(PRETRAINED_WEIGHTS)

    # 开始训练
    print("[训练] 开始训练...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=EXP_NAME,
        exist_ok=True,           # 覆盖已有实验
        patience=20,             # 早停耐心值
        save=True,               # 保存检查点
        save_period=10,          # 每10轮保存一次
        val=True,                # 训练时验证
        augment=AUGMENT,         # 数据增强
        # 小目标检测优化参数
        close_mosaic=10,         # 最后10轮关闭 mosaic
        # 学习率
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # 损失函数权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # 多尺度训练
        multi_scale=False,       # K230 固定尺寸，关掉
    )

    print("\n[完成] 训练结束！")
    print(f"  最佳模型: {PROJECT_NAME}/{EXP_NAME}/weights/best.pt")
    print(f"  最后模型: {PROJECT_NAME}/{EXP_NAME}/weights/last.pt")

    return results


def validate():
    """验证模型"""
    print("\n[验证] 在验证集上评估模型...")
    model_path = f"{PROJECT_NAME}/{EXP_NAME}/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型: {model_path}")
        return

    model = YOLO(model_path)
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
    )

    print("\n[验证结果]")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    return metrics


def predict_test():
    """用几张测试图预测看看效果"""
    print("\n[预测] 测试模型效果...")
    model_path = f"{PROJECT_NAME}/{EXP_NAME}/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型: {model_path}")
        return

    model = YOLO(model_path)

    # 找验证集里的图片测试
    val_img_dir = os.path.join(os.path.dirname(DATA_YAML), "images", "val")
    if os.path.exists(val_img_dir):
        test_imgs = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir)[:3]]
    else:
        print("[警告] 找不到验证集图片，跳过预测")
        return

    results = model.predict(
        test_imgs,
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        project=PROJECT_NAME,
        name=f"{EXP_NAME}_predict",
    )

    print(f"  预测结果保存在: {PROJECT_NAME}/{EXP_NAME}_predict/")
    return results


def export_onnx():
    """导出 ONNX 模型（用于后续转 kmodel）"""
    print("\n[导出] 导出 ONNX 格式...")
    model_path = f"{PROJECT_NAME}/{EXP_NAME}/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"[错误] 找不到模型: {model_path}")
        return

    model = YOLO(model_path)
    exported = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=12,          # 必须 12，K230 nncase 支持最好
        simplify=True,     # 简化模型
        dynamic=False,     # 固定输入尺寸
    )

    print(f"  ONNX 模型路径: {exported}")
    return exported


if __name__ == "__main__":
    # 训练
    train()

    # 验证
    validate()

    # 测试预测
    predict_test()

    # 导出 ONNX
    export_onnx()

    print("\n" + "=" * 50)
    print("  全部完成！")
    print("=" * 50)
    print("下一步: 运行 04_模型转换/ 中的脚本将 ONNX 转为 kmodel")
