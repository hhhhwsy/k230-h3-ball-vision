"""
YOLOv8 PyTorch 模型 → ONNX 导出脚本
功能：将训练好的 .pt 模型导出为 .onnx 格式，用于后续转 kmodel
使用方法：
  1. 修改下面的配置参数
  2. 运行: python export_onnx.py

依赖安装：
  pip install ultralytics onnx onnxsim
"""

from ultralytics import YOLO
import os

# ==================== 配置参数 ====================
# 训练好的模型路径
PT_MODEL = r"D:\k230\ball_detector\yolov8n_320\weights\best.pt"

# 输出 ONNX 路径（默认和 pt 同目录）
OUTPUT_ONNX = ""  # 留空则自动生成 best.onnx

# 输入图像尺寸（必须和训练时一致）
IMG_SIZE = 320

# ONNX opset 版本（K230 nncase 推荐用 12）
OPSET = 12

# 是否简化模型
SIMPLIFY = True

# 是否动态尺寸（K230 部署用固定尺寸，关掉）
DYNAMIC = False
# ==================================================


def main():
    print("=" * 50)
    print("  YOLOv8 → ONNX 导出")
    print("=" * 50)

    # 检查模型文件
    if not os.path.exists(PT_MODEL):
        print(f"[错误] 找不到模型文件: {PT_MODEL}")
        print("请先运行 03_训练代码/train_yolov8.py 训练模型")
        return

    # 确定输出路径
    if not OUTPUT_ONNX:
        OUTPUT_ONNX = PT_MODEL.replace(".pt", ".onnx")

    print(f"输入模型: {PT_MODEL}")
    print(f"输出 ONNX: {OUTPUT_ONNX}")
    print(f"输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Opset: {OPSET}")
    print("-" * 50)

    # 加载模型
    print("[加载] 加载模型...")
    model = YOLO(PT_MODEL)

    # 导出 ONNX
    print("[导出] 导出 ONNX 中...")
    exported_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=OPSET,
        simplify=SIMPLIFY,
        dynamic=DYNAMIC,
    )

    print("-" * 50)
    print("[完成] ONNX 导出成功！")
    print(f"  输出文件: {exported_path}")

    # 检查文件大小
    if os.path.exists(exported_path):
        size_mb = os.path.getsize(exported_path) / 1024 / 1024
        print(f"  文件大小: {size_mb:.2f} MB")

    print("\n下一步: 运行 onnx2kmodel.py 将 ONNX 转为 K230 可用的 kmodel")


if __name__ == "__main__":
    main()
