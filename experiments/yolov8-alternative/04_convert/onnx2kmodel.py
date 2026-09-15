"""
ONNX → KModel 转换脚本（K230 专用）
功能：将 ONNX 模型量化并转换为 K230 可运行的 .kmodel 格式
使用方法：
  1. 安装 nncase: pip install nncase
  2. 准备校准图片集（100~300 张真实采集的图片）
  3. 修改下面的配置参数
  4. 运行: python onnx2kmodel.py

注意事项：
  - 校准集非常重要！必须用真实采集的图片，否则量化后精度会崩
  - 推荐用 K230 自己拍的图片做校准
"""

import os
import shutil
import random

# ==================== 配置参数 ====================
# 输入 ONNX 模型路径
ONNX_MODEL = r"D:\k230\ball_detector\yolov8n_320\weights\best.onnx"

# 输出 kmodel 路径
OUTPUT_KMODEL = r"D:\k230\ball_detector\yolov8n_320\weights\best_k230.kmodel"

# 校准图片文件夹（用真实采集的图片，100~300 张）
CALIB_DIR = r"D:\k230\ball_yolo\images\train"

# 校准图片数量（用多少张做校准，越多越准但越慢）
CALIB_COUNT = 200

# 目标平台
TARGET = "k230"

# 量化方法: "kld" 或 "noclip"
# kld: 精度更好，适合分类/检测
# noclip: 速度更快，精度稍低
CALIBRATE_METHOD = "kld"

# 量化类型: "uint8" 或 "int8"
QUANT_TYPE = "uint8"

# 输入图像尺寸
IMG_SIZE = 320

# 输入均值和标准差（YOLOv8 默认是 0~255 归一化到 0~1）
# 注意：K230 推理时的预处理必须和这里一致！
INPUT_MEAN = [0, 0, 0]
INPUT_STD = [255, 255, 255]
# ==================================================


def prepare_calib_dataset(src_dir, dst_dir, count):
    """
    从源目录随机选图片作为校准集
    nncase 需要校准图片放在单独的文件夹里
    """
    os.makedirs(dst_dir, exist_ok=True)

    # 收集所有图片
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_imgs = [f for f in os.listdir(src_dir) if f.lower().endswith(exts)]

    if not all_imgs:
        print(f"[错误] 源目录没有图片: {src_dir}")
        return None

    # 随机选取
    random.seed(42)
    selected = random.sample(all_imgs, min(count, len(all_imgs)))

    # 复制到校准目录
    for f in selected:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    print(f"[校准集] 从 {len(all_imgs)} 张中选了 {len(selected)} 张")
    print(f"  校准集目录: {dst_dir}")
    return dst_dir


def convert_with_nncase_python_api():
    """使用 nncase Python API 转换"""
    try:
        import nncase
        from nncase import CompileOptions, ImportOptions
    except ImportError:
        print("[错误] 未安装 nncase")
        print("请运行: pip install nncase")
        return False

    print("[转换] 使用 nncase Python API...")

    # 准备校准集
    calib_dir = os.path.join(os.path.dirname(OUTPUT_KMODEL), "calib_images")
    calib_dir = prepare_calib_dataset(CALIB_DIR, calib_dir, CALIB_COUNT)
    if not calib_dir:
        return False

    # 编译选项
    compile_options = CompileOptions()
    compile_options.target = TARGET
    compile_options.calibrate_method = CALIBRATE_METHOD
    compile_options.quant_type = QUANT_TYPE
    compile_options.input_type = "float32"
    compile_options.output_type = "float32"

    # 导入选项
    import_options = ImportOptions()
    import_options.input_layout = "NCHW"
    import_options.output_layout = "NCHW"

    # 读取 ONNX
    with open(ONNX_MODEL, "rb") as f:
        onnx_data = f.read()

    # 编译
    print("[编译] 正在编译模型...")
    compiler = nncase.Compiler(compile_options)
    compiler.import_onnx(onnx_data, import_options)

    # 使用校准集量化
    print("[量化] 使用校准集量化中...")
    compiler.use_ptq_target(TARGET)
    compiler.calibrate_with_dataset(calib_dir, "image")

    # 编译生成 kmodel
    print("[生成] 生成 kmodel...")
    kmodel_data = compiler.compile()

    # 保存
    with open(OUTPUT_KMODEL, "wb") as f:
        f.write(kmodel_data)

    return True


def convert_with_ncc_command():
    """使用 ncc 命令行工具转换（备选方案）"""
    import subprocess

    print("[转换] 使用 ncc 命令行工具...")

    # 准备校准集
    calib_dir = os.path.join(os.path.dirname(OUTPUT_KMODEL), "calib_images")
    calib_dir = prepare_calib_dataset(CALIB_DIR, calib_dir, CALIB_COUNT)
    if not calib_dir:
        return False

    # 构建命令
    cmd = [
        "ncc", "compile",
        ONNX_MODEL,
        OUTPUT_KMODEL,
        "-i", "onnx",
        "-t", TARGET,
        "--dataset", calib_dir,
        "--dataset-format", "image",
        "--calibrate-method", CALIBRATE_METHOD,
        "--quant-type", QUANT_TYPE,
        "--input-type", "float32",
        "--output-type", "float32",
    ]

    print(f"[命令] {' '.join(cmd)}")
    print("-" * 50)

    # 执行
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("[stderr]", result.stderr)
        return result.returncode == 0
    except FileNotFoundError:
        print("[错误] 找不到 ncc 命令")
        print("请确保已安装 nncase 并配置好环境变量")
        return False


def main():
    print("=" * 50)
    print("  ONNX → KModel 转换工具（K230）")
    print("=" * 50)

    # 检查输入
    if not os.path.exists(ONNX_MODEL):
        print(f"[错误] 找不到 ONNX 模型: {ONNX_MODEL}")
        print("请先运行 export_onnx.py 导出 ONNX 模型")
        return

    print(f"输入 ONNX: {ONNX_MODEL}")
    print(f"输出 KModel: {OUTPUT_KMODEL}")
    print(f"目标平台: {TARGET}")
    print(f"量化方法: {CALIBRATE_METHOD}")
    print(f"量化类型: {QUANT_TYPE}")
    print(f"校准集: {CALIB_DIR} ({CALIB_COUNT} 张)")
    print("-" * 50)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_KMODEL), exist_ok=True)

    # 尝试用 Python API 转换
    success = convert_with_nncase_python_api()

    # 如果 Python API 失败，尝试命令行
    if not success:
        print("\n[重试] Python API 失败，尝试命令行方式...")
        success = convert_with_ncc_command()

    # 结果
    print("-" * 50)
    if success and os.path.exists(OUTPUT_KMODEL):
        size_mb = os.path.getsize(OUTPUT_KMODEL) / 1024 / 1024
        print("[完成] 转换成功！")
        print(f"  输出文件: {OUTPUT_KMODEL}")
        print(f"  文件大小: {size_mb:.2f} MB")
        print("\n下一步:")
        print("  1. 将 best_k230.kmodel 拷贝到 K230 的 SD 卡")
        print("  2. 运行 05_部署代码/detect_ball.py 进行实时检测")
    else:
        print("[失败] 转换失败，请检查错误信息")


if __name__ == "__main__":
    main()
