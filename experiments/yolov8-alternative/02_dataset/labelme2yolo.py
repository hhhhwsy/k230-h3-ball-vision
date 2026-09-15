"""
Labelme 标注格式 → YOLO 训练格式 转换脚本
功能：将 Labelme 标注的 JSON 文件批量转换为 YOLOv8 训练所需的格式
使用方法：
  1. 把所有图片和对应的 .json 标注文件放在同一个文件夹（如 ball_dataset）
  2. 修改下面的配置参数
  3. 运行: python labelme2yolo.py

依赖安装：
  pip install scikit-learn
"""

import os
import json
import shutil
import random

# ==================== 配置参数 ====================
# Labelme 标注文件夹（包含 .jpg 和 .json）
LABELME_DIR = r"D:\k230\ball_dataset"

# 输出的 YOLO 格式数据集目录
OUTPUT_DIR = r"D:\k230\ball_yolo"

# 类别列表（顺序很重要，对应 YOLO 的 class_id）
CLASSES = ["ball"]  # 只有小铁球一个类别

# 验证集比例
VAL_RATIO = 0.2  # 20% 用作验证集

# 随机种子（保证每次划分一致）
SEED = 42
# ==================================================

def convert_bbox_to_yolo(shape, img_w, img_h):
    """
    将 Labelme 的矩形/多边形标注转换为 YOLO 格式
    YOLO 格式: class_id center_x center_y width height （全部归一化到 0~1）
    """
    points = shape['points']
    label = shape['label']

    if label not in CLASSES:
        print(f"  [警告] 未知类别: {label}，跳过")
        return None

    cls_id = CLASSES.index(label)

    # 计算边界框
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    # 转 YOLO 格式（中心坐标 + 宽高，归一化）
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    # 确保数值在合理范围内
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))

    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_json(json_path, output_img_dir, output_label_dir):
    """处理单个 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data['imageWidth']
    img_h = data['imageHeight']
    img_filename = data['imagePath']

    # 图片源路径
    img_src = os.path.join(os.path.dirname(json_path), img_filename)
    if not os.path.exists(img_src):
        # 尝试用 json 同名的 jpg
        img_src = json_path.replace('.json', '.jpg')
        if not os.path.exists(img_src):
            print(f"  [错误] 找不到图片: {img_filename}")
            return False

    # 复制图片
    img_dst = os.path.join(output_img_dir, img_filename)
    shutil.copy2(img_src, img_dst)

    # 生成标签文件
    label_lines = []
    for shape in data['shapes']:
        if shape['shape_type'] in ['rectangle', 'polygon']:
            line = convert_bbox_to_yolo(shape, img_w, img_h)
            if line:
                label_lines.append(line)

    # 写入 txt
    label_filename = os.path.splitext(img_filename)[0] + '.txt'
    label_path = os.path.join(output_label_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))

    return True


def create_data_yaml():
    """生成 data.yaml 配置文件"""
    yaml_content = f"""# YOLOv8 数据集配置
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val

nc: {len(CLASSES)}
names: {CLASSES}
"""
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"[生成] 配置文件: {yaml_path}")


def main():
    print("=" * 50)
    print("  Labelme → YOLO 格式转换工具")
    print("=" * 50)

    # 检查输入目录
    if not os.path.exists(LABELME_DIR):
        print(f"[错误] 输入目录不存在: {LABELME_DIR}")
        return

    # 创建输出目录结构
    dirs = [
        os.path.join(OUTPUT_DIR, 'images', 'train'),
        os.path.join(OUTPUT_DIR, 'images', 'val'),
        os.path.join(OUTPUT_DIR, 'labels', 'train'),
        os.path.join(OUTPUT_DIR, 'labels', 'val'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 收集所有 JSON 文件
    json_files = [
        f for f in os.listdir(LABELME_DIR)
        if f.lower().endswith('.json')
    ]

    if not json_files:
        print("[错误] 没有找到 JSON 标注文件")
        return

    print(f"[统计] 找到 {len(json_files)} 个标注文件")
    print(f"[类别] {CLASSES}")
    print(f"[划分] 训练集 {int((1-VAL_RATIO)*100)}% / 验证集 {int(VAL_RATIO*100)}%")
    print("-" * 50)

    # 随机划分
    random.seed(SEED)
    random.shuffle(json_files)
    val_count = int(len(json_files) * VAL_RATIO)
    val_files = json_files[:val_count]
    train_files = json_files[val_count:]

    # 处理训练集
    print("[处理] 训练集...")
    train_img_dir = os.path.join(OUTPUT_DIR, 'images', 'train')
    train_label_dir = os.path.join(OUTPUT_DIR, 'labels', 'train')
    success_train = 0
    for f in train_files:
        json_path = os.path.join(LABELME_DIR, f)
        if process_json(json_path, train_img_dir, train_label_dir):
            success_train += 1
    print(f"  训练集: {success_train}/{len(train_files)} 张")

    # 处理验证集
    print("[处理] 验证集...")
    val_img_dir = os.path.join(OUTPUT_DIR, 'images', 'val')
    val_label_dir = os.path.join(OUTPUT_DIR, 'labels', 'val')
    success_val = 0
    for f in val_files:
        json_path = os.path.join(LABELME_DIR, f)
        if process_json(json_path, val_img_dir, val_label_dir):
            success_val += 1
    print(f"  验证集: {success_val}/{len(val_files)} 张")

    # 生成 data.yaml
    create_data_yaml()

    print("-" * 50)
    print("[完成] 数据集转换成功！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  训练集: {success_train} 张")
    print(f"  验证集: {success_val} 张")
    print("\n下一步: 运行 03_训练代码/train_yolov8.py 开始训练")


if __name__ == "__main__":
    main()
