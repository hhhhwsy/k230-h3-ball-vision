"""
庐山派 K230 - YOLOv8 小铁球实时检测
功能：加载 kmodel 模型，实时检测摄像头画面中的小铁球
使用方法：
  1. 将 best_k230.kmodel 放到 SD 卡的 /sdcard/models/ 目录
  2. 将此文件拷贝到 K230 开发板
  3. 在 CanMV IDE 中打开并运行

依赖：
  - 嘉楠 CanMV 固件（支持 nncase_runtime）
  - 摄像头模块
  - LCD 屏幕（可选）
"""

import os
import sys
import time
import gc
import random
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import image

# ==================== 配置参数 ====================
# 模型路径（SD 卡上）
MODEL_PATH = "/sdcard/models/best_k230.kmodel"

# 类别名称
LABELS = ["ball"]

# 输入图像尺寸（必须和训练时一致！）
IMG_SIZE = 320

# 置信度阈值
CONF_THRESHOLD = 0.5

# NMS 非极大值抑制阈值
NMS_THRESHOLD = 0.3

# 最大检测数量
MAX_DETECT = 10

# 颜色配置
COLORS = [
    (255, 0, 0),    # 红
    (0, 255, 0),    # 绿
    (0, 0, 255),    # 蓝
    (255, 255, 0),  # 黄
]
# ==================================================


class YOLOv8Detector:
    """YOLOv8 目标检测器"""

    def __init__(self, model_path, img_size=320, conf_thres=0.5, nms_thres=0.3):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        # 加载 KPU 模型
        print("[模型] 加载中:", model_path)
        self.kpu = nn.kpu()
        self.kpu.load_kmodel(model_path)
        print("[模型] 加载完成")

        # 获取模型输入输出信息
        self.input_shape = self.kpu.inputs_info()[0].shape
        self.output_shape = self.kpu.outputs_info()[0].shape
        print("[模型] 输入 shape:", self.input_shape)
        print("[模型] 输出 shape:", self.output_shape)

    def preprocess(self, img):
        """
        图像预处理
        将摄像头图像转为模型输入格式
        """
        # 缩放到模型输入尺寸
        img_resized = img.resize(self.img_size, self.img_size)

        # 转 RGB565 → 字节数据
        img_bytes = img_resized.to_bytes()

        # 转为 numpy 数组，shape: (H, W, 3)
        img_array = np.array(img_bytes, dtype=np.uint8)
        img_array = img_array.reshape((self.img_size, self.img_size, 3))

        # 增加 batch 维度，shape: (1, H, W, 3)
        # 注意：K230 nncase 的输入布局是 NHWC
        img_array = img_array.reshape((1, self.img_size, self.img_size, 3))

        # 转 float32 并归一化到 0~1
        # 注意：这里的预处理必须和模型转换时一致！
        img_array = img_array.astype(np.float32) / 255.0

        return img_array

    def detect(self, img):
        """
        执行检测
        返回: [(class_id, confidence, x1, y1, x2, y2), ...]
        """
        # 预处理
        input_data = self.preprocess(img)

        # 设置输入
        self.kpu.set_input(0, input_data)

        # 推理
        self.kpu.run()

        # 获取输出
        output = self.kpu.get_output(0)
        output_data = output.to_numpy()

        # 后处理
        detections = self.postprocess(output_data)

        return detections

    def postprocess(self, output):
        """
        YOLOv8 后处理
        输出格式: [batch, num_boxes, 4 + num_classes]
        4 = (cx, cy, w, h) 边界框
        num_classes = 类别置信度
        """
        # output shape: (1, num_boxes, 4 + num_classes)
        # 去掉 batch 维度
        output = output[0]

        num_boxes = output.shape[0]
        num_classes = output.shape[1] - 4

        # 分离边界框和类别置信度
        boxes = output[:, :4]      # cx, cy, w, h
        scores = output[:, 4:]     # class scores

        # 获取每个框的最大类别分数和类别ID
        # （ulab numpy 功能有限，用循环实现）
        detections = []

        for i in range(num_boxes):
            # 找最大类别分数
            max_score = 0
            max_class = 0
            for c in range(num_classes):
                if scores[i][c] > max_score:
                    max_score = scores[i][c]
                    max_class = c

            # 过滤低置信度
            if max_score < self.conf_thres:
                continue

            # 边界框转换: cx, cy, w, h → x1, y1, x2, y2
            cx = boxes[i][0]
            cy = boxes[i][1]
            w = boxes[i][2]
            h = boxes[i][3]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            detections.append([max_class, max_score, x1, y1, x2, y2])

        # 按置信度排序
        detections.sort(key=lambda x: x[1], reverse=True)

        # NMS 非极大值抑制
        detections = self.nms(detections)

        return detections[:MAX_DETECT]

    def nms(self, detections):
        """
        非极大值抑制
        去掉重叠度高的重复框
        """
        if len(detections) == 0:
            return []

        keep = []

        while len(detections) > 0:
            # 取置信度最高的
            best = detections.pop(0)
            keep.append(best)

            # 和剩下的比较
            i = 0
            while i < len(detections):
                # 计算 IoU
                iou = self.calculate_iou(best[2:], detections[i][2:])
                if iou > self.nms_thres:
                    detections.pop(i)
                else:
                    i += 1

        return keep

    def calculate_iou(self, box1, box2):
        """计算两个框的 IoU（交并比）"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 交集
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        if x2 <= x1 or y2 <= y1:
            return 0

        intersection = (x2 - x1) * (y2 - y1)

        # 并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0

        return intersection / union


def draw_detections(img, detections, labels, img_size):
    """
    在图像上绘制检测结果
    """
    if not detections:
        return

    # 计算缩放比例（模型输入尺寸 → 实际显示尺寸）
    scale_x = img.width() / img_size
    scale_y = img.height() / img_size

    for det in detections:
        class_id, conf, x1, y1, x2, y2 = det

        # 坐标映射回原图尺寸
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        # 颜色
        color = COLORS[class_id % len(COLORS)]

        # 画框
        img.draw_rectangle(x1, y1, x2, y2, color=color)

        # 标签文字
        label = "%s %.1f%%" % (labels[class_id], conf * 100)
        img.draw_string(x1, max(0, y1 - 12), label, color=color)


def main():
    print("=" * 40)
    print("  庐山派 K230 - YOLOv8 小铁球检测")
    print("=" * 40)

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print("[错误] 找不到模型文件:", MODEL_PATH)
        print("请将 best_k230.kmodel 放到 SD 卡的 /sdcard/models/ 目录")
        return

    # 初始化摄像头
    print("[摄像头] 初始化...")
    sensor = Sensor()
    sensor.reset()
    sensor.set_pixformat(Sensor.RGB565)
    sensor.set_framesize(width=IMG_SIZE, height=IMG_SIZE)
    sensor.skip_frames(time=2000)
    print("[摄像头] 就绪: %dx%d" % (IMG_SIZE, IMG_SIZE))

    # 初始化 LCD
    print("[LCD] 初始化...")
    try:
        lcd.init()
        has_lcd = True
        print("[LCD] 就绪")
    except:
        has_lcd = False
        print("[LCD] 未检测到屏幕")

    # 初始化媒体管理器
    MediaManager.init()

    # 启动摄像头
    sensor.run(1)

    # 加载模型
    detector = YOLOv8Detector(
        model_path=MODEL_PATH,
        img_size=IMG_SIZE,
        conf_thres=CONF_THRESHOLD,
        nms_thres=NMS_THRESHOLD,
    )

    # 主循环
    clock = time.clock()
    print("\n[运行] 开始检测，按 Ctrl+C 退出")
    print("-" * 40)

    try:
        while True:
            clock.tick()

            # 采集图像
            img = sensor.snapshot()

            # 检测
            detections = detector.detect(img)

            # 画框
            if detections:
                draw_detections(img, detections, LABELS, IMG_SIZE)

            # 显示 FPS
            fps = clock.fps()
            img.draw_string(2, 2, "FPS: %.1f" % fps, color=(0, 255, 0))

            # 显示检测数量
            img.draw_string(2, 16, "Det: %d" % len(detections), color=(0, 255, 0))

            # LCD 显示
            if has_lcd:
                lcd.display(img)

            # 串口输出检测结果
            if detections:
                for det in detections:
                    class_id, conf, x1, y1, x2, y2 = det
                    print("[检测] %s (%.1f%%) 框: (%d,%d)-(%d,%d)" % (
                        LABELS[class_id], conf * 100,
                        int(x1), int(y1), int(x2), int(y2)
                    ))

            # 垃圾回收
            gc.collect()

    except KeyboardInterrupt:
        print("\n[退出] 检测停止")
    except Exception as e:
        print("[错误]", e)
    finally:
        sensor.run(0)
        MediaManager.deinit()
        print("[完成] 资源已释放")


if __name__ == "__main__":
    main()
