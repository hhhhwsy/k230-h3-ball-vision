# 立创·庐山派-K230-CanMV开发板资料与相关扩展板软硬件资料官网全部开源
# 开发板官网：www.lckfb.com
# 技术支持常驻论坛，任何技术问题欢迎随时交流学习
# 立创论坛：www.jlc-bbs.com/lckfb
# 关注bilibili账号：【立创开发板】，掌握我们的最新动态！
# 不靠卖板赚钱，以培养中国工程师为己任
# 编写者：LCKFB-AI-Asst

"""
钢珠实时检测程序（YOLOv8 + PipeLine）

功能说明：
  摄像头实时采集画面 → AI 检测钢珠 → LCD 显示检测框 + 最佳目标中心点
  终端每 10 帧打印一次检测结果（数量、坐标、置信度）

运行要求：
  - 将本文件和 steel_ball_yolov8n_320.kmodel 放到 /sdcard/sdcardsteel_ball/ 目录
  - 如需开机自启，复制到 /sdcard/main.py
"""
from media.sensor import Sensor        # 新增：适配v1.2.2固件必须导入
from libs.PipeLine import PipeLine      # 摄像头+显示的封装管线
from libs.YOLO import YOLOv8            # YOLOv8 目标检测推理
from libs.Utils import ScopedTiming     # 代码段计时（调试用）
import gc                               # 垃圾回收，防止内存溢出


# ============================================================
#  ↓↓↓                 用户可修改参数区                ↓↓↓
# ============================================================

# ---------- 模型配置 ----------
# 模型文件路径（板端路径，必须和 kmodel 文件实际存放位置一致）
MODEL_PATH = "/sdcard/sdcardsteel_ball/steel_ball_yolov8n_320.kmodel"
LABELS = ["steel_ball"]                  # 检测类别名称列表，可改为 ["ball","cube"] 等多类
MODEL_INPUT_SIZE = [320, 320]            # 模型输入尺寸 [宽, 高]，需和 kmodel 转换时一致

# ---------- 摄像头 ----------
SENSOR_ID = 2                            # CSI 接口编号：0/1/2，庐山派默认 CSI2
RGB888P_SIZE = [640, 360]                # AI 推理用帧分辨率 [宽, 高]，越小越快但精度越低

# ---------- 画面方向 ----------
# 四种组合对应不同旋转效果：
#   H_MIRROR=False, V_FLIP=False  → 原始方向
#   H_MIRROR=True,  V_FLIP=False  → 左右镜像（自拍效果）
#   H_MIRROR=False, V_FLIP=True   → 上下颠倒
#   H_MIRROR=True,  V_FLIP=True   → 旋转 180°
H_MIRROR = False                         # 水平镜像：True=左右对调 / False=原始
V_FLIP = False                           # 垂直翻转：True=上下颠倒 / False=原始

# ---------- 检测参数 ----------
CONFIDENCE_THRESHOLD = 0.65             # 置信度阈值 0~1，低于此值的检测框被丢弃（调高=更少误检）
NMS_THRESHOLD = 0.45                    # 非极大值抑制阈值 0~1，用于去除重叠框（越小去重越狠）
MAX_BOXES = 50                          # 每帧最多保留的检测框数量

# ---------- 终端输出 ----------
PRINT_EVERY_N_FRAMES = 10               # 每隔 N 帧打印一次检测结果（0=不打印）

# ============================================================
#  ↑↑↑                 用户可修改参数区                ↑↑↑
# ============================================================


def add_status_overlay(result, osd_image, frame_index):
    """
    在屏幕 OSD 叠加层上绘制检测结果信息：
      - 左上角：检测到的钢珠数量（绿色文字）
      - 置信度最高的钢珠：黄色十字 + 中心坐标
    """
    count = 0
    best_score = -1.0                    # 记录最高置信度
    best_center = None                   # 记录最高置信度目标的中心点 (x, y)

    # --- 遍历所有检测框，找置信度最高的那个 ---
    if result and len(result[0]) > 0:
        count = len(result[0])           # 本帧检测到的钢珠总数
        for index in range(count):
            x, y, width, height = result[0][index]      # 检测框：左上角坐标 + 宽高
            score = float(result[2][index])              # 该框的置信度分数
            if score > best_score:
                best_score = score
                best_center = (
                    int(round(x + width / 2)),            # 框中心 X = 左上角 + 宽/2
                    int(round(y + height / 2)),            # 框中心 Y = 左上角 + 高/2
                )

    # --- 左上角：钢珠数量 ---
    osd_image.draw_string_advanced(
        5, 5,                                # 坐标 (x=5, y=5)
        24,                                  # 字号
        "steel_ball: %d" % count,            # 显示文字
        color=(0, 255, 0)                    # 绿色
    )

    # --- 最高置信度目标：十字 + 坐标 ---
    if best_center is not None:
        center_x, center_y = best_center
        osd_image.draw_cross(
            center_x, center_y,              # 十字中心位置
            color=(255, 255, 0),             # 黄色
            size=12,                         # 十字臂长（像素）
            thickness=3,                     # 线条粗细
        )
        osd_image.draw_string_advanced(
            5, 34, 20,                       # (x=5, y=34) 字号=20
            "best center: %d,%d" % (center_x, center_y),
            color=(255, 255, 0),             # 黄色
        )

    # --- 终端打印（每 PRINT_EVERY_N_FRAMES 帧打印一次）---
    if frame_index % PRINT_EVERY_N_FRAMES == 0:
        if best_center is None:
            print("[steel_ball] count=0")
        else:
            print(
                "[steel_ball] count=%d best_center=(%d,%d) score=%.3f"
                % (count, best_center[0], best_center[1], best_score)
            )


def main():
    pipeline = None                        # PipeLine 实例
    detector = None                        # YOLOv8 检测器实例

    try:
        # ===== 第 1 步：初始化摄像头 + LCD 显示 =====
        # PipeLine 自动完成：Sensor 初始化 → Display 初始化 → 画面绑定
        pipeline = PipeLine(rgb888p_size=RGB888P_SIZE, display_mode="lcd")
        # ==========【核心修改】适配v1.2.2固件 ==========
        sensor = Sensor(id=SENSOR_ID)
        pipeline.create(
            sensor=sensor,
            hmirror=H_MIRROR,
            vflip=V_FLIP,
        )
        display_size = pipeline.get_display_size()     # 获取实际屏幕分辨率
        print("[steel_ball] 摄像头已启动, 显示: %dx%d" % (display_size[0], display_size[1]))

        # ===== 第 2 步：初始化 YOLOv8 检测器 =====
        detector = YOLOv8(
            task_type="detect",            # 任务类型：detect(检测) / classify(分类) / segment(分割)
            mode="video",                  # 模式：video(连续帧) / image(单张)
            kmodel_path=MODEL_PATH,        # kmodel 模型文件路径
            labels=LABELS,                 # 类别名列表
            rgb888p_size=RGB888P_SIZE,     # AI 推理帧分辨率（输入给模型前会用这个尺寸做预处理）
            model_input_size=MODEL_INPUT_SIZE,  # 模型实际输入尺寸
            display_size=display_size,     # 屏幕分辨率（用于缩放检测坐标）
            conf_thresh=CONFIDENCE_THRESHOLD,    # 置信度阈值
            nms_thresh=NMS_THRESHOLD,            # NMS 去重阈值
            max_boxes_num=MAX_BOXES,             # 最大检测框数
            debug_mode=0,                  # 0=关闭调试 / 1=开启计时
        )
        detector.config_preprocess()       # 配置模型预处理（必须调用）
        print("[steel_ball] YOLOv8 已就绪")

        # ===== 第 3 步：推理主循环 =====
        frame_index = 0                    # 帧计数器（仅用于终端打印频率判断）
        while True:
            with ScopedTiming("total", 1):  # 计时（debug_mode=0 时静默）
                # ① 从摄像头获取一帧图像（RGBP888 格式的 ulab numpy 数组）
                frame = pipeline.get_frame()

                # ② YOLOv8 推理，返回检测结果
                #    result 结构：[boxes坐标, classes类别ID, scores置信度]
                result = detector.run(frame)

                # ③ 在 OSD 图层上绘制检测框
                detector.draw_result(result, pipeline.osd_img)

                # ④ 叠加自定义状态信息（数量 + 最佳中心点）
                add_status_overlay(result, pipeline.osd_img, frame_index)

                # ⑤ 将 OSD 图层刷新到 LCD 屏幕
                pipeline.show_image()

                frame_index += 1
                gc.collect()               # 手动回收内存，防止长时间运行导致 OOM

    except KeyboardInterrupt:
        # Ctrl+C 或 IDE 停止按钮
        print("[steel_ball] 用户停止")
    except BaseException as error:
        # 其他异常（如模型文件不存在、内存不足等）
        print("[steel_ball] error:", error)
        raise                              # 抛出到终端显示完整错误栈
    finally:
        # ===== 第 4 步：释放资源（无论是否异常都会执行）=====
        if detector is not None:
            try:
                detector.deinit()          # 释放 YOLO 推理资源
            except BaseException as e:
                print("[steel_ball] detector cleanup:", e)
        if pipeline is not None and getattr(pipeline, "sensor", None) is not None:
            try:
                pipeline.destroy()         # 停止摄像头 + 关闭显示 + 释放媒体缓冲区
            except BaseException as e:
                print("[steel_ball] pipeline cleanup:", e)
        gc.collect()                       # 最后一次内存回收


# ===== 程序入口 =====
main()
