"""
K230 钢球位置识别与 MSPM0 串口发送

运行环境：
    CanMV K230 MicroPython（在 VS Code 的 CanMV 插件中直接运行）

识别策略：
    1. 只处理摆杆凹槽附近的窄ROI，排除车体、舵机、螺丝等干扰。
    2. 利用“白色水管较亮、钢球外圈和阴影较暗”的特点直接提取暗色区域。
    3. 使用小核闭运算连接被高光分开的钢球暗边，再开运算去除孤立小点。
    4. 用面积、尺寸、宽高比和凹槽中心线约束排除水管边缘、刻度和接缝。
    5. SEARCH不受旧坐标限制；TRACK利用上一帧位置和速度抑制错误跳变。
    6. 使用三帧中值滤波和一阶低通滤波，减小坐标抖动。
    7. 长期丢失清空旧位置；串口发送VALID=0，主控不得用旧位置闭环。

重要操作：
    程序不再采集空槽背景，启动预热完成后可以直接放置并移动钢球。
    ROI必须尽量只覆盖白色水管凹槽，不能包含大面积黑色车体。
"""

import gc
import os
import time

import cv2
from machine import FPIOA, UART
from media.display import *
from media.media import *
from media.sensor import *
from ulab import numpy as np


# =============================================================================
# 一、摄像头与显示配置
# =============================================================================

# 640x480兼顾识别精度与运行速度。
# 赛题摆杆长25cm，建议通过调整摄像头高度，让摆杆在画面里占450~540像素。
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 用户当前K230带800x480屏幕，因此默认使用ST7701。
# to_ide=True会把同一画面送到VS Code CanMV的图像窗口。
DISPLAY_DEVICE = Display.ST7701
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

# 640x480图像在800x480屏幕上水平居中，避免之前画面被裁掉。
DISPLAY_X = (DISPLAY_WIDTH - FRAME_WIDTH) // 2
DISPLAY_Y = 0


# =============================================================================
# 二、ROI与像素坐标标定
# =============================================================================

# 下面这组数值按照装车后的实际640x480画面重新测量：
# 白色水管在画面中大约位于Y=275~335，凹槽/钢球中心约为Y=305。
# 横向尽量覆盖当前画面中能够看到的水管，但左右仍各留20像素安全边界。
ROI_X = 80
ROI_Y = 270
ROI_W = 550
ROI_H = 55

# 凹槽中心线在整幅640x480图像中的Y坐标。
# 候选钢球圆心离这条线太远会被判为车体反光或背景干扰。
GROOVE_CENTER_Y = 297
GROOVE_Y_TOLERANCE = 18

# 三点像素标定：
# 手动把球依次放到-5cm、O点、+5cm，读取画面左上角的RAW_X，
# 然后将三个实测像素写到这里。
#
# 如果图像中+方向朝左，PIXEL_AT_POS_50MM可以小于PIXEL_AT_NEG_50MM，
# 换算公式仍然成立，不需要另加负号。
PIXEL_AT_NEG_50MM = 220
PIXEL_AT_ZERO_MM = 320
PIXEL_AT_POS_50MM = 420


# =============================================================================
# 三、白色水管灰度暗区域参数
# =============================================================================

# 摄像头启动后先等待曝光和白平衡基本稳定。
CAMERA_WARMUP_MS = 1500

# 灰度小于该值的像素进入暗色候选。
# 白色水管过曝时，钢球中心也可能很亮，但钢球暗边和下方阴影仍会被提取。
# 漏检钢球：每次增加5；水管刻度和接缝误检多：每次减小5。
DARK_BALL_THRESHOLD = 125

# 使用3x3小核先闭运算连接高光分裂的钢球，再开运算去除小点。
# 核太大会把钢球与凹槽刻度、边缘粘成一个长条区域。
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 1


# =============================================================================
# 四、钢球候选区域约束
# =============================================================================

# 以下默认值按“直径1cm钢球在画面中约20~30像素”设置。
# 摄像头高度改变后，优先调整EXPECTED_DIAMETER_PX、最小/最大尺寸和面积。
EXPECTED_DIAMETER_PX = 24

MIN_BLOB_AREA = 50
MAX_BLOB_AREA = 1000

MIN_BLOB_WIDTH = 10
MAX_BLOB_WIDTH = 40
MIN_BLOB_HEIGHT = 8
MAX_BLOB_HEIGHT = 40

# 钢球受凹槽遮挡、反光影响后不一定是完美圆，所以范围不能卡得太死。
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.00
MIN_FILL_RATIO = 0.16
MIN_CIRCULARITY = 0.10

# 当前先关闭严格圆度和填充率淘汰。
# 银色钢球有中心高光、凹槽遮挡，二值区域经常不是完整实心圆。
# 圆度和填充率仍参与评分，但不会在SEARCH阶段直接把钢球排除。
ENABLE_STRICT_SHAPE_FILTER = False

# 首次搜索时必须连续几帧在相近位置看到候选，才正式锁定。
ACQUIRE_CONFIRM_FRAMES = 3
ACQUIRE_MAX_STEP_PX = 35

# 锁定后的最大允许跳变。
# 钢球真实运动连续，超过该值通常是刻度、高光或车体反光。
MAX_TRACK_JUMP_PX = 65

# 连续丢失达到该帧数后，退出TRACK并回到全ROI搜索。
LOST_TO_SEARCH_FRAMES = 4

# 长期丢失后清空旧位置和旧速度。
# 这样画面不会一直显示一个已经失效的LAST_X，重新找球也从干净状态开始。
LONG_LOST_RESET_FRAMES = 10


# =============================================================================
# 五、位置与速度滤波参数
# =============================================================================

# 先对最近3个原始坐标取中值，再进行一阶低通。
MEDIAN_WINDOW = 3

# 越大越灵敏、延迟越小；越小越平滑、延迟越大。
# 滚球闭环不宜滤得过重，初始建议0.45~0.65。
POSITION_FILTER_ALPHA = 0.55

# 速度只用于预测候选位置，不直接发送给MSPM0。
VELOCITY_FILTER_ALPHA = 0.35

# 预测时间过长会把搜索窗口带偏，因此限制为最多0.15秒。
MAX_PREDICT_DT_S = 0.15


# =============================================================================
# 六、K230与MSPM0串口配置
# =============================================================================

# 保持和之前已经通信成功的接线一致：
#   K230 GPIO5  / UART2_TX  -> MSPM0 PA22 / UART2_RX
#   K230 GPIO6  / UART2_RX  <- MSPM0 PB15 / UART2_TX
#   K230 GND                  - MSPM0 GND
#
# 两块板必须共地；K230单独可靠供电，不要由MSPM0的3.3V脚给K230供电。
UART_TX_PIN = 5
UART_RX_PIN = 6
UART_ID = UART.UART2
UART_BAUDRATE = 115200

# 限制发送频率，避免无意义地占用主控串口中断。
UART_SEND_INTERVAL_MS = 40


# =============================================================================
# 七、调试输出配置
# =============================================================================

# True：显示候选框、ROI、状态和坐标，适合当前调试阶段。
# 若需要更高帧率，可改为 False。
DRAW_DEBUG_OVERLAY = True

# 必要时才打开原始候选框调试。默认False可以避免画面布满小框；
# RAW/PASS等拒绝统计仍会保留，最终钢球绿色框也始终保留。
#   黄色细框：二值图中的原始轮廓
#   蓝色框：通过面积、尺寸、宽高比和纵向位置过滤
#   绿色框：最终选择的钢球
DRAW_ALL_CANDIDATES = False
MAX_DEBUG_CANDIDATES = 20

# 每隔多少帧主动回收一次内存，避免长时间运行产生碎片。
GC_INTERVAL_FRAMES = 10


# =============================================================================
# 工具函数
# =============================================================================

def clamp(value, low, high):
    """把数值限制在[low, high]范围内。"""
    if value < low:
        return low
    if value > high:
        return high
    return value


def round_to_int(value):
    """不依赖CPython的round规则，实现对正负数均对称的四舍五入。"""
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


def median_of_values(values):
    """返回小列表的中值；本程序窗口固定为3，计算量很小。"""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def pixel_to_position_0p1mm(pixel_x):
    """
    将钢球横向像素换算为相对中心O的位置。

    返回单位：0.1mm
        +500 = +50.0mm = +5cm
        -500 = -50.0mm = -5cm

    主换算使用-5cm与+5cm两点，因此比例由100mm实际距离确定。
    PIXEL_AT_ZERO_MM主要用于在画面中画中心标记和检查安装偏差。
    """
    denominator = PIXEL_AT_POS_50MM - PIXEL_AT_NEG_50MM
    if denominator == 0:
        return 0

    position_mm = (
        -50.0
        + (float(pixel_x - PIXEL_AT_NEG_50MM) * 100.0 / float(denominator))
    )

    # 主控协议允许-125.0mm~+125.0mm，和25cm摆杆物理范围一致。
    position_0p1mm = round_to_int(position_mm * 10.0)
    return int(clamp(position_0p1mm, -1250, 1250))


def xor_checksum(payload):
    """
    计算ASCII载荷逐字节异或校验。

    注意：
        payload不包含开头的'$'，也不包含'*CS'和换行符。
    """
    checksum = 0
    for char in payload:
        checksum ^= ord(char)
    return checksum


def make_ball_packet(sequence, position_0p1mm, pixel_x, valid, quality):
    """
    生成与现有MSPM0 K230.c完全兼容的数据包。

    格式：
        $BALL,SEQ,POS10,PIXEL_X,VALID,QUALITY*CS\\r\\n

    字段：
        SEQ      0~9999循环帧序号
        POS10    相对O点位置，单位0.1mm
        PIXEL_X  滤波后的原始横向像素，便于标定
        VALID    1=本帧测量有效，0=丢球/尚未稳定锁定
        QUALITY  0~100识别质量
        CS       从字符B到星号前一字符的逐字节异或校验
    """
    payload = "BALL,%04d,%d,%d,%d,%d" % (
        sequence,
        int(position_0p1mm),
        int(pixel_x),
        int(valid),
        int(quality),
    )
    checksum = xor_checksum(payload)
    return "$" + payload + "*%02X\r\n" % checksum


def init_uart():
    """配置GPIO复用并初始化UART2为115200、8N1。"""
    fpioa = FPIOA()
    fpioa.set_function(UART_TX_PIN, FPIOA.UART2_TXD)
    fpioa.set_function(UART_RX_PIN, FPIOA.UART2_RXD)

    uart = UART(
        UART_ID,
        baudrate=UART_BAUDRATE,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
        timeout=0,
    )
    return uart


class UartCommandReceiver:
    """
    非阻塞接收MSPM0发来的简单命令。

    支持：
        PING,n  -> 回复PONG,n
        CALBG   -> 兼容旧主控命令，重置钢球跟踪状态

    命令处理不会使用等待循环，因此不会拖慢图像识别。
    """

    def __init__(self):
        self.line_buffer = ""

    def poll(self, uart):
        reset_requested = False
        data = uart.read()
        if not data:
            return False

        try:
            text = data.decode()
        except Exception:
            return False

        for char in text:
            if char == "\n":
                line = self.line_buffer.strip()
                self.line_buffer = ""

                if line.startswith("PING,"):
                    uart.write(("PONG," + line[5:] + "\r\n").encode())
                elif line == "CALBG":
                    reset_requested = True
            elif char != "\r":
                if len(self.line_buffer) < 63:
                    self.line_buffer += char
                else:
                    # 超长命令直接丢弃，等下一行重新同步。
                    self.line_buffer = ""

        return reset_requested


class BallTracker:
    """
    一维钢球跟踪器。

    SEARCH阶段：
        全ROI选形状最合理的候选，连续3帧位置相近后才锁定。

    TRACK阶段：
        根据上一帧滤波位置和速度预测本帧位置，优先选择预测点附近候选；
        超过最大跳变的候选不接受，避免坐标突然跳到刻度或反光点。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.locked = False
        self.filtered_x = None
        self.velocity_px_s = 0.0
        self.last_update_ms = None
        self.raw_history = []
        self.lost_count = 0
        self.acquire_count = 0
        self.acquire_x = None
        self.last_candidate = None

    def state_name(self):
        if self.locked:
            return "TRACK"
        return "SEARCH"

    def predicted_x(self, now_ms):
        # SEARCH阶段禁止继续使用旧位置预测。这样钢球被手动挪远或长期
        # 丢失后，旧LAST_X不会把真正的钢球候选永久排除在外。
        if (
            not self.locked
            or self.filtered_x is None
            or self.last_update_ms is None
        ):
            return None

        dt_s = time.ticks_diff(now_ms, self.last_update_ms) / 1000.0
        dt_s = clamp(dt_s, 0.0, MAX_PREDICT_DT_S)
        return self.filtered_x + self.velocity_px_s * dt_s

    def _record_miss(self):
        self.lost_count += 1
        self.last_candidate = None

        if self.lost_count >= LOST_TO_SEARCH_FRAMES:
            # 短暂丢失后退出TRACK，SEARCH重新扫描整个凹槽。
            self.locked = False
            self.acquire_count = 0
            self.acquire_x = None
            self.velocity_px_s = 0.0

        if self.lost_count >= LONG_LOST_RESET_FRAMES:
            # 长期丢失后彻底清空旧位置和滤波历史。串口仍会用主循环里
            # 保存的最后有效位置发送VALID=0，不会错误发送0位置。
            self.filtered_x = None
            self.last_update_ms = None
            self.raw_history = []

        return None

    def update(self, candidate, now_ms):
        """
        输入本帧最佳候选，返回跟踪结果字典或None。

        返回None表示本帧不能给主控提供有效测量。
        """
        if candidate is None:
            return self._record_miss()

        raw_x = float(candidate["cx"])

        if not self.locked:
            # 首次锁定要求连续帧位置相近，避免单帧反光误触发。
            if (
                self.acquire_x is None
                or abs(raw_x - self.acquire_x) > ACQUIRE_MAX_STEP_PX
            ):
                self.acquire_x = raw_x
                self.acquire_count = 1
            else:
                self.acquire_x = 0.6 * self.acquire_x + 0.4 * raw_x
                self.acquire_count += 1

            self.last_candidate = candidate
            if self.acquire_count < ACQUIRE_CONFIRM_FRAMES:
                return None

            self.locked = True
            self.filtered_x = raw_x
            self.velocity_px_s = 0.0
            self.last_update_ms = now_ms
            self.raw_history = [raw_x]
            self.lost_count = 0
        else:
            predicted = self.predicted_x(now_ms)
            if predicted is not None and abs(raw_x - predicted) > MAX_TRACK_JUMP_PX:
                return self._record_miss()

            self.raw_history.append(raw_x)
            if len(self.raw_history) > MEDIAN_WINDOW:
                self.raw_history.pop(0)

            median_x = median_of_values(self.raw_history)
            old_filtered_x = self.filtered_x

            self.filtered_x = (
                POSITION_FILTER_ALPHA * median_x
                + (1.0 - POSITION_FILTER_ALPHA) * old_filtered_x
            )

            dt_s = time.ticks_diff(now_ms, self.last_update_ms) / 1000.0
            if dt_s > 0.001:
                instant_velocity = (self.filtered_x - old_filtered_x) / dt_s
                self.velocity_px_s = (
                    VELOCITY_FILTER_ALPHA * instant_velocity
                    + (1.0 - VELOCITY_FILTER_ALPHA) * self.velocity_px_s
                )

            self.last_update_ms = now_ms
            self.lost_count = 0
            self.last_candidate = candidate

        return {
            "raw_x": int(raw_x),
            "filtered_x": int(self.filtered_x + 0.5),
            "quality": candidate["quality"],
            "candidate": candidate,
        }


def make_gray_roi(image_np):
    """
    将RGB888图像转换为灰度并截取ROI。

    CanMV的OpenCV模板中RGB888帧通过to_numpy_ref()零拷贝转成ulab数组。
    这里使用BGR2GRAY与官方K230 OpenCV摄像头模板保持一致。
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    roi = gray[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]
    return cv2.GaussianBlur(roi, (5, 5), 0)


def build_dark_ball_mask(gray_roi, kernel):
    """
    根据白色水管中的暗色钢球外圈生成二值图。

    钢球中心即使因为反光接近白色，外围轮廓和下方阴影通常仍明显偏暗。
    因此先做反向灰度阈值，再用小核闭运算连接高光造成的断裂，最后用
    开运算去掉孤立噪点。静态刻度和接缝不再依赖背景差分消除，而由后续
    的面积、尺寸、宽高比和纵向位置条件排除。
    """
    _, dark_binary = cv2.threshold(
        gray_roi,
        DARK_BALL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # 先闭运算把钢球被高光分开的区域连起来，再用一次开运算去除小噪点。
    closed = cv2.morphologyEx(
        dark_binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_CLOSE_ITERATIONS,
    )
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        kernel,
        iterations=MORPH_OPEN_ITERATIONS,
    )
    return opened


def find_best_candidate(mask, predicted_x, tracking):
    """
    在二值图中寻找最像钢球的候选，并返回调试统计。

    SEARCH阶段完全不使用LAST_X，只根据尺寸、纵向位置和宽高比重新找球。
    TRACK阶段才加入预测位置约束。圆度与填充率默认只参与评分、不做硬
    过滤，因为钢球高光、阴影和凹槽遮挡都可能破坏完整圆形。
    """
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best = None
    best_score = 1000000.0
    diagnostics = {
        "raw": 0,
        "passed": 0,
        "area": 0,
        "size": 0,
        "ratio": 0,
        "shape": 0,
        "y": 0,
        "jump": 0,
        "boxes": [],
    }

    for contour in contours:
        diagnostics["raw"] += 1

        x, y, w, h = cv2.boundingRect(contour)
        debug_box = None
        if (
            DRAW_ALL_CANDIDATES
            and len(diagnostics["boxes"]) < MAX_DEBUG_CANDIDATES
        ):
            debug_box = {
                "x": x + ROI_X,
                "y": y + ROI_Y,
                "w": w,
                "h": h,
                "passed": False,
            }
            diagnostics["boxes"].append(debug_box)

        area = float(cv2.contourArea(contour))
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            diagnostics["area"] += 1
            continue

        if (
            w < MIN_BLOB_WIDTH
            or w > MAX_BLOB_WIDTH
            or h < MIN_BLOB_HEIGHT
            or h > MAX_BLOB_HEIGHT
        ):
            diagnostics["size"] += 1
            continue

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            diagnostics["ratio"] += 1
            continue

        rectangle_area = float(w * h)
        fill_ratio = area / rectangle_area

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.01:
            diagnostics["shape"] += 1
            continue

        circularity = (4.0 * 3.1415926 * area) / (perimeter * perimeter)
        if ENABLE_STRICT_SHAPE_FILTER and (
            fill_ratio < MIN_FILL_RATIO
            or circularity < MIN_CIRCULARITY
        ):
            diagnostics["shape"] += 1
            continue

        # 轮廓坐标属于ROI，因此需要加ROI偏移得到整幅图坐标。
        center, radius = cv2.minEnclosingCircle(contour)
        cx = float(center[0]) + ROI_X
        cy = float(center[1]) + ROI_Y

        y_error = abs(cy - GROOVE_CENTER_Y)
        if y_error > GROOVE_Y_TOLERANCE:
            diagnostics["y"] += 1
            continue

        diameter = 2.0 * float(radius)
        size_error = abs(diameter - EXPECTED_DIAMETER_PX)
        circle_penalty = (1.0 - clamp(circularity, 0.0, 1.0)) * 20.0
        fill_penalty = abs(fill_ratio - 0.65) * 10.0

        motion_error = 0.0
        if tracking and predicted_x is not None:
            motion_error = abs(cx - predicted_x)
            if motion_error > MAX_TRACK_JUMP_PX:
                diagnostics["jump"] += 1
                continue

        diagnostics["passed"] += 1
        if debug_box is not None:
            debug_box["passed"] = True

        score = (
            4.0 * y_error
            + 1.0 * size_error
            + 0.40 * circle_penalty
            + 0.30 * fill_penalty
            + (2.0 * motion_error if tracking else 0.0)
        )

        if score < best_score:
            # 质量值只用于给主控判断测量可信度，不参与位置换算。
            quality = int(clamp(100.0 - 0.8 * score, 0.0, 100.0))
            best_score = score
            best = {
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "x": x + ROI_X,
                "y": y + ROI_Y,
                "w": w,
                "h": h,
                "area": area,
                "quality": quality,
                "score": score,
            }

    return best, diagnostics


def draw_common_overlay(image_np, status_text):
    """绘制ROI、凹槽中心线和三点标定线。"""
    cv2.rectangle(
        image_np,
        (ROI_X, ROI_Y),
        (ROI_X + ROI_W, ROI_Y + ROI_H),
        (0, 255, 0),
        2,
    )
    cv2.line(
        image_np,
        (ROI_X, GROOVE_CENTER_Y),
        (ROI_X + ROI_W, GROOVE_CENTER_Y),
        (255, 255, 0),
        1,
    )

    # -5cm、O、+5cm三条竖线便于直接观察标定是否正确。
    for x, color in (
        (PIXEL_AT_NEG_50MM, (255, 0, 255)),
        (PIXEL_AT_ZERO_MM, (0, 255, 255)),
        (PIXEL_AT_POS_50MM, (255, 0, 255)),
    ):
        cv2.line(
            image_np,
            (x, ROI_Y),
            (x, ROI_Y + ROI_H),
            color,
            1,
        )

    cv2.putText(
        image_np,
        status_text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )


def show_timed_message(sensor, text, delay_ms):
    """在等待阶段持续显示实时画面，便于确认ROI确实覆盖整个凹槽。"""
    start_ms = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start_ms) < delay_ms:
        os.exitpoint()
        image = sensor.snapshot()
        image_np = image.to_numpy_ref()
        draw_common_overlay(image_np, text)
        Display.show_image(image, x=DISPLAY_X, y=DISPLAY_Y)
        time.sleep_ms(20)


def draw_tracking_overlay(
    image_np,
    tracker,
    result,
    packet_valid,
    position_0p1mm,
    diagnostics,
):
    """把调试信息叠加到原图；不支持中文字体，因此屏幕文字使用英文。"""
    draw_common_overlay(
        image_np,
        "%s  VALID:%d" % (tracker.state_name(), packet_valid),
    )

    # 黄色框：二值图中找到的原始轮廓；蓝色框：通过基本过滤的候选。
    # 最终锁定的钢球仍用绿色框和圆显示。
    if DRAW_ALL_CANDIDATES:
        for box in diagnostics["boxes"]:
            box_color = (255, 0, 0) if box["passed"] else (0, 255, 255)
            cv2.rectangle(
                image_np,
                (box["x"], box["y"]),
                (box["x"] + box["w"], box["y"] + box["h"]),
                box_color,
                1,
            )

    candidate = tracker.last_candidate
    if candidate is not None:
        color = (0, 255, 0) if packet_valid else (255, 255, 0)
        cv2.rectangle(
            image_np,
            (candidate["x"], candidate["y"]),
            (
                candidate["x"] + candidate["w"],
                candidate["y"] + candidate["h"],
            ),
            color,
            2,
        )
        cv2.circle(
            image_np,
            (int(candidate["cx"]), int(candidate["cy"])),
            max(3, int(candidate["radius"])),
            color,
            2,
        )

    predicted = tracker.predicted_x(time.ticks_ms())
    if predicted is not None:
        predicted_int = int(clamp(predicted, ROI_X, ROI_X + ROI_W))
        cv2.drawMarker(
            image_np,
            (predicted_int, GROOVE_CENTER_Y),
            (255, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=1,
        )

    if result is not None:
        text = "RAW:%d FIL:%d POS:%+.1fmm Q:%d" % (
            result["raw_x"],
            result["filtered_x"],
            position_0p1mm / 10.0,
            result["quality"],
        )
        color = (0, 255, 0)
    else:
        held_x = -1 if tracker.filtered_x is None else int(tracker.filtered_x)
        text = "NO VALID BALL  LAST_X:%d LOST:%d" % (
            held_x,
            tracker.lost_count,
        )
        color = (0, 0, 255)

    cv2.putText(
        image_np,
        text,
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    reject_text = "RAW:%d PASS:%d A:%d S:%d R:%d SH:%d Y:%d J:%d" % (
        diagnostics["raw"],
        diagnostics["passed"],
        diagnostics["area"],
        diagnostics["size"],
        diagnostics["ratio"],
        diagnostics["shape"],
        diagnostics["y"],
        diagnostics["jump"],
    )
    cv2.putText(
        image_np,
        reject_text,
        (12, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
    )


# =============================================================================
# 主程序
# =============================================================================

def main():
    sensor = None
    uart = None

    try:
        # 1. 初始化串口。图像算法即使暂时找不到球，也会定期发送VALID=0。
        uart = init_uart()

        # 2. 初始化摄像头为RGB888；cv2通过to_numpy_ref()处理共享图像内存。
        sensor = Sensor(width=1280, height=960, fps=60)
        sensor.reset()
        sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)
        sensor.set_pixformat(Sensor.RGB888)

        # 3. 800x480实体屏与VS Code IDE预览同时启用。
        Display.init(
            DISPLAY_DEVICE,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            to_ide=True,
        )
        MediaManager.init()
        sensor.run()

        print("H3 BALL TRACKER START")
        print("UART2: GPIO5 TX, GPIO6 RX, 115200 8N1")
        print("WHITE PIPE DARK-BLOB MODE")

        show_timed_message(sensor, "CAMERA WARMUP", CAMERA_WARMUP_MS)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        )

        tracker = BallTracker()
        command_receiver = UartCommandReceiver()

        sequence = 0
        last_send_ms = time.ticks_ms()
        frame_counter = 0

        # 丢球时仍保存最后一个滤波位置，但必须把VALID置0。
        last_pixel_x = PIXEL_AT_ZERO_MM
        last_position_0p1mm = 0

        while True:
            os.exitpoint()
            now_ms = time.ticks_ms()

            # 兼容旧主控的CALBG命令：灰度暗块模式不采背景，只重置跟踪器。
            if command_receiver.poll(uart):
                print("CALBG RECEIVED -> TRACKER RESET")
                tracker.reset()
                last_pixel_x = PIXEL_AT_ZERO_MM
                last_position_0p1mm = 0

            image = sensor.snapshot()
            image_np = image.to_numpy_ref()

            gray_roi = make_gray_roi(image_np)
            foreground_mask = build_dark_ball_mask(
                gray_roi,
                kernel,
            )

            predicted_x = tracker.predicted_x(now_ms)
            candidate, diagnostics = find_best_candidate(
                foreground_mask,
                predicted_x,
                tracker.locked,
            )
            result = tracker.update(candidate, now_ms)

            if result is not None:
                packet_valid = 1
                last_pixel_x = result["filtered_x"]
                last_position_0p1mm = pixel_to_position_0p1mm(last_pixel_x)
                quality = result["quality"]
            else:
                # 没有本帧有效测量时绝不能假装位置是0。
                packet_valid = 0
                quality = 0

            # 串口按固定最短间隔发送，不使用等待ACK的阻塞循环。
            if time.ticks_diff(now_ms, last_send_ms) >= UART_SEND_INTERVAL_MS:
                packet = make_ball_packet(
                    sequence,
                    last_position_0p1mm,
                    last_pixel_x,
                    packet_valid,
                    quality,
                )
                uart.write(packet.encode())

                # VS Code终端中每10帧打印一次，既能观察又不会刷屏过快。
                if (sequence % 10) == 0:
                    print(packet.strip())
                    print(
                        "DBG RAW:%d PASS:%d AREA:%d SIZE:%d "
                        "RATIO:%d SHAPE:%d Y:%d JUMP:%d"
                        % (
                            diagnostics["raw"],
                            diagnostics["passed"],
                            diagnostics["area"],
                            diagnostics["size"],
                            diagnostics["ratio"],
                            diagnostics["shape"],
                            diagnostics["y"],
                            diagnostics["jump"],
                        )
                    )

                sequence = (sequence + 1) % 10000
                last_send_ms = now_ms

            if DRAW_DEBUG_OVERLAY:
                draw_tracking_overlay(
                    image_np,
                    tracker,
                    result,
                    packet_valid,
                    last_position_0p1mm,
                    diagnostics,
                )

            Display.show_image(image, x=DISPLAY_X, y=DISPLAY_Y)

            frame_counter += 1
            if (frame_counter % GC_INTERVAL_FRAMES) == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("H3 BALL TRACKER STOPPED BY USER")
    except Exception as error:
        # 板端发生异常时打印原因，便于在VS Code CanMV终端直接定位。
        print("H3 BALL TRACKER ERROR:", error)
        raise
    finally:
        # 严格按Sensor -> Display -> MediaManager顺序释放媒体资源。
        if isinstance(sensor, Sensor):
            sensor.stop()

        Display.deinit()
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        MediaManager.deinit()

        if uart is not None:
            uart.deinit()


if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    main()
