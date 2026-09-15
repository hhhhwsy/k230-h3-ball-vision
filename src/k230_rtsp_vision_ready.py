# CanMV K230 v1.8+ - 双通道视觉预留型 H.264 RTSP 图传
#
# 通道 0：1280x720 YUV420SP -> VENC H.264 -> RTSP -> OBS
# 通道 1：640x360 RGB565，预留给后续传统视觉识别钢球
#
# 本程序只负责图传与视觉取帧，不在 K230 上保存视频；录像由 OBS 手动完成。

import gc
import os
import time
import _thread
import uctypes

import multimedia as mm
import network
from machine import FPIOA, Pin
from media.media import *
from media.sensor import *
from media.vencoder import *


# ======================== 必须检查的配置 ========================

WIFI_SSID = "请改成你的热点名称"
WIFI_PASSWORD = "请改成你的热点密码"

RTSP_PORT = 8554
RTSP_SESSION = "ball"

# 传感器工作模式固定为 1080p30，再由硬件输出所需的两个通道。
# 这样可避免部分 GC2093/OV5647 在低分辨率模式下自动切到 60/90 fps。
SENSOR_MODE_WIDTH = 1920
SENSOR_MODE_HEIGHT = 1080
SENSOR_MODE_FPS = 30

# OBS 图传参数：俯视完整管道时推荐 720p30。
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
VIDEO_BIT_RATE_KBPS = 2000
VIDEO_GOP = 30
VENC_OUTPUT_BUFFERS = 8

# 视觉通道默认只配置、不取帧，因此现在几乎不占 Python CPU。
# 后续实现 traditional_vision_step() 后再把 PROCESS_VISION 改为 True。
ENABLE_VISION_CHANNEL = True
PROCESS_VISION = False
VISION_WIDTH = 640
VISION_HEIGHT = 360
VISION_INTERVAL_MS = 33
VISION_SNAPSHOT_TIMEOUT_MS = 100

# 0 表示使用自动曝光。固定灯光搭建完成后，可尝试 6000~10000 us
# 以减少钢球运动拖影；具体值必须按现场亮度实测。
MANUAL_EXPOSURE_US = 0

# 官方建议的 Wi-Fi 稳定等待：DHCP 获得 IP 后继续观察 8 秒。
WIFI_CONNECT_TIMEOUT_S = 25
WIFI_STABILIZE_S = 8
RECONNECT_DELAY_MS = 2000

# 优先用物理地址发送，减少每帧 bytes 拷贝；若固件不提供该接口会自动回退。
# 如果实测出现花屏，可临时设为 False 对比验证。
PREFER_ZERO_COPY = True
STREAM_GET_TIMEOUT_MS = 500
MAX_CONSECUTIVE_STREAM_FAILURES = 6
STATUS_INTERVAL_MS = 10000
GC_INTERVAL_FRAMES = 150

# 立创·庐山派 K230-CanMV 标准版：USER=GPIO53，按下为高电平。
# 不需要 USER 键时设为 False；不同板型请先核对原理图再修改 GPIO。
ENABLE_USER_BUTTON = True
USER_BUTTON_GPIO = 53
USER_BUTTON_LONG_PRESS_MS = 1000


def print_exception_safe(prefix, exc):
    """兼容没有 sys.print_exception() 的 CanMV 固件。"""
    print(prefix, exc)


def validate_config():
    """尽早发现会导致编码器或网络异常的配置。"""
    if not WIFI_SSID or WIFI_SSID.startswith("请改成"):
        raise ValueError("请先修改 WIFI_SSID 和 WIFI_PASSWORD")
    if WIFI_PASSWORD is None:
        raise ValueError("WIFI_PASSWORD 不能为 None")

    if VIDEO_WIDTH < 320 or VIDEO_HEIGHT < 240:
        raise ValueError("图传分辨率过低")
    if VIDEO_WIDTH % 16 != 0:
        raise ValueError("VIDEO_WIDTH 必须是 16 的倍数")
    if VIDEO_HEIGHT % 2 != 0:
        raise ValueError("VIDEO_HEIGHT 必须是偶数")
    if VIDEO_FPS < 5 or VIDEO_FPS > SENSOR_MODE_FPS:
        raise ValueError("VIDEO_FPS 必须在 5~SENSOR_MODE_FPS 之间")
    if VIDEO_BIT_RATE_KBPS < 100 or VIDEO_BIT_RATE_KBPS > 20000:
        raise ValueError("VIDEO_BIT_RATE_KBPS 必须在 100~20000 之间")
    if VIDEO_GOP < 1:
        raise ValueError("VIDEO_GOP 必须大于 0")

    if ENABLE_VISION_CHANNEL:
        if VISION_WIDTH % 16 != 0 or VISION_HEIGHT % 2 != 0:
            raise ValueError("视觉通道宽度须为 16 的倍数，高度须为偶数")
    elif PROCESS_VISION:
        raise ValueError("PROCESS_VISION=True 时必须启用视觉通道")


def wifi_has_ip(sta):
    if sta is None:
        return False

    try:
        if hasattr(sta, "isconnected") and not sta.isconnected():
            return False
        ip_address = sta.ifconfig()[0]
        return bool(ip_address) and ip_address != "0.0.0.0"
    except Exception:
        return False


def connect_wifi(ssid, password, timeout_s=WIFI_CONNECT_TIMEOUT_S):
    """连接 2.4 GHz Wi-Fi，并等待 DHCP 真正获得 IP。"""
    sta_interface = getattr(network, "STA_IF", 0)
    sta = network.WLAN(sta_interface)

    try:
        sta.config(auto_reconnect=True)
    except Exception:
        pass

    if wifi_has_ip(sta):
        print("Wi-Fi 已连接：", sta.ifconfig())
        return sta

    print("正在连接 Wi-Fi：", ssid)
    sta.connect(ssid, password)
    start_ms = time.ticks_ms()

    while not wifi_has_ip(sta):
        if time.ticks_diff(time.ticks_ms(), start_ms) >= timeout_s * 1000:
            raise RuntimeError(
                "Wi-Fi 获取 IP 超时；请检查 2.4 GHz 热点、信号和密码"
            )
        os.exitpoint()
        time.sleep_ms(200)

    print("Wi-Fi 连接成功：", sta.ifconfig())
    try:
        print("Wi-Fi 状态：", sta.status())
    except Exception:
        pass
    return sta


def wait_wifi_stable(sta, delay_s=WIFI_STABILIZE_S):
    """获得 IP 后继续等待 8 秒，并持续检查掉线和 IP 变化。"""
    initial_ip = sta.ifconfig()[0]
    start_ms = time.ticks_ms()
    last_printed_second = -1

    while True:
        elapsed_ms = time.ticks_diff(time.ticks_ms(), start_ms)
        if elapsed_ms >= delay_s * 1000:
            break

        os.exitpoint()
        if not wifi_has_ip(sta):
            raise RuntimeError("Wi-Fi 稳定等待期间掉线")
        if sta.ifconfig()[0] != initial_ip:
            raise RuntimeError("Wi-Fi 稳定等待期间 IP 发生变化")

        elapsed_second = elapsed_ms // 1000
        if elapsed_second != last_printed_second:
            remaining = delay_s - elapsed_second
            print("Wi-Fi 稳定等待：还剩 %d 秒" % remaining)
            last_printed_second = elapsed_second

        time.sleep_ms(100)

    print("Wi-Fi 已稳定，开始启动摄像头和 RTSP")


class UserButton:
    """USER 键长按检测，避免机械抖动导致误触发。"""

    def __init__(self, gpio=USER_BUTTON_GPIO,
                 long_press_ms=USER_BUTTON_LONG_PRESS_MS):
        self.long_press_ms = long_press_ms
        self.press_start_ms = None
        self.triggered = False

        fpioa = FPIOA()
        gpio_function = getattr(FPIOA, "GPIO%d" % gpio)
        fpioa.set_function(
            gpio,
            gpio_function,
            ie=1,
            oe=0,
            pu=0,
            pd=1
        )
        self.pin = Pin(gpio, Pin.IN, pull=Pin.PULL_DOWN)

    def poll(self):
        now_ms = time.ticks_ms()
        pressed = self.pin.value() != 0

        if not pressed:
            self.press_start_ms = None
            self.triggered = False
            return False

        if self.press_start_ms is None:
            self.press_start_ms = now_ms
            return False

        if (
            not self.triggered and
            time.ticks_diff(now_ms, self.press_start_ms) >= self.long_press_ms
        ):
            self.triggered = True
            return True

        return False


def create_user_button():
    if not ENABLE_USER_BUTTON:
        return None

    try:
        button = UserButton()
        print(
            "USER 键已启用：GPIO%d，长按 %.1f 秒停止图传" %
            (USER_BUTTON_GPIO, USER_BUTTON_LONG_PRESS_MS / 1000.0)
        )
        return button
    except Exception as exc:
        print_exception_safe("警告：USER 键初始化失败，仍可通过 IDE 停止：", exc)
        return None


def traditional_vision_step(image):
    """
    后续传统视觉算法入口。

    建议流程：固定管道 ROI -> 灰度/颜色阈值 -> 连通域或圆检测 ->
    计算钢球中心 x -> 像素到厘米标定。这里不要保存或长期持有 image。

    返回值可自行定义，例如钢球横坐标；当前返回 None，不执行识别。
    """
    return None


class VisionReadyRtspServer:
    """CanMV v1.8+ Sensor -> VENC -> RTSP 双通道图传服务。"""

    def __init__(self):
        self.sensor = None
        self.encoder = None
        self.link = None
        self.rtspserver = mm.rtsp_server()

        self.start_stream = False
        self.thread_finished = True
        self.last_error = None

        self.encoder_configured = False
        self.encoder_created = False
        self.encoder_started = False
        self.sensor_started = False
        self.rtsp_init_attempted = False
        self.rtsp_initialized = False
        self.rtsp_session_created = False
        self.rtsp_started = False

    def _configure_sensor(self):
        self.sensor = Sensor(
            width=SENSOR_MODE_WIDTH,
            height=SENSOR_MODE_HEIGHT,
            fps=SENSOR_MODE_FPS
        )
        self.sensor.reset()

        # 通道 0 只供硬件编码器使用，Python 不搬运原始 720p 图像。
        self.sensor.set_framesize(
            chn=CAM_CHN_ID_0,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
            alignment=12
        )
        self.sensor.set_pixformat(
            Sensor.YUV420SP,
            chn=CAM_CHN_ID_0
        )

        if ENABLE_VISION_CHANNEL:
            self.sensor.set_framesize(
                chn=CAM_CHN_ID_1,
                width=VISION_WIDTH,
                height=VISION_HEIGHT,
                alignment=12
            )
            # RGB565 适合后续传统阈值、连通域和画线调试。
            self.sensor.set_pixformat(
                Sensor.RGB565,
                chn=CAM_CHN_ID_1
            )

        if MANUAL_EXPOSURE_US > 0:
            try:
                self.sensor.auto_exposure(False)
            except Exception as exc:
                print_exception_safe("提示：关闭自动曝光失败，将继续自动曝光：", exc)

    def _configure_encoder(self):
        self.encoder = Encoder()
        self.encoder.SetOutBufs(
            VENC_OUTPUT_BUFFERS,
            VIDEO_WIDTH,
            VIDEO_HEIGHT
        )
        self.encoder_configured = True

        # v1.8 参数顺序：payload, profile, width, height,
        # bit_rate, gop_len, src_frame_rate, dst_frame_rate。
        chn_attr = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H264,
            self.encoder.H264_PROFILE_BASELINE,
            VIDEO_WIDTH,
            VIDEO_HEIGHT,
            VIDEO_BIT_RATE_KBPS,
            VIDEO_GOP,
            VIDEO_FPS,
            VIDEO_FPS
        )
        self.encoder.Create(chn_attr)
        self.encoder_created = True

        # v1.8 由 Create() 自动申请通道，必须使用 encoder.chn。
        self.link = MediaManager.link(
            self.sensor.bind_info(chn=CAM_CHN_ID_0)["src"],
            (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, self.encoder.chn)
        )

    def _configure_rtsp(self):
        self.rtsp_init_attempted = True
        ret = self.rtspserver.rtspserver_init(RTSP_PORT)
        if ret not in (None, 0):
            raise OSError("初始化 RTSP 端口失败：", ret)
        self.rtsp_initialized = True

        ret = self.rtspserver.rtspserver_createsession(
            RTSP_SESSION,
            mm.multi_media_type.media_h264,
            False
        )
        if ret not in (None, 0):
            raise OSError("创建 RTSP 会话失败：", ret)
        self.rtsp_session_created = True

        self.rtspserver.rtspserver_start()
        self.rtsp_started = True

    def start(self):
        self.last_error = None
        print("正在初始化双通道摄像头、H.264 编码器和 RTSP……")

        self._configure_sensor()
        self._configure_encoder()
        self._configure_rtsp()

        self.encoder.Start()
        self.encoder_started = True
        self.sensor.run()
        self.sensor_started = True

        if MANUAL_EXPOSURE_US > 0:
            try:
                self.sensor.exposure(MANUAL_EXPOSURE_US)
                print("手动曝光：%d us" % MANUAL_EXPOSURE_US)
            except Exception as exc:
                print_exception_safe("提示：设置手动曝光失败：", exc)

        self.start_stream = True
        self.thread_finished = False
        _thread.start_new_thread(self._stream_loop, ())

    def get_url(self):
        return self.rtspserver.rtspserver_getrtspurl(RTSP_SESSION)

    def snapshot_for_vision(self):
        if not ENABLE_VISION_CHANNEL:
            raise RuntimeError("视觉通道未启用")
        if not self.sensor_started:
            raise RuntimeError("摄像头尚未启动")
        return self.sensor.snapshot(
            chn=CAM_CHN_ID_1,
            timeout=VISION_SNAPSHOT_TIMEOUT_MS,
            dump_frame=False
        )

    @staticmethod
    def _pack_timestamp(stream_data, pack_idx):
        pts_array = getattr(stream_data, "pts", None)
        if pts_array is not None:
            pts = pts_array[pack_idx]
            if pts is not None and pts > 0:
                return pts
        return time.ticks_ms()

    def _send_pack(self, stream_data, pack_idx):
        size = stream_data.data_size[pack_idx]
        if size <= 0:
            return 0

        timestamp = self._pack_timestamp(stream_data, pack_idx)

        if PREFER_ZERO_COPY:
            send_by_phy = getattr(
                self.rtspserver,
                "rtspserver_sendvideodata_byphyaddr",
                None
            )
            phy_array = getattr(stream_data, "phy_addr", None)
            if send_by_phy is not None and phy_array is not None:
                phy_addr = phy_array[pack_idx]
                if phy_addr:
                    ret = send_by_phy(
                        RTSP_SESSION,
                        phy_addr,
                        size,
                        timestamp
                    )
                    if ret not in (None, 0):
                        raise OSError("RTSP 物理地址发送失败：", ret)
                    return size

        # 兼容路径：复制编码包后发送。稳定但会增加 Python 内存带宽。
        packet = bytes(
            uctypes.bytearray_at(stream_data.data[pack_idx], size)
        )
        ret = self.rtspserver.rtspserver_sendvideodata(
            RTSP_SESSION,
            packet,
            size,
            timestamp
        )
        if ret not in (None, 0):
            raise OSError("RTSP bytes 发送失败：", ret)
        return size

    def _stream_loop(self):
        stream_data = StreamData()
        stats_start_ms = time.ticks_ms()
        stats_frames = 0
        stats_bytes = 0
        total_frames = 0
        consecutive_failures = 0

        try:
            while self.start_stream:
                os.exitpoint()
                acquired = False

                try:
                    ret = self.encoder.GetStream(
                        stream_data,
                        STREAM_GET_TIMEOUT_MS
                    )
                    if ret not in (None, 0):
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_CONSECUTIVE_STREAM_FAILURES:
                            raise OSError("连续获取编码帧失败：", ret)
                        continue

                    acquired = True
                    frame_bytes = 0
                    for pack_idx in range(stream_data.pack_cnt):
                        frame_bytes += self._send_pack(stream_data, pack_idx)

                    consecutive_failures = 0
                    total_frames += 1
                    stats_frames += 1
                    stats_bytes += frame_bytes

                finally:
                    if acquired:
                        self.encoder.ReleaseStream(stream_data)

                if total_frames % GC_INTERVAL_FRAMES == 0:
                    gc.collect()

                now_ms = time.ticks_ms()
                elapsed_ms = time.ticks_diff(now_ms, stats_start_ms)
                if elapsed_ms >= STATUS_INTERVAL_MS:
                    actual_fps = stats_frames * 1000.0 / elapsed_ms
                    actual_kbps = stats_bytes * 8.0 / elapsed_ms
                    try:
                        free_heap_kb = gc.mem_free() // 1024
                        print(
                            "推流状态：%.1f fps，%.0f kbps，空闲堆 %d KB" %
                            (actual_fps, actual_kbps, free_heap_kb)
                        )
                    except Exception:
                        print(
                            "推流状态：%.1f fps，%.0f kbps" %
                            (actual_fps, actual_kbps)
                        )

                    stats_start_ms = now_ms
                    stats_frames = 0
                    stats_bytes = 0

        except BaseException as exc:
            if self.start_stream:
                self.last_error = exc
                print_exception_safe("RTSP 推流线程异常：", exc)
        finally:
            self.thread_finished = True

    def stop(self):
        self.start_stream = False

        wait_start = time.ticks_ms()
        while not self.thread_finished:
            if time.ticks_diff(time.ticks_ms(), wait_start) >= 3000:
                print("警告：等待推流线程退出超时，继续释放资源")
                break
            time.sleep_ms(50)

        if self.sensor_started and self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception as exc:
                print_exception_safe("停止摄像头时出现提示：", exc)
            self.sensor_started = False

        if self.link is not None:
            try:
                self.link.destroy()
            except Exception as exc:
                print_exception_safe("解除 Sensor/VENC 绑定时出现提示：", exc)
            self.link = None

        if self.encoder_started and self.encoder is not None:
            try:
                self.encoder.Stop()
            except Exception as exc:
                print_exception_safe("停止编码器时出现提示：", exc)
            self.encoder_started = False

        if (
            (self.encoder_created or self.encoder_configured) and
            self.encoder is not None
        ):
            try:
                self.encoder.Destroy()
            except Exception as exc:
                print_exception_safe("销毁编码器时出现提示：", exc)
            self.encoder_created = False
            self.encoder_configured = False

        if self.rtsp_started:
            try:
                self.rtspserver.rtspserver_stop()
            except Exception as exc:
                print_exception_safe("停止 RTSP 服务时出现提示：", exc)
            self.rtsp_started = False

        if self.rtsp_session_created:
            try:
                self.rtspserver.rtspserver_destroysession(RTSP_SESSION)
            except Exception as exc:
                print_exception_safe("销毁 RTSP 会话时出现提示：", exc)
            self.rtsp_session_created = False

        if self.rtsp_initialized or self.rtsp_init_attempted:
            try:
                self.rtspserver.rtspserver_deinit()
            except Exception as exc:
                print_exception_safe("释放 RTSP 服务时出现提示：", exc)
            self.rtsp_initialized = False
            self.rtsp_init_attempted = False

        try:
            self.rtspserver.rtspserver_destroy()
        except Exception:
            pass

        self.sensor = None
        self.encoder = None
        self.thread_finished = True
        gc.collect()
        print("RTSP 媒体链路已停止")


def stop_server_safe(server):
    if server is None:
        return
    try:
        server.stop()
    except Exception as exc:
        print_exception_safe("清理图传资源时出现提示：", exc)


def wait_before_reconnect(button, delay_ms=RECONNECT_DELAY_MS):
    start_ms = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start_ms) < delay_ms:
        os.exitpoint()
        if button is not None and button.poll():
            return False
        time.sleep_ms(50)
    return True


def main():
    os.exitpoint(os.EXITPOINT_ENABLE)
    validate_config()

    sta = None
    server = None
    button = create_user_button()
    stop_requested = False

    try:
        while not stop_requested:
            try:
                sta = connect_wifi(WIFI_SSID, WIFI_PASSWORD)
                wait_wifi_stable(sta)

                server = VisionReadyRtspServer()
                server.start()
                rtsp_url = server.get_url()

                print("")
                print("========== 视觉预留型图传已启动 ==========")
                print("RTSP 地址：", rtsp_url)
                print(
                    "图传：%dx%d @ %d fps，H.264 Baseline，%d kbps" %
                    (VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS,
                     VIDEO_BIT_RATE_KBPS)
                )
                if ENABLE_VISION_CHANNEL:
                    print(
                        "视觉通道：%dx%d RGB565，算法执行=%s" %
                        (VISION_WIDTH, VISION_HEIGHT, PROCESS_VISION)
                    )
                print("OBS 中添加媒体源并输入上面的完整地址")
                print("OBS 录像格式建议选择 MKV")
                print("==========================================")
                print("")

                last_vision_ms = time.ticks_ms()
                while True:
                    os.exitpoint()

                    if button is not None and button.poll():
                        print("检测到 USER 键长按，正在停止图传……")
                        stop_requested = True
                        break

                    if not wifi_has_ip(sta):
                        raise RuntimeError("Wi-Fi 已掉线")
                    if server.thread_finished:
                        if server.last_error is not None:
                            raise RuntimeError(
                                "推流线程退出：" + str(server.last_error)
                            )
                        raise RuntimeError("推流线程意外退出")

                    if PROCESS_VISION:
                        now_ms = time.ticks_ms()
                        if time.ticks_diff(now_ms, last_vision_ms) >= VISION_INTERVAL_MS:
                            image = server.snapshot_for_vision()
                            if image is not None and image != -1:
                                try:
                                    traditional_vision_step(image)
                                finally:
                                    del image
                            last_vision_ms = now_ms

                    time.sleep_ms(5 if PROCESS_VISION else 100)

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print_exception_safe("图传链路异常，将自动重建：", exc)
            finally:
                stop_server_safe(server)
                server = None

            if not stop_requested:
                print("%d 毫秒后重试……" % RECONNECT_DELAY_MS)
                if not wait_before_reconnect(button):
                    stop_requested = True

    except KeyboardInterrupt:
        print("收到停止指令")
    finally:
        stop_server_safe(server)
        try:
            if sta is not None:
                sta.disconnect()
        except Exception:
            pass
        gc.collect()

    print("RTSP 图传已结束，可以安全断电")


if __name__ == "__main__":
    main()
