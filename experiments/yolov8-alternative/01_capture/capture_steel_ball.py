import os,sys
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import time,image,random,gc
from libs.Utils import *

#-----------------------------其他必要方法---------------------------------------------
# 多目标检测 非最大值抑制方法实现
def nms(boxes,scores,thresh):
    """Pure Python NMS baseline."""
    x1,y1,x2,y2 = boxes[:, 0],boxes[:, 1],boxes[:, 2],boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores,axis = 0)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        new_x1,new_y1,new_x2,new_y2,new_areas = [],[],[],[],[]
        for order_i in order:
            new_x1.append(x1[order_i])
            new_x2.append(x2[order_i])
            new_y1.append(y1[order_i])
            new_y2.append(y2[order_i])
            new_areas.append(areas[order_i])
        new_x1 = np.array(new_x1)
        new_x2 = np.array(new_x2)
        new_y1 = np.array(new_y1)
        new_y2 = np.array(new_y2)
        xx1 = np.maximum(x1[i], new_x1)
        yy1 = np.maximum(y1[i], new_y1)
        xx2 = np.minimum(x2[i], new_x2)
        yy2 = np.minimum(y2[i], new_y2)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        new_areas = np.array(new_areas)
        ovr = inter / (areas[i] + new_areas - inter)
        new_order = []
        for ovr_i,ind in enumerate(ovr):
            if ind < thresh:
                new_order.append(order[ovr_i])
        order = np.array(new_order,dtype=np.uint8)
    return keep

# 计算padding缩放比例和上下左右padding大小
def letterbox_pad_param(input_size,output_size):
    ratio_w = output_size[0] / input_size[0]  # 宽度缩放比例
    ratio_h = output_size[1] / input_size[1]   # 高度缩放比例
    ratio = min(ratio_w, ratio_h)  # 取较小的缩放比例
    new_w = int(ratio * input_size[0])  # 新宽度
    new_h = int(ratio * input_size[1])  # 新高度
    dw = (output_size[0] - new_w) / 2  # 宽度差
    dh = (output_size[1] - new_h) / 2  # 高度差
    top = int(round(0))
    bottom = int(round(dh * 2 + 0.1))
    left = int(round(0))
    right = int(round(dw * 2 - 0.1))
    return top, bottom, left, right,ratio


#-----------------------------Sensor/Display初始化部分-------------------------------

# 定义屏幕显示分辨率
DISPLAY_WIDTH = ALIGN_UP(1920, 16)
DISPLAY_HEIGHT = 1080

# 定义AI推理帧分辨率
AI_RGB888P_WIDTH = ALIGN_UP(1280, 16)
AI_RGB888P_HEIGHT = 720

sensor = Sensor()
sensor.reset()
# 设置水平镜像和垂直翻转，不同板子的方向不同，通过配置这两个参数使画面转正
#sensor.set_hmirror(False)
#sensor.set_vflip(False)

# 配置sensor的多通道出图，每个通道的出图格式和分辨率可以不同，最多可以出三路图，参考sensor API文档
# 通道0直接给到显示VO，格式为YUV420
sensor.set_framesize(width = DISPLAY_WIDTH, height = DISPLAY_HEIGHT,chn=CAM_CHN_ID_0)
sensor.set_pixformat(Sensor.YUV420SP,chn=CAM_CHN_ID_0)
# 通道1给到AI做算法处理，格式为RGB888P
sensor.set_framesize(width = AI_RGB888P_WIDTH , height = AI_RGB888P_HEIGHT, chn=CAM_CHN_ID_1)
# set chn2 output format
sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_1)

# 绑定通道0的摄像头图像到屏幕，防止另一个通道的AI推理过程太慢影响显示过程，导致出现卡顿效果
sensor_bind_info = sensor.bind_info(x = 0, y = 0, chn = CAM_CHN_ID_0)
Display.bind_layer(**sensor_bind_info, layer = Display.LAYER_VIDEO1)

# OSD图像初始化,创建一帧和屏幕分辨率同样大的透明图像，用于绘制AI推理结果
osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)

# 设置为LT9611显示，默认1920x1080
Display.init(Display.LT9611,width=DISPLAY_WIDTH,height=DISPLAY_HEIGHT,osd_num=1, to_ide = True)
## 如果使用ST7701的LCD屏幕显示，默认800*480,还支持640*480等，具体参考Display模块API文档
#Display.init(Display.ST7701, width=DISPLAY_WIDTH,height=DISPLAY_HEIGHT,osd_num=1, to_ide=True)

# 限制bind通道的帧率，防止生产者太快
sensor._set_chn_fps(chn = CAM_CHN_ID_0, fps = Display.fps())


#-----------------------------AI模型初始化部分-------------------------------
# Kmodel模型路径
kmodel_path="/sdcard/examples/kmodel/yolov8n_224.kmodel"
# 类别标签
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# 模型输入分辨率
model_input_size=[224,224]
# 其它参数设置，包括阈值、最大检测框数量等
confidence_threshold = 0.3
nms_threshold = 0.4
max_boxes_num = 50
# 不同类别框的颜色
colors=get_colors(len(labels))

# 初始化ai2d预处理，并配置ai2d padding+resize预处理，预处理过程输入分辨率为图片分辨率，输出分辨率模型输入的需求分辨率，实现image->preprocess->model的过程
ai2d=nn.ai2d()
# 配置ai2d模块的输入输出数据类型和格式
ai2d.set_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
# 设置padding的参数，上下左右padding的大小和三个通道padding的具体值
top,bottom,left,right,ratio=letterbox_pad_param([AI_RGB888P_WIDTH,AI_RGB888P_HEIGHT],model_input_size)
ai2d.set_pad_param(True,[0,0,0,0,top,bottom,left,right], 0, [128,128,128])
# 设置resize参数，配置插值方法
ai2d.set_resize_param(True,nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
# 设置ai2d模块的输入输出维度，并构建builder实例
ai2d_builder = ai2d.build([1,3,AI_RGB888P_HEIGHT,AI_RGB888P_WIDTH], [1,3,model_input_size[1],model_input_size[0]])
# 初始化一个空的tensor，用于ai2d输出和kpu输入，因为一般ai2d的输出会直接送给kpu，因此这里使用一个变量共用
input_init_data = np.ones((1,3,model_input_size[1],model_input_size[0]),dtype=np.uint8)
kpu_input_tensor = nn.from_numpy(input_init_data)


# 创建kpu实例
kpu=nn.kpu()
# 加载kmodel模型
kpu.load_kmodel(kmodel_path)

# media初始化
MediaManager.init()
# 启动sensor
sensor.run()
# 测试帧率
fps = time.clock()
while True:
    fps.tick()
    #------------------------从摄像头dump一帧图像并处理----------------------------------
    # 从摄像头1通道dump一帧RGB888P格式的Image图像
    img=sensor.snapshot(chn=CAM_CHN_ID_1)
    # 转换成ulab.numpy.ndarray格式的数据，CHW
    img_np=img.to_numpy_ref()
    # 创建nncase_runtime.tensor用于给到ai2d进行预处理
    ai2d_input_tensor=nn.from_numpy(img_np)
    #------------------------推理前的预处理步骤----------------------------------------
    # 执行预处理过程
    ai2d_builder.run(ai2d_input_tensor, kpu_input_tensor)
    #------------------------使用kpu完成模型推理--------------------------------------
    # 设置kpu的第0个输入为ai2d预处理后的tensor，如果有多个，可以依次设置
    kpu.set_input_tensor(0,kpu_input_tensor)
    # 在kpu上执行模型推理
    kpu.run()
    #------------------------获取模型推理结束的输出----------------------------------------
    # 获取模型推理的输出tensor，并将其转换成ulab.numpy.ndarray数据进行后处理
    results=[]
    for i in range(kpu.outputs_size()):
        output_i_tensor = kpu.get_output_tensor(i)
        result_i = output_i_tensor.to_numpy()
        results.append(result_i)
        del output_i_tensor
    #------------------------推理输出的后处理步骤----------------------------------------
    # YOLOv8检测模型输出只有1个，也就是results[0]的shape为[1,box_dim，box_num]，results[0][0]表示[box_dim,box_num]，转换成[box_num,box_dim]方便依次处理每个框
    output_data=results[0][0].transpose()
    # 每个框前四个数据为中心点坐标和宽高
    boxes_ori = output_data[:,0:4]
    # 剩余数据为每个类别的分数，通过argmax找到分数最大的类别编号和分数值
    class_ori = output_data[:,4:]
    class_res=np.argmax(class_ori,axis=-1)
    scores_ = np.max(class_ori,axis=-1)
    # 通过置信度阈值筛选框（小于置信度阈值的丢弃），同时处理坐标为x1,y1,x2,y2，为框的左上和右下的坐标,注意比例变换，将输入分辨率坐标(model_input_size)转换成原图坐标(AI_RGB888P_WIDTH,AI_RGB888P_HEIGHT)
    boxes,inds,scores=[],[],[]
    for i in range(len(boxes_ori)):
        if scores_[i]>confidence_threshold:
            x,y,w,h=boxes_ori[i][0],boxes_ori[i][1],boxes_ori[i][2],boxes_ori[i][3]
            x1 = int((x - 0.5 * w)/ratio)
            y1 = int((y - 0.5 * h)/ratio)
            x2 = int((x + 0.5 * w)/ratio)
            y2 = int((y + 0.5 * h)/ratio)
            boxes.append([x1,y1,x2,y2])
            inds.append(class_res[i])
            scores.append(scores_[i])
    #如果第一轮筛选后无框，继续下一帧处理
    if len(boxes)==0:
        continue
    # 将list转换成ulab.numpy.ndarray方便处理
    boxes = np.array(boxes)
    scores = np.array(scores)
    inds = np.array(inds)
    # NMS过程,去除重叠的冗余框，keep为
    keep = nms(boxes,scores,nms_threshold)
    dets = np.concatenate((boxes, scores.reshape((len(boxes),1)), inds.reshape((len(boxes),1))), axis=1)
    # 得到最后的检测框的结果
    det_res = []
    for keep_i in keep:
        det_res.append(dets[keep_i])
    det_res = np.array(det_res)
    # 去前max_box_num个，防止检测框过多
    det_res = det_res[:max_boxes_num, :]
    #------------------------绘制检测框结果----------------------------------------
    osd_img.clear()
    # 分别处理每一个框，将原图坐标(AI_RGB888P_WIDTH,AI_RGB888P_HEIGHT)转换成显示屏幕坐标(DISPLAY_WIDTH,DISPLAY_HEIGHT)
    for det in det_res:
        x_1, y_1, x_2, y_2 = map(lambda pos: int(round(pos, 0)), det[:4])
        draw_x= int(x_1 * DISPLAY_WIDTH // AI_RGB888P_WIDTH)
        draw_y= int(y_1 * DISPLAY_HEIGHT // AI_RGB888P_HEIGHT)
        draw_w = int((x_2 - x_1) * DISPLAY_WIDTH // AI_RGB888P_WIDTH)
        draw_h = int((y_2 - y_1) * DISPLAY_HEIGHT // AI_RGB888P_HEIGHT)
        osd_img.draw_rectangle(draw_x,draw_y, draw_w, draw_h, color=colors[int(det[5])],thickness=4)
        osd_img.draw_string_advanced( draw_x , max(0,draw_y-50), 24, labels[int(det[5])] + " {0:.3f}".format(det[4]), color=colors[int(det[5])])
    #------------------------在屏幕显示检测框结果----------------------------------------
    Display.show_image(osd_img)
    print("det fps:",fps.fps())
    gc.collect()

#退出循环，释放资源
del ai2d
del kpu
sensor.stop()
Display.deinit()
time.sleep_ms(50)
MediaManager.deinit()
nn.shrink_memory_pool()