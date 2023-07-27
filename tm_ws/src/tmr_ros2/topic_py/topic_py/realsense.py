
# License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
## Open CV and Numpy integration ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import darknet

i=0
data_path = '/home/iarc/tm_train/cfg/first_train.data'
cfg_path = '/home/iarc/tm_train/cfg/yolov4-tiny.cfg'
weights_path = '/home/iarc/tm_train/cfg/weights/yolov4-tiny_final.weights'

network, class_names, class_colors = darknet.load_network(
cfg_path,
data_path,
weights_path,
batch_size=1
)


"""
影像檢測
輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
輸出:(檢測後影像,檢測結果)
註記:
"""
def image_detection(image, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
    interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)

    print(detections)

    return detections



"""
座標轉換
輸入:(YOLO座標,原圖寬度,原圖高度)
輸出:(框的左上座標,框的右下座標)
註記:
"""
def bbox2points(bbox,W,H):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    width = darknet.network_width(network) # YOLO壓縮圖片大小(寬)
    height = darknet.network_height(network) # YOLO壓縮圖片大小(高)

    x, y, w, h = bbox # (座標中心x,座標中心y,寬度比值,高度比值)
    x = x*W/width
    y = y*H/height
    w = w*W/width
    h = h*H/height
    x1 = int(round(x - (w / 2)))
    x2 = int(round(x + (w / 2)))
    y1 = int(round(y - (h / 2)))
    y2 = int(round(y + (h / 2)))

    return x1, y1, x2, y2



"""
原圖繪製檢測框線
輸入:(檢測結果,原圖位置,框線顏色集)
輸出:(影像結果)
註記:
"""
def draw_boxes(detections, image, colors):
    H,W,_ = image.shape
    img = image.copy()

    for label, confidence, bbox in detections:
        x1, y1, x2, y2 = bbox2points(bbox,W,H)

        cv2.rectangle(img, (x1, y1), (x2, y2), colors[label], 1)
        cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)),
        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        colors[label], 2)
    # 輸出框座標_加工格式座標(左上點座標,右上點座標)
    print("\t{}\t: {:3.2f}% (x1: {:4.0f} y1: {:4.0f} x2: {:4.0f} y2: {:4.0f})".format(label, float(confidence), x1, y1, x2, y2))


    return img







# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
i=700
try:
    while True:

    # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))



        detections = image_detection(color_image,network, class_names, class_colors, thresh=0.75)
        out_img = draw_boxes(detections, color_image, class_colors)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', out_img)
        n=cv2.waitKey(1)
        if n == ord('q'):
            break
        elif n == ord('s'):
            cv2.imwrite('img/'+ str(i) + '.jpg',out_img)
            # print('save:',file_name + '_' + str(i) + '.jpg')
            i = i + 1
            print(i)

finally:

    # Stop streaming
    pipeline.stop()