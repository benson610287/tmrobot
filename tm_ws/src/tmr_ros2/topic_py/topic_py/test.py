# import darknet
# import numpy as np
# import cv2
# data_path = '/home/glenn/YOLOv4_cpu_ram/all/cfg/all.data'
# cfg_path = '/home/glenn/YOLOv4_cpu_ram/all/cfg/yolov4-custom_all.cfg'
# weights_path = '/home/glenn/YOLOv4_cpu_ram/all/cfg/weights/yolov4-custom_all_last.weights'

# img_path='/home/glenn/tmr_ros2/custom_package/image/tm_calib/668.jpg'

# network, class_names, class_colors = darknet.load_network(
#         cfg_path,
#         data_path,
#         weights_path,
#         batch_size=1
# )

# # x1, y1, x2, y2 =0,0,0,0


# """
# 影像檢測
#     輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
#     輸出:(檢測後影像,檢測結果)
#     註記:
# """
# def image_detection(image, network, class_names, class_colors, thresh):
#     width = darknet.network_width(network)
#     height = darknet.network_height(network)
#     darknet_image = darknet.make_image(width, height, 3)

    
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_resized = cv2.resize(image_rgb, (width, height),
#                                interpolation=cv2.INTER_LINEAR)

#     darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
#     detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
#     darknet.free_image(darknet_image)
#     image = darknet.draw_boxes(detections, image_resized, class_colors)

#     print(detections)

#     return detections
    

# """
# 座標轉換
#     輸入:(YOLO座標,原圖寬度,原圖高度)
#     輸出:(框的左上座標,框的右下座標)
#     註記:
# """
# def bbox2points(bbox,W,H,network):
#     """
#     From bounding box yolo format
#     to corner points cv2 rectangle
#     """ 
#     width = darknet.network_width(network)      # YOLO壓縮圖片大小(寬)
#     height = darknet.network_height(network)    # YOLO壓縮圖片大小(高)

#     x, y, w, h = bbox                           # (座標中心x,座標中心y,寬度比值,高度比值)
#     x = x*W/width
#     y = y*H/height
#     w = w*W/width
#     h = h*H/height
#     x1 = int(round(x - (w / 2)))
#     x2 = int(round(x + (w / 2)))
#     y1 = int(round(y - (h / 2)))
#     y2 = int(round(y + (h / 2)))
    
#     return x1, y1, x2, y2

# """
# 原圖繪製檢測框線
#     輸入:(檢測結果,原圖位置,框線顏色集)
#     輸出:(影像結果)
#     註記:
# """
# def draw_boxes(detections, image, colors, network):
#     # yolo_info = [[0 for i in range(2)] for j in range(4)]
#     yolo_info = [[0 for i in range(2)] for j in range(4)]
#     H,W,_ = image.shape
#     img = image.copy()
#     i=0
#     j=0
#     for label, confidence, bbox in detections: 
#         x1, y1, x2, y2 = bbox2points(bbox,W,H,network)

#         cv2.rectangle(img, (x1, y1), (x2, y2), colors[label], 1)
#         cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)),
#                     (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     colors[label], 2)
#         # 輸出框座標_加工格式座標(左上點座標,右上點座標)
#         print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), x1, y1, x2, y2))

#         yolo_info[i][0] = (x1+x2)/2
#         yolo_info[i][1] = (y1+y2)/2
#         i+=1


#     return img,yolo_info










# if __name__=="__main__":

#     base_H_flange=np.array([[-1,      -0.0006,      0.028 ,  0.17047 ],
#                             [ -0.0006,       0.999,  0.033,  0.68553],
#                             [-0.028,       0.033,  -0.999, 0.52501],
#                             [0,       0,       0,      1]])                                         

#     flang_H_eye=np.array([[1,   0,  0,  0   ],#-0.0681
#                           [0,   1,  0,  0.0582       ],
#                           [0,   0,  1,  0.079  ],
#                           [0,   0,  0,  1       ]])
#     RX=np.array([[1,0,0,0],
#                         [0,-1,0,0],
#                         [0,0,-1,0],
#                         [0,0,0,1]])
#     RZ=np.array([[-1,0,0,0],
#                         [0,-1,0,0],
#                         [0,0,1,0],
#                         [0,0,0,1]])
    

#     base_H_eye=np.matmul(flang_H_eye,base_H_flange)
#     print('base_H_eye:\n', base_H_eye)


#     img_path=input("image path:")
#     cv2.namedWindow('ww',cv2.WINDOW_NORMAL ) #cv2.WINDOW_NORMAL    cv2.WINDOW_AUTOSIZE
#     img=cv2.imread(img_path)
#     detections_1 = image_detection(img,network, class_names, class_colors, thresh=0.9)
#     out_img1,yolo_info = draw_boxes(detections_1, img, class_colors,network)

#     cv2.imshow('ww', out_img1)
#     cv2.imwrite("out_img1.jpg",out_img1)

#     print("yolo_info_x=",yolo_info[0][0],"yolo_info_y=",yolo_info[0][1])
    
#     x=(0.395/2592)*(yolo_info[0][0]-1296)
#     y=-(0.29/1944)*(yolo_info[0][1]-972)
#     print("x=",x,"y=",y)
#     eye_H_obj=np.array([[1,0,0,x],
#                         [0,1,0,y],
#                         [0,0,1,-0.4],
#                         [0,0,0,1]])

#     base_H_obj=np.matmul(eye_H_obj,base_H_eye)

#     print("base_H_obj:\n",base_H_obj)


#     cv2.waitKey(0)





import cv2
import matplotlib.pyplot as plt
img_path=input("image path:")
img=cv2.imread(img_path)
# # arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoDict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250,5)
arucoParams = cv2.aruco.DetectorParameters()
# arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict ,
	parameters=arucoParams)


print(corners,'\n',ids,'\n',rejected)





# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# id = 11 # 這是書籤的標識符，您可以將其更改為您需要的任何內容
# img_size = 700 # 定義最終圖像的大小
# marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, img_size)
# plt.imshow(marker_img, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.show()
# cv2.imwrite("QQ.jpg",marker_img)
