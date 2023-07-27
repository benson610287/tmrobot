import rclpy
from rclpy.node import Node

#桌面對齊點位  右前{222,741,91,179,1.8,-179}
#            左前{-85,741,91,179,1.8,-179}
#            右後{87,382,91,179,1.8,-179}     -90.21



import numpy as np
import cv2
import math

from std_msgs.msg import Float64MultiArray
import topic_py.darknet
import tm_msgs.msg

#

# base_H_flange_new=np.array([[-0.999,      -0.019,      0.037 ,  -0.00732 ],
#                         [ -0.018,       0.999,  0.035,  0.4555],
#                         [-0.038,       0.034,  -0.999, 0.41313],
#                         [0,       0,       0,      1]])  
base_H_flange_new=np.array([[-0.999,      -0.019,      0.032 ,  -0.00732 ],
                        [ -0.017,       0.999,  0.051,  0.4555],
                        [-0.033,       0.051,  -0.998, 0.41313],
                        [0,       0,       0,      1]]) 



flang_H_eye=np.array([[1,   0,  0,  0   ],#-0.0681
                        [0,   1,  0,  0.0582       ],
                        [0,   0,  1,  0.079  ],
                        [0,   0,  0,  1       ]])

RX=np.array([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,-1,0],
                    [0,0,0,1]])
RZ=np.array([[-1,0,0,0],
                    [0,-1,0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
#拍照點位：-7.32,    455.5,   413.14,   179.07,   1.79,   -178.96
#開鎖: -93.97, 469.45, 350,178, 97, 0.93, -179.95
#cpu_check: -89.92, 548.75, 137.77, -177.13, -1.45, -177.36
#ram1_check1: 5.11, 503.89, 187.22 , 180, 0, -90
#ram1_check2: 5.11 ,64.55, 187.22 ,180 , 0, -90
#ram2_check1: 18.96, 515.89, 187.24, -180, 0 ,90
#ram2_check2: 18.97, 680.88, 187.25, -180, 0, -90

take_pic_posision_X=-7.32
take_pic_posision_Y=455.5
take_pic_posision_Z=413.13
take_pic_posision_RX=179.07/180*math.pi
take_pic_posision_RY=1.79/180*math.pi
take_pic_posision_RZ=-178.96/180*math.pi

data_path = '/home/glenn/work/tm_ws/src/tmr_ros2/YOLOv4_cpu_ram/all/cfg/all.data'
cfg_path = '/home/glenn/work/tm_ws/src/tmr_ros2/YOLOv4_cpu_ram/all/cfg/yolov4-custom_all.cfg'
weights_path = '/home/glenn/work/tm_ws/src/tmr_ros2/YOLOv4_cpu_ram/all/cfg/weights/yolov4-custom_all_last.weights'

img_path='/home/glenn/work/tm_ws/src/tmr_ros2/custom_package/tm_img.jpg'


network, class_names, class_colors = topic_py.darknet.load_network(
        cfg_path,
        data_path,
        weights_path,
        batch_size=1
)

k=0

x1, y1, x2, y2 =0,0,0,0



"""
影像檢測
    輸入:(影像位置,神經網路,物件名稱集,信心值閥值(0.0~1.0))
    輸出:(檢測後影像,檢測結果)
    註記:
"""
def image_detection(image, network, class_names, class_colors, thresh):
    width = topic_py.darknet.network_width(network)
    height = topic_py.darknet.network_height(network)
    darknet_image = topic_py.darknet.make_image(width, height, 3)

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    topic_py.darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = topic_py.darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    topic_py.darknet.free_image(darknet_image)
    image = topic_py.darknet.draw_boxes(detections, image_resized, class_colors)

    print(detections)

    return detections
    
"""
座標轉換
    輸入:(YOLO座標,原圖寬度,原圖高度)
    輸出:(框的左上座標,框的右下座標)
    註記:
"""
def bbox2points(bbox,W,H,network):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """ 
    width = topic_py.darknet.network_width(network)      # YOLO壓縮圖片大小(寬)
    height = topic_py.darknet.network_height(network)    # YOLO壓縮圖片大小(高)

    x, y, w, h = bbox                           # (座標中心x,座標中心y,寬度比值,高度比值)
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
def draw_boxes(detections, image, colors, network):
    yolo_info = [[0 for i in range(3)] for j in range(9)]
    H,W,_ = image.shape
    img = image.copy()
    i=0
    j=0
    for label, confidence, bbox in detections: 
        x1, y1, x2, y2 = bbox2points(bbox,W,H,network)

        cv2.rectangle(img, (x1, y1), (x2, y2), colors[label], 1)
        cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)),
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
        # 輸出框座標_加工格式座標(左上點座標,右上點座標)
        print("\t{}\t: {:3.2f}%    (x1: {:4.0f}   y1: {:4.0f}   x2: {:4.0f}   y2: {:4.0f})".format(label, float(confidence), x1, y1, x2, y2))
        if label=='cpu':
            yolo_info[i][0] = (x1+x2)/2
            yolo_info[i][1] = (y1+y2)/2
            yolo_info[i][2] = 1
            i+=1
        elif label=='ram':
            yolo_info[i][0] = (x1+x2)/2
            yolo_info[i][1] = (y1+y2)/2
            yolo_info[i][2] = 2
            i+=1
        elif label=='close_cpu_slot':
            yolo_info[i][0] = (x1+x2)/2
            yolo_info[i][1] = (y1+y2)/2
            yolo_info[i][2] = 3
            i+=1
        elif label=='ram_slot':
            yolo_info[i][0] = (x1+x2)/2
            yolo_info[i][1] = (y1+y2)/2
            yolo_info[i][2] = 4
            i+=1
  

    return img,yolo_info


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'topic', 10)
        self.subscription= self.create_subscription(
        tm_msgs.msg.SctResponse,
        "sct_response",
        self.timer_callback,
        10)

        self.ofset_Z=0.0035
        self.aruco_changed_RX = 0.0/180*math.pi
        self.aruco_changed_RY = 0.5555227751412871/180*math.pi
        self.yolo_info=[[0 for i in range(3)] for j in range(10)]
        self.flag_detected=0
        self.temp=0  #紀錄CPU位

        self.flag_CPU_on_arm=False   #以下flag要全false
        self.flag_CPU_done=False 
        self.flag_ram_on_arm=False
        self.flag_ram_done1=False
        self.flag_ram_done2=False
        self.i = 0
    def timer_callback(self,msg):
        if msg.id=='0':
            if self.flag_detected==0:
                self.get_logger().info('yolo detecting')
                img=cv2.imread(img_path)
                detections_1 = image_detection(img,network, class_names, class_colors, thresh=0.9)
                out_img1,self.yolo_info = draw_boxes(detections_1, img, class_colors,network)
                cv2.imwrite("out_img1.jpg",out_img1)
                self.yolo_info.sort(key=lambda x:x[2],reverse=True)
                self.flag_detected=1
                print(self.yolo_info)
                # cv2.namedWindow('ww',cv2.WINDOW_NORMAL ) #cv2.WINDOW_NORMAL    cv2.WINDOW_AUTOSIZE
                # cv2.imshow('ww', out_img1)
                # cv2.waitKey()
            else:
                pass
            a=[]
            msg1 = Float64MultiArray()
            k=0
            if self.flag_CPU_on_arm==False and self.flag_CPU_done==False:
                for k in range(9):
                    if self.yolo_info[k][2]==1 and self.yolo_info[k][0]<=1900: #yolo_info[k][2]=1 is cpu
                        base_H_eye=np.matmul(flang_H_eye,base_H_flange_new,)
                        print('base_H_eye:\n', base_H_eye)

                        print("yolo_info_x=",self.yolo_info[k][0],"yolo_info_y=",self.yolo_info[k][1])
                        
                        x=(0.4/2592)*(self.yolo_info[k][0]-1296)
                        y=-(0.295/1944)*(self.yolo_info[k][1]-972)
                        print("x=",x,"y=",y)
                        eye_H_obj=np.array([[1,0,0,x],
                                            [0,1,0,y],
                                            [0,0,1,-0.4],
                                            [0,0,0,1]]) 
                        base_H_obj=np.matmul(eye_H_obj,base_H_eye)
                        a.append(float('0'))
                        a.append(base_H_obj[0][3]+0.0085)#
                        a.append(base_H_obj[1][3]+0.00017)#
                        a.append(0.11-self.ofset_Z)#110
                        # a.append(3.14)
                        # a.append(0.0)
                        a.append(take_pic_posision_RX-0.7853981634 + self.aruco_changed_RX)#2.35619449 + self.aruco_changed_RX
                        a.append(take_pic_posision_RY - self.aruco_changed_RY)#0.03455751919 + self.aruco_changed_RY
                        a.append(-3.1311206781)
                        
                        msg1.data=a
                        self.publisher_.publish(msg1)
                        self.get_logger().info('taking CPU')
                        for j in range(7):
                            self.get_logger().info('Publishing: "%f"' % msg1.data[j])
                        # self.yolo_info.pop(k)
                        self.yolo_info[k][2]=1000
                        self.flag_CPU_on_arm=True
                        return 0
                    
            if self.flag_CPU_on_arm==True and self.flag_CPU_done==False:
                for k in range(9):
                    if self.yolo_info[k][2]==3: #yolo_info[k][2]=3 is close_cpu_slot
                        base_H_eye=np.matmul(flang_H_eye,base_H_flange_new)
                        print('base_H_eye:\n', base_H_eye)

                        print("yolo_info_x=",self.yolo_info[k][0],"yolo_info_y=",self.yolo_info[k][1])
                        x=(0.4/2592)*(self.yolo_info[k][0]-1296)
                        y=-(0.30/1944)*(self.yolo_info[k][1]-972)
                        print("x=",x,"y=",y)
                        eye_H_obj=np.array([[1,0,0,x],
                                            [0,1,0,y],
                                            [0,0,1,-0.415],
                                            [0,0,0,1]])

                        base_H_obj=np.matmul(eye_H_obj,base_H_eye)
                        print("base_H_obj:\n",base_H_obj)
                        a.append(float('0'))
                        a.append(base_H_obj[0][3]-0.0005)#-0.0022
                        a.append(base_H_obj[1][3]-0.003531)#-0.00271
                        a.append(0.115-self.ofset_Z)
                        a.append(take_pic_posision_RX-0.7853981634 + self.aruco_changed_RX)#2.35619449 + self.aruco_changed_RX
                        a.append(take_pic_posision_RY - self.aruco_changed_RY)#0.03455751919 + self.aruco_changed_RY
                        a.append(-3.1241393611)  

                        msg1.data=a
                        self.publisher_.publish(msg1)
                        self.get_logger().info('placing CPU')
                        for j in range(7):
                            self.get_logger().info('Publishing: "%f"' % msg1.data[j])
                        # self.yolo_info.pop(k)
                        self.yolo_info[k][2]=10000
                        self.flag_CPU_done=True
                        #self.flag_CPU_on_arm=False

                        self.temp=k
                        return 0
                    
            if self.flag_CPU_done==True and self.flag_ram_on_arm==False:
                for k in range(9):
                    if self.yolo_info[k][2]==2 and self.yolo_info[k][1]<=1000 : #yolo_info[k][2]=2 is ram  1000    
                        base_H_eye=np.matmul(flang_H_eye,base_H_flange_new)
                        print('base_H_eye:\n', base_H_eye)
                        print("yolo_info_x=",self.yolo_info[k][0],"yolo_info_y=",self.yolo_info[k][1])
                        x=(0.41/2592)*(self.yolo_info[k][0]-1296)
                        y=-(0.31/1944)*(self.yolo_info[k][1]-972)
                        print("x=",x,"y=",y)
                        eye_H_obj=np.array([[1,0,0,x],
                                            [0,1,0,y],
                                            [0,0,1,-0.43],
                                            [0,0,0,1]])

                        base_H_obj=np.matmul(eye_H_obj,base_H_eye)
                        print("base_H_obj:\n",base_H_obj)
                        a.append(float('0'))
                        a.append(base_H_obj[0][3]+0.008)
                        a.append(base_H_obj[1][3]+0.15)
                        a.append(0.080-self.ofset_Z)    #270
                        a.append(-2.4129176909+ self.aruco_changed_RX) # 
                        a.append(-0.0068067841+ self.aruco_changed_RY)
                        a.append(3.13583306706)
                        msg1.data=a
                        self.publisher_.publish(msg1)
                        self.get_logger().info('taking RAM1')
                        for j in range(7):
                            self.get_logger().info('Publishing: "%f"' % msg1.data[j])
                        # self.yolo_info.pop(k)
                        self.yolo_info[k][2]=100000
                        self.flag_ram_on_arm=True
                        print(self.flag_ram_on_arm)
                        self.get_logger().info('ADSDSDSDSD')
                        return 0



            if  self.flag_ram_on_arm==True and self.flag_ram_done1==False:
                for k in range(9):
                    if self.yolo_info[k][2]==4 and self.yolo_info[k][0]<=820:  #yolo_info[k][2]=4 is ram_slot
                        base_H_eye=np.matmul(flang_H_eye,base_H_flange_new)
                        print('base_H_eye:\n', base_H_eye)
                        print("yolo_info_x=",self.yolo_info[k][0],"yolo_info_y=",self.yolo_info[k][1])
                        x=(0.4/2592)*(self.yolo_info[k][0]-1296)
                        y=-(0.30/1944)*(self.yolo_info[k][1]-972)
                        print("x=",x,"y=",y)
                        eye_H_obj=np.array([[1,0,0,x],
                                            [0,1,0,y],
                                            [0,0,1,-0.415],
                                            [0,0,0,1]])
                        print("beforea=",a)
                        base_H_obj=np.matmul(eye_H_obj,base_H_eye)
                        print("base_H_obj:\n",base_H_obj)
                        a.append(float('0'))
                        a.append(base_H_obj[0][3]) #+0.00553
                        a.append(base_H_obj[1][3]) #+0.01904
                        a.append(0.200-self.ofset_Z)  
                        a.append(3.1236157623+ self.aruco_changed_RX) 
                        a.append(0.0+ self.aruco_changed_RY)
                        a.append(-1.57446111)
                        msg1.data=a
                        print("aftrera=",a)

                        self.publisher_.publish(msg1)
                        self.get_logger().info('placing RAM1')
                        for j in range(7):
                            self.get_logger().info('Publishing: "%f"' % msg1.data[j])
                        # self.yolo_info.pop(k)
                        self.yolo_info[k][2]=100000
                        self.flag_ram_done1=True
                        self.flag_ram_on_arm=False
                        return 0
                    



            if self.flag_ram_done1==True and self.flag_ram_on_arm==False:
                for k in range(9):
                    if self.yolo_info[k][2]==2 and self.yolo_info[k][1]<=1300: #yolo_info[k][2]=2 is ram  1000    
                        base_H_eye=np.matmul(flang_H_eye,base_H_flange_new)
                        print('base_H_eye:\n', base_H_eye)
                        print("yolo_info_x=",self.yolo_info[k][0],"yolo_info_y=",self.yolo_info[k][1])
                        x=(0.41/2592)*(self.yolo_info[k][0]-1296)
                        y=-(0.31/1944)*(self.yolo_info[k][1]-972)
                        print("x=",x,"y=",y)
                        eye_H_obj=np.array([[1,0,0,x],
                                            [0,1,0,y],
                                            [0,0,1,-0.43],
                                            [0,0,0,1]])

                        base_H_obj=np.matmul(eye_H_obj,base_H_eye)
                        print("base_H_obj:\n",base_H_obj)
                        a.append(float('0'))
                        a.append(base_H_obj[0][3]+0.008)
                        a.append(base_H_obj[1][3]+0.15)
                        a.append(0.080-self.ofset_Z)    #270
                        a.append(-2.4129176909+ self.aruco_changed_RX) # 
                        a.append(-0.0068067841+ self.aruco_changed_RY)
                        a.append(3.13583306706)
                        msg1.data=a
                        self.publisher_.publish(msg1)
                        self.get_logger().info('taking RAM2')
                        for j in range(7):
                            self.get_logger().info('Publishing: "%f"' % msg1.data[j])
                        # self.yolo_info.pop(k)
                        self.yolo_info[k][2]=100000
                        self.flag_ram_on_arm=True
                        print(self.flag_ram_on_arm)
                        return 0


            if self.flag_ram_on_arm==True and self.flag_ram_done2==False:
                for k in range(9):
                    if self.yolo_info[k][2]==4 : #yolo_info[k][2]=4 is ram_slot     and self.yolo_info[k][0]<=1450
                        base_H_eye=np.matmul(flang_H_eye,base_H_flange_new)
                        print('base_H_eye:\n', base_H_eye)
                        print("yolo_info_x=",self.yolo_info[k][0],"yolo_info_y=",self.yolo_info[k][1])
                        x=(0.4/2592)*(self.yolo_info[k][0]-1296)
                        y=-(0.30/1944)*(self.yolo_info[k][1]-972)
                        print("x=",x,"y=",y)
                        eye_H_obj=np.array([[1,0,0,x],
                                            [0,1,0,y],
                                            [0,0,1,-0.415],
                                            [0,0,0,1]])
                        print("beforea=",a)
                        base_H_obj=np.matmul(eye_H_obj,base_H_eye)
                        print("base_H_obj:\n",base_H_obj)
                        a.append(float('0'))
                        a.append(base_H_obj[0][3]) #+0.00553
                        a.append(base_H_obj[1][3]) #+0.01904
                        a.append(0.200-self.ofset_Z)  
                        a.append(3.1236157623+ self.aruco_changed_RX) 
                        a.append(0.0+ self.aruco_changed_RY)
                        a.append(-1.57446111)
                        msg1.data=a
                        print("aftrera=",a)

                        self.publisher_.publish(msg1)
                        self.get_logger().info('placing RAM2')
                        for j in range(7):
                            self.get_logger().info('Publishing: "%f"' % msg1.data[j])
                        # self.yolo_info.pop(k)
                        self.yolo_info[k][2]=100000
                        self.flag_ram_done1=True
                        self.flag_ram_on_arm=False
                        return 0



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()




if __name__ == '__main__':
    main()