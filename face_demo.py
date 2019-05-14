#!/usr/bin/python anaconda
# -*- coding:utf-8 -*-
import cv2
import numpy as np
# 添加模块和矩阵模块
# cap = cv2.VideoCapture(0)
# # # 打开摄像头，若打开本地视频，同opencv一样，只需将０换成("×××.avi")
# # while(1):    # get a frame
# #     ret, frame = cap.read()    # show a frame
# #     cv2.imshow("capture", frame)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# # cap.release()
# # cv2.destroyAllWindows()
# =======================================================
cap = cv2.VideoCapture(0)

# 告诉OpenCV使用人脸识别分类器
classfier = cv2.CascadeClassifier(r"F:\Anaconda3_5.2.0\Lib\site-packages\cv2\data/haarcascade_frontalface_alt2.xml")

# 识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
color = (0, 255, 0)

num = 0
while cap.isOpened():
    ok, frame = cap.read()  # 读取一帧数据
    if not ok:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

    # 人脸检测，1.2和3分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects) > 0:          # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect

            # 将当前帧保存为图片
            img_name = "%s/%d.jpg" % (r"C:\Users\Administrator\Desktop/face_recognition/video_pic", num)
            # print(img_name)
            image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
            cv2.imwrite(img_name, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

            num += 1
            if num > 19:   # 如果超过指定最大保存数量退出循环，num从0-19，则保存20张
                break

            # 画出矩形框
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

            # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数  # 可以去掉
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'num:%d/100' % num, (x + 30, y + 30), font, 1, (255, 0, 255), 4)

            # 超过指定最大保存数量结束程序
    if num > 20:
        break

    # 显示图像
    cv2.imshow("people", frame)
    c = cv2.waitKey(10)
    if c & 0xFF == ord('q'):
        break

        # 释放摄像头并销毁所有窗口
cap.release()
cv2.destroyAllWindows()
# 释放并销毁窗口

# videoCapture函数 bool videoCapture::open(string &filename) bool
# videoCapture::open(int device) filename–打开的视频文件名
# device–打开的食品捕获设备id，可以填0，1，2；
# CascadeClassifier 对象检测
# Haar特征分为边缘特征，线性特征，中心特征和对角线特征。
# Haar Cascade ：每个特征都应用于所有训练集图片中，对于每个特征，找出人脸图片分类效果最好的阈值。
# 显然，分类会有误分类，我们选择分类错误率最小的那些特征，也就是说这些特征可以最好的将人脸图片和非人脸图片区分开。

# frontalface：frontal正面的
# cvtColor：彩色空间转换
# classfier.detectMultiScale：可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、
# 大小（用矩形表示）void detectMultiScale( InputArray image,

# CV_OUT std::vector& objects,
# double scaleFactor = 1.1,
# int minNeighbors = 3, int flags = 0,
# Size minSize = Size(） Size maxSize = Size() );
# imwrite：输出图像到文件第一个参数const String& filename表示需要写入的文件名，第二个参数InputArray img图像数据。
# 第三个参数const std::vector& params表示为特定格式保存的参数编码，它有一个默认值std::vector< int >()，
# 所以一般情况下不用写。

# 9.rectangle：画出图像
# imshow：映射灰度
# putText ：显示文字
# waitKey：一个给定的时间内(单位ms)等待用户按键触发;如果用户没有按下 键,则接续等待(循环)


# ============================================================
# @Time    : 2019/5/5 0005   22:20
# @Author  : Nick
# @Email   : NickBarry@Gmail.com
# @File    : face_demo.py
# @Software: PyCharm
