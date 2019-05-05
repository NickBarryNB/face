#!/usr/bin/python anaconda
# -*- coding:utf-8 -*-
# ===================================================加载图片
# # 1.导入库
# import cv2
#
# # 2.加载图片
# img = cv2.imread(r"C:\Users\Administrator\Desktop\huge.jpg")
#
# # 3.创建窗口
# cv2.namedWindow("Nick window")
#
# # 4.显示图片
# cv2.imshow("nick", img)
# # 5.暂停窗口
# cv2.waitKey(0)
# # 6.关闭窗口
# cv2.destroyAllWindows()
# =================================================================识别图片中的人脸并标记
# # 1.导入库
# import cv2
#
# # 2.加载图片
# img = cv2.imread(r"C:\Users\Administrator\Desktop\huge.jpg")
#
# # 3.加载人脸模型
# face = cv2.CascadeClassifier("F:\Anaconda3_5.2.0\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
#
# # 4.调整图片灰度
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#
# # 5.检查人脸
# faces = face.detectMultiScale(gray)
#
# # 6.标记人脸
# for (x,y,w,h) in faces:
#     # # 四个参数，1，写图片  2，坐标原点  3，识别大小  4，颜色  5，线宽
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
#
# # 7.创建窗口
#     cv2.namedWindow("Nick")
#
# # 8.显示图片
#     cv2.imshow("nick", img)
#
# # 9.暂停窗口
#     cv2.waitKey(0)
# # 10.关闭窗口
#     cv2.destroyAllWindow()
# ===============================================================打开摄像头
# # 1.导入库
# import cv2

# # 2.打开摄像头
# capture = cv2.VideoCapture(0)

# # 3.获取实时画面
# cv2.namedWindow("Nick")   # 创建窗口
# while True:
#     # 3.1 读取摄像头帧画面
#     ret, frame = capture.read()
#     # 3.2显示图片（渲染画面）
#     cv2.imshow("nick", frame)
#     # 3.3暂停窗口
#     if cv2.waitKey(5) & 0xFF == ord("q"):
#         break
#
# # 4.释放资源
# capture.release()

# # 5.关闭窗口
# cv2.destroyAllWindows()
# ==================================================================实时标记摄像头的人脸
# 1.导入库
import cv2
# 2.加载人脸模型
face = cv2.CascadeClassifier("F:\Anaconda3_5.2.0\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
# 3.打开摄像头
capture = cv2.VideoCapture(0)
# 4.创建窗口
cv2.namedWindow("Nick")
# 5.获取实时画面
while True:
        # 5.1 读取摄像头帧画面
        ret, frame = capture.read()
        # 5.2 图片灰度调整
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 5.3 检查人脸
        faces = face.detectMultiScale(gray, 1.1, 3, 0, (100, 100))
        # 5.4 标记人脸
        for(x, y, w, h) in faces:
            # 有四个参数  1，写图片  2，坐标原点  3，识别大小  4，颜色  5，线宽
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # 5.5 显示图片
            cv2.imshow("nick", frame)
        # 5.6 暂停窗口
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

# 6.释放资源
capture.release()

# 7.关闭窗口
cv2.destroyAllWindows()
# ===================================================================
# @Time    : 2019/5/5 0005   22:54
# @Author  : Nick
# @Email   : NickBarry@Gmail.com
# @File    : face_demo1.py
# @Software: PyCharm
