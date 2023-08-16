import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data
from skimage import data, color
import matplotlib.pyplot as plt
import random
import pandas as pd

def get_rgbhsv_from_img(pic_file):
    img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 按R,G,B三个通道分别计算颜色直方图
    r_hist = cv2.calcHist([img_bgr], [2], None, [25], [0, 255]).flatten().tolist()
    g_hist = cv2.calcHist([img_bgr], [1], None, [25], [0, 255]).flatten().tolist()
    b_hist = cv2.calcHist([img_bgr], [0], None, [25], [0, 255]).flatten().tolist()
    # 按H,S,V三个通道分别计算颜色直方图
    h_hist = cv2.calcHist([img_hsv], [0], None, [25], [0, 255]).flatten().tolist()
    s_hist = cv2.calcHist([img_hsv], [1], None, [25], [0, 255]).flatten().tolist()
    v_hist = cv2.calcHist([img_hsv], [2], None, [25], [0, 255]).flatten().tolist()

    return r_hist+g_hist+b_hist+h_hist+s_hist+v_hist

color_character = []
for img_root in [r'D:\server_cs\ukb_original',r'D:\Topic\images\Color']:
    img_list = random.sample(os.listdir(img_root),500)
    for image in img_list:
        img_path = os.path.join(img_root,image)
        color_character.append([image]+get_rgbhsv_from_img(img_path))

data = pd.DataFrame(color_character)
data.to_csv('./ColorCha_500.csv')

print(len(color_character[1]))
quit()




pic_file = r'D:\deep_learning\GAN\Stargan-v2\data\fundus\train\ukb\1000468_21015_0_0.png'
# m,dev = cv2.meanStdDev(img_hsv)  #计算H、V、S三通道的均值和方差

'''显示三个通道的颜色直方图'''
plt.plot(h_hist, label='H', color='blue')
plt.plot(s_hist, label='S', color='green')
plt.plot(v_hist, label='V', color='red')
plt.legend(loc='best')
plt.xlim([0, 256])
plt.show()
cv2.waitKey(0)



quit()



# cv2.namedWindow("input", cv2.WINDOW_GUI_NORMAL)
# cv2.imshow("input", img_hsv)

# 分别获取三个通道的ndarray数据
# img_h = img_hsv[:, :, 0]
# img_s = img_hsv[:, :, 1]
# img_v = img_hsv[:, :, 2]











plt.figure(num='wmu', figsize=(16, 8))  # 创建一个名为astronaut的窗口,并设置大小


pic_file = r'D:\deep_learning\GAN\Stargan-v2\data\fundus\train\wmu\1099_OS_32.jpg'
img_wmu = np.array(Image.open(pic_file))

plt.subplot(2, 7, 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
plt.title('wmu image')  # 第一幅图片标题
plt.imshow(img_wmu)  # 绘制第一幅图片

plt.subplot(2, 7, 2)  # 第二个子图
plt.title('R channel')  # 第二幅图片标题
plt.imshow(img_wmu[:, :, 0], plt.cm.gray)  # 绘制第二幅图片,且为灰度图
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 3)  # 第三个子图
plt.title('G channel')  # 第三幅图片标题
plt.imshow(img_wmu[:, :, 1], plt.cm.gray)  # 绘制第三幅图片,且为灰度图
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 4)  # 第四个子图
plt.title('B channel')  # 第四幅图片标题
plt.imshow(img_wmu[:, :, 2], plt.cm.gray)  # 绘制第四幅图片,且为灰度图
plt.axis('off')  # 不显示坐标尺寸

hsv_wmu = color.rgb2hsv(img_wmu)

plt.subplot(2, 7, 5)
plt.imshow(hsv_wmu[:, :, 0], cmap=plt.cm.gray)
plt.title("H channel")
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 6)
plt.imshow(hsv_wmu[:, :, 1], cmap=plt.cm.gray)
plt.title("S channel")
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 7)
plt.imshow(hsv_wmu[:, :, 2], cmap=plt.cm.gray)
plt.title("V channel")
plt.axis('off')  # 不显示坐标尺寸

# ukb
pic_file = r'D:\deep_learning\GAN\Stargan-v2\data\fundus\train\ukb\1004859_21016_0_0.png'
img_ukb = np.array(Image.open(pic_file))

plt.subplot(2, 7, 8)  # 将窗口分为两行两列四个子图，则可显示四幅图片
plt.title('ukb image')  # 第一幅图片标题
plt.imshow(img_ukb)  # 绘制第一幅图片

plt.subplot(2, 7, 9)  # 第二个子图
plt.title('R channel')  # 第二幅图片标题
plt.imshow(img_ukb[:, :, 0], plt.cm.gray)  # 绘制第二幅图片,且为灰度图
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 10)  # 第三个子图
plt.title('G channel')  # 第三幅图片标题
plt.imshow(img_ukb[:, :, 1], plt.cm.gray)  # 绘制第三幅图片,且为灰度图
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 11)  # 第四个子图
plt.title('B channel')  # 第四幅图片标题
plt.imshow(img_ukb[:, :, 2], plt.cm.gray)  # 绘制第四幅图片,且为灰度图
plt.axis('off')  # 不显示坐标尺寸

hsv_ukb = color.rgb2hsv(img_ukb)

plt.subplot(2, 7, 12)
plt.imshow(hsv_ukb[:, :, 0], cmap=plt.cm.gray)
plt.title("H channel")
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 13)
plt.imshow(hsv_ukb[:, :, 1], cmap=plt.cm.gray)
plt.title("S channel")
plt.axis('off')  # 不显示坐标尺寸

plt.subplot(2, 7, 14)
plt.imshow(hsv_ukb[:, :, 2], cmap=plt.cm.gray)
plt.title("V channel")
plt.axis('off')  # 不显示坐标尺寸

plt.show()  # 显示窗口


