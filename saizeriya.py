!pip install opencv-python
!pip install opencv-contrib-python
!pip install matplotlib
import cv2
import matplotlib.pyplot as plt 

#=============================
# パラメータ
#=============================
#画像１
img_path_1 = '14-16_1.png' 

#画像２
img_path_2 = '14-16_2.png'

#=============================
# 画像表示関数
#=============================
def show(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img, vmin = 0, vmax = 255)
    plt.show()
   plt.close()

#=============================
# 画像読み込み
#=============================
img_1 = cv2.imread(img_path_1)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
img_2 = cv2.imread(img_path_2)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB) 

#=============================
# 画像の差分
#=============================
#準備
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=2)

#差分マスクの計算
fgmask = fgbg.apply(img_1)
fgmask = fgbg.apply(img_2)
show(fgmask)

#画像１を暗くして差分マスクを重ねる
img_1 = img_1 // 4
img_1[fgmask==255] = (255, 0, 0)
show(img_1)