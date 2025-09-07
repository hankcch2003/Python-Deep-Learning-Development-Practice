# 匯入numpy套件，用於數值計算與陣列處理
import numpy as np

# 匯入OpenCV套件，用於影像處理
import cv2

# 開啟預設攝影機(編號0表示第一個攝影機裝置)
camera = cv2.VideoCapture(0)

# 設定攝影機的亮度屬性
# 10代表亮度屬性(對應OpenCV的常數cv2.CAP_PROP_BRIGHTNESS)
# 200為欲設定的亮度值，實際範圍會因攝影機不同而有所差異
camera.set(10, 200)

# 設定ROI區域的起始與結束位置(以畫面比例表示)
cap_region_x_begin = 0.5  # ROI水平區域起始位置，佔畫面寬度的50%
cap_region_y_end = 0.8    # ROI垂直區域結束位置，佔畫面高度的80%

# 使用while迴圈持續讀取攝影機影像
while camera.isOpened():
    # 從攝影機讀取一幀影像，ret表示讀取成功與否，frame為影像陣列
    ret, frame = camera.read()

    # 使用雙邊濾波(bilateralFilter)平滑影像，保留邊緣並降低噪點
    # 參數：影像(frame)、濾波直徑 = 5、顏色標準差 = 50、座標標準差 = 100
    frame = cv2.bilateralFilter(frame, 5, 50, 100)

    # 將影像水平翻轉，常用於鏡像效果(自拍模式)
    # flipCode參數：1 = 水平翻轉，0 = 垂直翻轉，-1 = 水平+垂直翻轉
    frame = cv2.flip(frame, 1)

    # 在影像上畫出ROI(目標區域)矩形框
    # 左上角座標：(cap_region_x_begin * 寬度, 0)
    # 右下角座標：(影像寬度, cap_region_y_end * 高度)
    # 顏色：紅色(BGR: 0, 0, 255)，線寬：2
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (0, 0, 255), 2)

    # 顯示視窗名稱為'original'的影像視窗，內容為處理後的frame
    cv2.imshow('original', frame)

    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者有按下任何鍵(非255)，就跳出迴圈
    if key != 255:
        break

# 釋放攝影機資源並關閉所有OpenCV視窗
camera.release()
cv2.destroyAllWindows()