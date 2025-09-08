# 匯入numpy套件，用於數值計算與陣列處理
import numpy as np

# 匯入OpenCV套件，用於影像處理
import cv2

# 匯入tempfile套件，用於建立臨時檔案
import tempfile

# 定義load_cascade_from_chinese_path函式，解決中文路徑讀取cascade XML檔案問題
def load_cascade_from_chinese_path(chinese_path: str):
    # 以二進位模式讀取cascade XML檔案內容，並用with語句確保檔案能被正確關閉
    with open(chinese_path, 'rb') as f:
        cascade_bytes = f.read()

    # 建立一個臨時檔案，路徑為純英文，避免中文路徑造成的讀取問題
    with tempfile.NamedTemporaryFile(delete = False, suffix=".xml") as tmpfile:
        # 將剛剛讀取的檔案內容寫入臨時檔案
        tmpfile.write(cascade_bytes)

        # 取得臨時檔案的路徑
        tmp_cascade_path = tmpfile.name

    # 使用臨時檔案路徑建立CascadeClassifier物件
    cascade = cv2.CascadeClassifier(tmp_cascade_path)

    # 傳回CascadeClassifier物件
    return cascade

# 設定臉部cascade XML檔案的完整路徑
face_cascade_path = r"E:\Colab第三階段202505\ch22_20250902\cascade_files\haarcascade_frontalface_alt.xml"

# 載入臉部cascade檔案，利用臨時檔案避免中文路徑錯誤
face_cascade = load_cascade_from_chinese_path(face_cascade_path)

# 檢查是否成功載入cascade檔案
if face_cascade.empty():
    raise IOError("無法載入cascade檔案，請確認路徑是否正確")

# 開啟預設攝影機(編號0表示第一個攝影機裝置)
cap = cv2.VideoCapture(0)

# 設定縮放比例(預設0.5)，讓即時畫面不要太大
scaling_factor = 0.5

# 使用while迴圈持續讀取攝影機影像
while True:
    # 從攝影機讀取一幀影像，ret表示讀取成功與否，frame為影像陣列
    ret, frame = cap.read()

    # 檢查是否成功擷取影像
    if not ret:
        print("無法讀取影像，請確認路徑是否正確")
        break

    # 使用cv2.resize()函式縮放影像，根據指定的縮放比例調整大小
    # frame：目前從攝影機讀取到的彩色影像(numpy陣列)，形狀為(高度, 寬度, 色彩通道數)
    # None：表示不指定目標尺寸，目標大小由fx和fy決定
    # fx：寬度縮放比例，fy：高度縮放比例
    # interpolation：縮小影像時使用區域插值(cv2.INTER_AREA)，能保留較佳畫質

    frame = cv2.resize(frame, None, fx = scaling_factor, fy = scaling_factor, interpolation = cv2.INTER_AREA)

    # 將彩色影像轉為灰階，Haar cascade偵測需使用灰階影像以提高效率與準確度
    # frame：原始彩色影像(BGR格式)
    # cv2.COLOR_BGR2GRAY：OpenCV顏色空間轉換參數，從BGR轉為灰階

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用臉部cascade對灰階影像進行人臉偵測
    # gray：輸入的灰階影像
    # scaleFactor：每次影像縮小的比例(預設1.3)，表示每次縮小30%用來建立影像金字塔，越接近1越精細但越慢
    # minNeighbors：判定人臉的鄰近矩形數量(預設5)，數字越大偵測越嚴格(降低誤判)
    # 傳回值face_rects為偵測到人臉的矩形框列表，每個元素是(x, y, w, h)

    face_rects = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

    # 逐一畫出每個偵測到的臉部，並在畫面上以綠色矩形框標示臉部位置
    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # 顯示視窗名稱為'Face Detector'的影像視窗，內容為標記出人臉的影像
    cv2.imshow('Face Detector', frame)

    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者有按下任何鍵(非255)，就跳出迴圈
    if key != 255:
        break

# 釋放攝影機資源並關閉所有OpenCV視窗
cap.release()
cv2.destroyAllWindows()