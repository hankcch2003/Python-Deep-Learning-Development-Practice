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
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".xml") as tmpfile:
        # 將剛剛讀取的檔案內容寫入臨時檔案
        tmpfile.write(cascade_bytes)

        # 取得臨時檔案的路徑
        tmp_cascade_path = tmpfile.name

    # 使用臨時檔案路徑建立CascadeClassifier物件
    cascade = cv2.CascadeClassifier(tmp_cascade_path)

    # 傳回CascadeClassifier物件
    return cascade

# 設定三個cascade XML檔案的完整路徑
face_cascade_path = r"E:\Colab第三階段202505\ch22_20250902\cascade_files\haarcascade_frontalface_default.xml"
eye_cascade_path = r"E:\Colab第三階段202505\ch22_20250902\cascade_files\haarcascade_eye.xml"
smile_cascade_path = r"E:\Colab第三階段202505\ch22_20250902\cascade_files\haarcascade_smile.xml"

# 載入三個cascade檔案，利用臨時檔案避免中文路徑錯誤
face_cascade = load_cascade_from_chinese_path(face_cascade_path)
eye_cascade = load_cascade_from_chinese_path(eye_cascade_path)
smile_cascade = load_cascade_from_chinese_path(smile_cascade_path)

# 設定測試圖片的完整路徑
img_path = r"E:\Colab第三階段202505\ch22_20250902\lena.jpg"

# 以二進位方式讀取圖片檔案，並轉成numpy陣列
with open(img_path, 'rb') as f:
    img_bytes = np.frombuffer(f.read(), np.uint8)

# 使用OpenCV解碼圖片資料成為BGR彩色影像
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

# 將彩色圖片轉為灰階，Haar cascade偵測需灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用臉部cascade偵測臉部位置，輸出為多個矩形框(x, y, w, h)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 逐一處理每個偵測到的人臉
for (x, y, w, h) in faces:
    # 在圖片上畫藍色矩形框表示人臉
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 擷取人臉區域的灰階與彩色影像
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = img[y: y + h, x: x + w]

    # 偵測眼睛，並用綠色矩形框標記
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # 逐一畫出每個偵測到的眼睛
    for (ex, ey, ew, eh) in eyes:
        # 在圖片上畫綠色矩形框表示眼睛
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # 偵測笑容，只畫出第一個偵測到的笑容，並用紅色矩形框標記
    smile = smile_cascade.detectMultiScale(roi_gray)

    # 如果偵測到笑容，在圖片上畫紅色矩形框表示第一個笑容
    if len(smile) > 0:
        sx, sy, sw, sh = smile[0]
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

# 顯示視窗名稱為"img"的圖片，內容為標記後的圖片
cv2.imshow('img', img)

# 使用while迴圈持續等待使用者輸入
while True:
    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者有按下任何鍵(非255)，就跳出迴圈
    if key != 255:
        break

# 關閉所有OpenCV視窗
cv2.destroyAllWindows()