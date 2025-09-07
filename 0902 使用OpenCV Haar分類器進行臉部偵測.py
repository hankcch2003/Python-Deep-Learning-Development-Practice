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

# 定義show_detection函式，在圖片上畫出偵測到的臉部位置
def show_detection(image, faces):
    # 逐一處理每個偵測到的臉部框
    for item in faces:
        # 確保item是一維陣列(格式為x, y, w, h)
        if item.ndim != 1:
            # 若不是一維陣列，跳過此項
            continue

        # 取得人臉框的左上角座標與寬高
        x, y, w, h = item

        # 在圖片上畫藍色矩形框，線寬為5
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)

    # 傳回標記後的image圖片
    return image

# 設定兩個cascade XML檔案的完整路徑
cascade_alt2_path = r"E:\Colab第三階段202505\ch22_20250902\cascade_files\haarcascade_frontalface_alt2.xml"
cascade_default_path = r"E:\Colab第三階段202505\ch22_20250902\cascade_files\haarcascade_frontalface_default.xml"

# 載入兩個cascade檔案，利用臨時檔案避免中文路徑錯誤
cas_alt2 = load_cascade_from_chinese_path(cascade_alt2_path)
cas_default = load_cascade_from_chinese_path(cascade_default_path)

# 設定測試圖片的完整路徑
img_path = r"E:\Colab第三階段202505\ch22_20250902\test_face_detection.jpg"

# 以二進位方式讀取圖片檔案，並轉成numpy陣列
with open(img_path, 'rb') as f:
    img_bytes = np.frombuffer(f.read(), np.uint8)

# 使用OpenCV解碼圖片資料成為BGR彩色影像
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

# 將彩色圖片轉為灰階，Haar cascade偵測需灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用兩個cascade模型分別偵測臉部，輸出為多個矩形框(x, y, w, h)
faces_alt2 = cas_alt2.detectMultiScale(gray)
faces_default = cas_default.detectMultiScale(gray)

# 在複製的原圖上標記default cascade偵測到的臉部位置
img_faces_default = show_detection(img.copy(), faces_default)

# 顯示視窗名稱為"Face Detection"的圖片，內容為標記後的圖片
cv2.imshow("Face Detection", img_faces_default)

# 使用while迴圈持續等待使用者輸入
while True:
    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者有按下任何鍵(非255)，就跳出迴圈
    if key != 255:
        break

# 關閉所有OpenCV視窗
cv2.destroyAllWindows()