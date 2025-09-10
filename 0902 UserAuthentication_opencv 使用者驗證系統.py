# 匯入numpy套件，用於數值計算與陣列處理
import numpy as np

# 匯入os模組，用於與作業系統互動
import os

# 匯入OpenCV套件，用於影像處理
import cv2

# 匯入tempfile套件，用於建立臨時檔案
import tempfile

# 定義imread_via_tempfile函式，解決中文路徑讀取圖片問題
def imread_via_tempfile(img_path):
    # 以二進位模式開啟圖片檔案，讀取內容，並用with語句確保檔案正確關閉
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    # 建立臨時檔案，路徑為純英文，避免中文路徑導致讀取失敗
    with tempfile.NamedTemporaryFile(delete = False, suffix=".png") as tmpfile:
        # 將圖片內容寫入臨時檔案
        tmpfile.write(img_bytes)

        # 取得臨時檔案路徑
        tmp_path = tmpfile.name
    
    # 以灰階模式讀取臨時圖片檔案
    img = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)

    # 傳回圖片的numpy陣列
    return img

# 定義detect_and_crop_face函式，輸入圖片與人臉分類器，並傳回裁切並調整大小後的人臉影像
# img：輸入的影像資料(通常為numpy陣列格式)
# face_cascade：OpenCV的CascadeClassifier物件，用來偵測人臉

def detect_and_crop_face(img, face_cascade):
    faces = face_cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5)

    # 如果未偵測到任何人臉，則傳回None
    if len(faces) == 0:
        return None
    
    # 取得第一張偵測到的人臉位置座標(x, y)及寬度、高度(w, h)
    x, y, w, h = faces[0]

    # 根據偵測到的位置從原始影像中裁切出人臉區域
    face = img[y: y + h, x: x + w]

    # 將裁切出的人臉影像縮放為200x200的尺寸
    face_resized = cv2.resize(face, (200, 200))

    # 傳回處理後的人臉影像
    return face_resized

# 載入人臉偵測分類器(Haar cascade)，用來偵測人臉
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 判斷分類器是否成功載入，若失敗則印出錯誤訊息並結束程式
if face_cascade.empty():
    print("無法載入 Haar cascade，請確認 OpenCV 是否完整安裝")
    exit()

# 設定圖片資料夾路徑，存放訓練與測試圖片
Image_Path = r"E:\Colab第三階段202505\ch22_20250902"

# 訓練圖片檔名列表，包含兩組不同類別
train_image_files = ["a1.png", "a2.png", "b1.png", "b2.png"]

# 對應訓練圖片的標籤，a組為0，b組為1
labels = [0, 0, 1, 1]

# 標籤名稱，用於顯示預測結果
images_label = ["a", "b"]

# 儲存裁切後的人臉影像
images = []

# 逐一處理訓練圖片檔名列表中的圖片
for img_file in train_image_files:
    # 組合資料夾路徑與檔名，取得完整圖片路徑
    img_path = os.path.join(Image_Path, img_file)
    
    # 使用自訂函式讀取圖片(支援中文路徑)
    img = imread_via_tempfile(img_path)
    
    # 如果無法讀取圖片，輸出錯誤訊息並跳過該圖片
    if img is None:
        print(f"無法讀取訓練圖片：{img_path}")
        continue
    
    # 使用人臉偵測函式，從圖片中裁切出人臉區域
    face = detect_and_crop_face(img, face_cascade)
    
    # 如果沒偵測到人臉，輸出錯誤訊息並跳過該圖片
    if face is None:
        print(f"訓練圖片無臉部偵測到：{img_path}")
        continue
    
    # 將成功裁切出的人臉影像加入訓練資料清單
    images.append(face)

# 檢查訓練圖片數量與標籤數量是否一致，若不一致則輸出錯誤訊息並終止程式
if len(images) != len(labels):
    print("訓練圖片與標籤數量不符，請檢查圖片與臉部偵測結果")
    exit()

# 使用cv2.face.LBPHFaceRecognizer_create()函式建立LBPH人臉辨識器物件recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 使用裁切好的人臉影像與標籤訓練LBPH辨識器
recognizer.train(images, np.array(labels))

# 測試圖片檔名列表，包含多張待辨識的圖片
test_images = ["a1.png", "a2.png", "a3.png", "b1.png", "b2.png"]

# 依序對測試圖片進行辨識
for test_img_name in test_images:
    # 產生測試圖片的完整路徑並讀取圖片
    predict_path = os.path.join(Image_Path, test_img_name)
    
    # 使用自訂函式讀取測試圖片
    predict_img = imread_via_tempfile(predict_path)
    
    # 若無法讀取圖片，輸出錯誤訊息並跳過該圖片
    if predict_img is None:
        print(f"無法讀取測試圖片：{predict_path}")
        continue

    # 偵測並裁切人臉影像，失敗回傳None
    predict_face = detect_and_crop_face(predict_img, face_cascade)

    # 若無法偵測到人臉，輸出錯誤訊息並跳過該圖片
    if predict_face is None:
        print(f"測試圖片無法偵測到臉部：{predict_path}")
        continue

    # 使用辨識器對裁切後的人臉進行預測，傳回標籤與信心度
    label, confidence = recognizer.predict(predict_face)

    # 輸出測試圖片名稱、預測標籤名稱與信心度
    print(f"測試圖片：{test_img_name}，預測結果：{images_label[label]}，信心度：{confidence:.2f}")