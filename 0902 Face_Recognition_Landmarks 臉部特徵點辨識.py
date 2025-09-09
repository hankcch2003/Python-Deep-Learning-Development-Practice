# 匯入PIL庫中的Image和ImageDraw模組，用於影像處理與繪圖
from PIL import Image, ImageDraw

# 匯入face_recognition庫，用於臉部辨識與特徵點偵測
import face_recognition

# 載入指定路徑的影像檔案，face_recognition會將圖片讀取成numpy陣列格式，方便後續影像處理和分析
image = face_recognition.load_image_file(r"E:\Colab第三階段202505\ch22_20250902\2008_001322.jpg")

# 使用face_recognition的face_landmarks函式偵測影像中所有人臉特徵點，傳回列表，列表中每個元素是包含臉部各特徵點座標的字典
FaceLandmarksList = face_recognition.face_landmarks(image)

# 使用len(FaceLandmarksList)計算偵測到的臉部數量，並格式化輸出結果
print("Number {} face(s) recognized in this image.".format(len(FaceLandmarksList)))

# 使用Image.fromarray()方法將NumPy陣列格式的影像轉換成PIL Image物件，方便後續繪圖操作
PilImage = Image.fromarray(image)

# 使用ImageDraw.Draw()方法建立可在PIL影像上繪圖的物件
DrawPilImage = ImageDraw.Draw(PilImage)

# 針對影像中每一張偵測到的臉，face_landmarks是一個字典，包含該臉所有特徵點的座標
for face_landmarks in FaceLandmarksList:
    # 格式化每個臉部特徵點名稱(如'left_eye', 'nose_tip'等)及其對應的座標列表
    # facial_feature：臉部特徵名稱，用來索引對應的座標列表
    # face_landmarks.keys()：傳回所有臉部特徵名稱的列表

    for facial_feature in face_landmarks.keys():
        print("{} points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # 繪製該臉部所有特徵點連線，使用DrawPilImage物件將各特徵點座標依序連接
    # width = 5：設定線寬為5像素，使線條更粗、更容易辨識

    for facial_feature in face_landmarks.keys():
        DrawPilImage.line(face_landmarks[facial_feature], width = 5)

# 顯示繪製完特徵點連線的影像視窗
PilImage.show()