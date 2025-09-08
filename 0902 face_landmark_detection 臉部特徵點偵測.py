# 匯入sys模組，用於與系統互動
import sys

# 匯入os模組，用於與作業系統互動
import os

# 匯入OpenCV套件，用於影像處理
import cv2

# 匯入glob模組，用於取得符合特定模式的檔案列表(例如取得所有jpg檔)
import glob

# 匯入dlib函式庫，用於人臉偵測及臉部特徵點辨識
import dlib

# 檢查命令列參數數量是否正確，必須提供兩個參數：
# 1.預訓練臉部特徵點模型檔案路徑、2.臉部圖片資料夾路徑

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

    # 如果參數錯誤，終止程式執行
    exit()

# 讀取命令列參數：模型路徑與臉部圖片資料夾路徑
predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

# 建立dlib提供的人臉偵測器(HOG特徵+SVM分類器)
detector = dlib.get_frontal_face_detector()

# 載入預先訓練好的臉部特徵點偵測模型
predictor = dlib.shape_predictor(predictor_path)

# 使用glob.glob()取得指定資料夾中所有副檔名為.jpg的影像檔案路徑清單
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # 印出目前正在處理的圖片檔案名稱(包含完整路徑)
    print("Processing file: {}".format(f))

    # 用OpenCV的imread()函數讀取圖片，讀取後的影像格式為BGR(藍綠紅)
    img_bgr = cv2.imread(f)

    # 如果無法讀取圖片(cv2.imread()傳回None)，則輸出錯誤訊息並跳過該檔案
    if img_bgr is None:
        print(f"Cannot read image: {f}")
        continue
    
    # 將BGR格式轉為RGB格式，因為dlib需要使用RGB圖片
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 使用dlib偵測器找出圖片中所有人臉位置
    # 參數1表示圖片金字塔數量，數字越大偵測精度越高但速度較慢

    dets = detector(img_rgb, 1)

    # 輸出偵測到的人臉數量，len(dets)是偵測結果的清單長度
    print("Number of faces detected: {}".format(len(dets)))

    # 針對每一張偵測到的人臉執行特徵點偵測
    for k, d in enumerate(dets):
       # 輸出該人臉位置座標
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        
        # 使用預訓練模型取得68個臉部特徵點位置
        shape = predictor(img_rgb, d)
        
        # 在原始BGR圖片上畫出每個特徵點(用綠色小圓點標示)
        # 迭代臉部特徵點索引範圍(0到67，共68個點)

        for i in range(68):
            # 取得第i個特徵點的x座標
            x = shape.part(i).x

            # 取得第i個特徵點的y座標
            y = shape.part(i).y

            # 在原始的BGR圖片上，以(x, y)座標為中心畫一個半徑為2像素的綠色實心圓點
            # img_bgr：要繪製的圖片
            # (x, y)：圓心座標(臉部特徵點的位置)
            # 2：圓的半徑(2像素)
            # (0, 255, 0)：顏色，這裡為綠色(BGR格式)
            # -1：表示填滿圓形(實心圓)

            cv2.circle(img_bgr, (x, y), 2, (0, 255, 0), -1)

    # 顯示視窗名稱為"Face Landmarks"的影像視窗，內容為標記出臉部特徵點的圖片
    cv2.imshow("Face Landmarks", img_bgr)

    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者有按下任何鍵(非255)，就跳出迴圈
    if key != 255:
        break

# 關閉所有OpenCV視窗
cv2.destroyAllWindows()