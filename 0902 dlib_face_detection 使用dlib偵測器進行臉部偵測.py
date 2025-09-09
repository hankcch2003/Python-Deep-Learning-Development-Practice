# 匯入numpy套件，用於數值計算與陣列處理
import numpy as np

# 匯入time模組，用於時間相關操作
import time

# 匯入OpenCV套件，用於影像處理
import cv2

# 匯入dlib函式庫，用於人臉偵測及臉部特徵點辨識
import dlib

# 使用dlib內建的正面臉部偵測器
detector = dlib.get_frontal_face_detector()

# 開啟預設攝影機(編號0表示第一個攝影機裝置)
cap = cv2.VideoCapture(0)

# 設定目標每秒影格數(fps)，決定畫面更新頻率
target_fps = 2

# 根據目標fps計算每兩影格間的時間間隔(秒)
frame_interval = 1.0 / target_fps

# 用來記錄上一次輸出影像訊息和臉部偵測訊息的時間
last_print_img_time = 0
last_print_face_time = 0

# 使用while迴圈持續讀取攝影機影像
while True:
    # 紀錄每次迴圈開始的時間，用來計算處理時間與幀率控制
    start_time = time.time()

    # 從攝影機讀取一幀影像，ret表示讀取成功與否，image為影像陣列
    ret, image = cap.read()

    # 檢查是否成功擷取影像
    if not ret or image is None:
        print("無法讀取影像，請確認攝影機是否正常連接")
        break

    # 取得影像的高度與寬度，image.shape[:2]傳回(height, width)
    height, width = image.shape[:2]

    # 計算縮放比例，將影像寬度縮小到320像素，並依比例縮放高度以維持長寬比
    scale = 320 / width

    # 使用cv2.resize()方法重新調整影像大小，尺寸為(寬度 = 320, 高度 = height * scale)
    image_resized = cv2.resize(image, (320, int(height * scale)))

    # 取得目前時間，用於控制輸出頻率
    current_time = time.time()

    # 如果目前時間與上次輸出臉部數量的時間差大於1秒
    if current_time - last_print_img_time > 1:
        # 輸出目前影像的資料型態(dtype)與形狀(shape)
        print(f"image dtype: {image_resized.dtype}, shape: {image_resized.shape}")

        # 更新上次輸出時間為目前時間，避免訊息重複頻繁輸出
        last_print_img_time = current_time

    # 確認影像格式是否為uint8
    if image_resized.dtype != np.uint8:
        print("影像格式錯誤，必須是uint8")
        break

    # 將彩色影像轉為灰階影像，dlib偵測器需要灰階影像以提高效率與準確度
    # image_resized：原始彩色影像(BGR格式)
    # cv2.COLOR_BGR2GRAY：OpenCV色彩空間轉換參數，從BGR轉為灰階

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    try:
        # 使用dlib偵測器偵測臉部，第二個參數0表示不使用影像金字塔升採樣，偵測速度較快但可能較不精細
        rects = detector(gray, 0)

        # 如果目前時間與上次輸出臉部數量的時間差大於2秒
        if current_time - last_print_face_time > 2:
            # 輸出目前影像中偵測到的臉部數量(len(rects))
            print(f"偵測到 {len(rects)} 張臉")

            # 更新上次輸出時間為目前時間，避免訊息重複頻繁輸出
            last_print_face_time = current_time
    
    # 捕捉偵測器可能出現的錯誤
    except RuntimeError as e:
        print("偵測器錯誤:", e)
        break

    # 如果偵測到至少一張臉，取得第一張臉的矩形區域
    if len(rects) > 0:
        rect = rects[0]

        try:
            # 在影像上繪製綠色矩形方框來標示臉部位置，並設定線寬為2
            # image_resized：要繪製的影像
            # rect.left(), rect.top()：左上角座標
            # rect.right(), rect.bottom()：右下角座標

            cv2.rectangle(image_resized, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        
        # 捕捉繪製過程中可能發生的錯誤
        except Exception as e:
            print(f"畫框失敗: {e}")

    # 顯示視窗名稱為'Output'的影像視窗，內容為標記出人臉的影像
    # image_resized：要顯示的影像(已標記人臉)

    cv2.imshow("Output", image_resized)

    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者有按下任何鍵(非255)，就跳出迴圈
    if key != 255:
        break

    # 計算處理這一幀所花費的時間，等於目前時間減去開始時間
    elapsed = time.time() - start_time

    # 若處理時間少於幀間隔，暫停一段時間以維持目標FPS
    # elapsed：本幀處理所花費的時間(秒)
    # frame_interval：兩幀之間的目標時間間隔(秒)，通常為1除以目標FPS

    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)

# 釋放攝影機資源並關閉所有OpenCV視窗
cap.release()
cv2.destroyAllWindows()