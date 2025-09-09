# 匯入OpenCV套件，用於影像處理
import cv2

# 匯入dlib函式庫，用於人臉偵測及臉部特徵點辨識
import dlib

# 匯入math模組，用於進行數學運算
import math

# 定義函式draw_text_info()，用於在畫面上顯示提示文字
def draw_text_info():
    # menu_pos_1：座標(10, 20)，用於設定第一行提示文字顯示的位置(x = 10, y = 20，畫面左上角)
    menu_pos_1 = (10, 20)

    # menu_pos_2：座標(10, 40)，用於設定第二行提示文字顯示的位置(位於第一行下方)
    menu_pos_2 = (10, 40)

    # 在影像畫面上顯示白色提示文字(BGR：255, 255, 255)，內容為"Use '1' to re-initialize tracking"
    # frame：目前影像幀，文字會繪製於該畫面上
    # menu_pos_1：文字顯示的位置座標(左上角起點)
    # FONT_HERSHEY_SIMPLEX：OpenCV提供的內建英文字型(簡單樣式字體)
    # 0.5：文字縮放比例，數值越大文字越大

    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    # 根據是否正在追蹤人臉，顯示不同狀態提示文字
    if tracking_face:
        # 追蹤中，顯示綠色提示文字(BGR：0, 255, 0)，內容為"tracking the face"
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        # 尚未追蹤，顯示紅色提示文字(BGR：0, 0, 255)，內容為"detecting a face to initialize tracking..."
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# 讀取影片檔案作為輸入來源，檔案路徑用原始字串避免跳脫字元錯誤
capture = cv2.VideoCapture(r'E:\Colab第三階段202505\ch22_20250902\hamilton_clip.mp4')

# 建立Dlib的正面人臉偵測器
detector = dlib.get_frontal_face_detector()

# 建立Dlib的相關追蹤器(correlation tracker)
tracker = dlib.correlation_tracker()

# 設定是否正在追蹤人臉的布林值，初始為False，表示目前尚未追蹤任何人臉
tracking_face = False

# 使用while迴圈持續讀取影片的每一幀影像
while True:
    # 從影片讀取一幀影像，ret表示讀取成功與否，frame為影像陣列
    ret, frame = capture.read()

    # 檢查是否成功讀取影片
    if not ret:
        print("影片讀取完畢或失敗，結束程式")
        break

    # 呼叫函式，在畫面上顯示操作提示文字，例如追蹤狀態與重新初始化提示
    draw_text_info()

    # 如果目前沒有追蹤目標，則執行人臉偵測以初始化追蹤
    if not tracking_face:
        # 將彩色影像轉為灰階影像，dlib偵測器需要灰階影像以提高效率與準確度
        # frame：原始彩色影像(BGR格式)
        # cv2.COLOR_BGR2GRAY：OpenCV色彩空間轉換參數，從BGR轉為灰階

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用Dlib偵測畫面中的人臉，傳回人臉位置的bounding boxes列表
        # gray：輸入的灰階影像
        # 0：表示使用預設的影像金字塔縮放層數(提升偵測效果的精細度)

        rects = detector(gray, 0)

        # 如果偵測到至少一張人臉，使用第一張人臉的矩形區域來初始化追蹤器
        # frame：目前的影像畫面，用於初始化追蹤
        # rects[0]：偵測到的第一張人臉的位置(dlib.rectangle物件)

        if len(rects) > 0:
            tracker.start_track(frame, rects[0])

            # 更新追蹤狀態為True，表示正在追蹤人臉
            tracking_face = True

    # 若目前正在追蹤人臉，使用目前影像更新追蹤器狀態並取得追蹤信心度(信心值越高代表追蹤效果越好)
    if tracking_face:
        confidence = tracker.update(frame)

        # 輸出追蹤信心度，方便觀察追蹤效果
        print(f"Tracking confidence: {confidence}")

        # 若信心值不是NaN且大於0，表示追蹤成功
        if not math.isnan(confidence) and confidence > 0:
            # 取得目前追蹤到的人臉位置
            pos = tracker.get_position()

            # 取出左上角與右下角座標，用於繪製追蹤框
            # coords：包含人臉追蹤框的四個邊界座標(left, top, right, bottom)
            # pos.left()：追蹤框左邊界的x座標
            # pos.top()：追蹤框上邊界的y座標
            # pos.right()：追蹤框右邊界的x座標
            # pos.bottom()：追蹤框下邊界的y座標

            coords = [pos.left(), pos.top(), pos.right(), pos.bottom()]

            # 確保所有座標都是有效數值(非NaN)
            if all(not math.isnan(c) for c in coords):
                # 在影像上繪製綠色(BGR：0, 255, 0)矩形框，標示追蹤到的人臉位置
                cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)
            else:
                # 若座標包含無效值(如NaN)，顯示警告訊息並停止追蹤
                print("偵測到無效座標，停止追蹤")
                tracking_face = False
        else:
            # 若追蹤信心度不足或更新失敗，也顯示警告訊息並停止追蹤
            print("追蹤失敗或信心度低，停止追蹤")
            tracking_face = False

    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者按下 "1" 鍵，重置追蹤狀態
    if key == ord("1"):
        tracking_face = False

    # 如果使用者按下 "q" 鍵，就跳出迴圈
    if key == ord("q"):
        break

    # 顯示已標記人臉的影像(frame)，視窗標題為"Face tracking using dlib frontal face detector and correlation filters for tracking"
    cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)

# 釋放影片資源並關閉所有OpenCV視窗
capture.release()
cv2.destroyAllWindows()