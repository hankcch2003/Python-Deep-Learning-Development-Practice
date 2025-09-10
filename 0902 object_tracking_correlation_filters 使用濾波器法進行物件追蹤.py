# 匯入OpenCV套件，用於影像處理
import cv2

# 匯入dlib函式庫，用於人臉偵測及臉部特徵點辨識，也包含物件追蹤演算法
import dlib

# 定義函式draw_text_info()，用於在畫面上顯示提示文字
def draw_text_info():
    # menu_pos_1：座標(10, 20)，用於設定第一行提示文字顯示的位置(x = 10, y = 20，畫面左上角)
    menu_pos_1 = (10, 20)

    # menu_pos_2：座標(10, 40)，用於設定第二行提示文字顯示的位置(位於第一行下方)
    menu_pos_2 = (10, 40)

    # menu_pos_3：座標(10, 60)，用於設定第三行提示文字顯示的位置(位於第二行下方)
    menu_pos_3 = (10, 60)
    
    # 在影像畫面上顯示白色提示文字(BGR：255, 255, 255)，內容為"Use left click of the mouse to select the object to track"
    info_1 = "Use left click of the mouse to select the object to track"

    # 在影像畫面上顯示白色提示文字(BGR：255, 255, 255)，內容為"Use '1' to start tracking, '2' to reset tracking and 'q' to exit"
    info_2 = "Use '1' to start tracking, '2' to reset tracking and 'q' to exit"

    # frame：目前影像幀，文字會繪製於該畫面上
    # menu_pos_1：文字顯示的位置座標(左上角起點)
    # menu_pos_2：文字顯示的位置座標(左上角起點，通常y座標比menu_pos_1大，避免文字重疊)
    # FONT_HERSHEY_SIMPLEX：OpenCV提供的內建英文字型(簡單樣式字體)
    # 0.5：文字縮放比例，數值越大文字越大

    cv2.putText(frame, info_1, menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, info_2, menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    
    # 若目前正在追蹤，顯示追蹤狀態文字
    # tracking_state：布林值，表示是否正在追蹤中
    # menu_pos_3：文字顯示的位置座標(左上角起點，通常y座標比menu_pos_2大，避免文字重疊)
    # (0, 255, 0)：綠色，表示正在追蹤的狀態文字
    # (0, 0, 255)：紅色，表示未追蹤的狀態文字

    if tracking_state:
        cv2.putText(frame, "tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "not tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# 用於儲存滑鼠選取的兩個點，分別是物件邊界框的左上角和右下角座標
points = []

# 定義函式mouse_event_handler，用於記錄使用者框選物件的兩個角點(左上與右下)
# event：滑鼠事件類型，例如左鍵按下、放開、滑動等
# x, y：滑鼠事件發生時的座標位置
# flags：事件相關的輔助鍵狀態，例如Ctrl、Shift等(此處未使用)
# param：額外參數(此處未使用)
# global points：使用全域變數points，儲存使用者框選的兩個座標點

def mouse_event_handler(event, x, y, flags, param):
    global points

    # 當滑鼠左鍵按下，記錄第一個點(左上角)
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]

    # 當滑鼠左鍵放開，記錄第二個點(右下角)
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))

# 開啟預設攝影機(編號0表示第一個攝影機裝置)
capture = cv2.VideoCapture(0)

# 設定視窗名稱為"Object tracking using dlib correlation filter algorithm"
window_name = "Object tracking using dlib correlation filter algorithm"

# 建立影像顯示視窗，名稱為window_name所指定的字串
cv2.namedWindow(window_name)

# 設定滑鼠事件的回呼函式，當使用者在視窗中操作滑鼠時會觸發mouse_event_handler
cv2.setMouseCallback(window_name, mouse_event_handler)

# 建立dlib的相關性追蹤器(correlation tracker)物件
tracker = dlib.correlation_tracker()

# 設定初始追蹤狀態為False，表示尚未開始追蹤
tracking_state = False

# 使用while迴圈持續從攝影機擷取每一幀影像
while True:
    # 從影片讀取一幀影像，ret表示讀取成功與否，frame為影像陣列
    ret, frame = capture.read()

    # 在畫面上繪製操作說明文字(如快捷鍵提示與追蹤狀態)
    draw_text_info()

    # 如果滑鼠已選取兩個點，畫出矩形框並建立dlib的矩形物件
    if len(points) == 2:
        # 在影像上繪製紅色(BGR：0, 0, 255)矩形框，並設定線寬為3，標示選取的物件位置
        # points[0]：第一個點(左上角)的(x, y)座標
        # points[1]：第二個點(右下角)的(x, y)座標
        # points[0][0]：第一個點的x座標
        # points[0][1]：第一個點的y座標
        # points[1][0]：第二個點的x座標
        # points[1][1]：第二個點的y座標

        cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 3)

        # 建立dlib用的矩形格式，方便追蹤器使用
        dlib_rectangle = dlib.rectangle(points[0][0], points[0][1], points[1][0], points[1][1])

    # 如果追蹤狀態為True，更新追蹤器並畫出追蹤框(綠色)
    if tracking_state == True:
        # 更新追蹤器，讓它追蹤新影像中的物件
        tracker.update(frame)

        # 取得目前追蹤的物件位置
        pos = tracker.get_position()

        # 在影像上繪製綠色(BGR：0, 255, 0)矩形框，並設定線寬為3，標示選取的物件位置
        # frames：當前讀取的影像幀，用於繪製追蹤框與顯示畫面
        # pos.left()：追蹤框左邊界的x座標
        # pos.top()：追蹤框上邊界的y座標
        # pos.right()：追蹤框右邊界的x座標
        # pos.bottom()：追蹤框下邊界的y座標

        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)
    
    # 等待鍵盤輸入，1毫秒刷新一次畫面，並與0xFF做AND運算確保相容性
    key = cv2.waitKey(1) & 0xFF

    # 如果使用者按下 "1" 鍵，且已選取兩個點，開始追蹤
    if key == ord("1"):
        if len(points) == 2:
            # 用選取的矩形初始化追蹤器
            # frame：當前影像幀，追蹤器根據此影像進行追蹤
            # dlib_rectangle：初始化追蹤目標的矩形框(左上角與右下角座標)

            tracker.start_track(frame, dlib_rectangle)

            # 設定追蹤狀態為 True，表示開始追蹤
            tracking_state = True

            # 清除已選取的點，避免重複使用
            points = []

    # 如果使用者按下 "2" 鍵，重置追蹤狀態和選取點
    if key == ord("2"):
        points = []
        tracking_state = False

    # 如果使用者按下 "q" 鍵，就跳出迴圈
    if key == ord('q'):
        break

    # 顯示已標記追蹤目標的影像(frame)，視窗標題為window_name
    cv2.imshow(window_name, frame)

# 釋放攝影機資源並關閉所有OpenCV視窗
capture.release()
cv2.destroyAllWindows()