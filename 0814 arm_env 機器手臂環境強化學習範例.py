"""
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet

class ArmEnv(object):
    # 動作範圍上下界
    action_bound = [-1, 1]

    # 動作維度(兩個關節角度變化)
    action_dim = 2

    # 狀態維度(手臂狀態相關)
    state_dim = 7
    dt = .1                       # 每次step的時間間隔(刷新率)
    arm1l = 100                   # 第一節手臂長度
    arm2l = 100                   # 第二節手臂長度
    viewer = None                 # 顯示視窗物件
    viewer_xy = (400, 400)        # 視窗大小
    get_point = False             # 是否成功抓取目標點
    mouse_in = np.array([False])  # 滑鼠是否在視窗內(用於互動)
    point_l = 15                  # 目標點半徑大小
    grab_counter = 0              # 計數器，判斷是否持續接觸目標點

    def __init__(self, mode='easy'):
        self.mode = mode
        self.arm_info = np.zeros((2, 4))                  # 紀錄手臂長度、角度、末端座標(x,y)
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        self.point_info = np.array([250, 303])            # 目標點座標
        self.point_info_init = self.point_info.copy()     # 初始目標點位置備份
        self.center_coord = np.array(self.viewer_xy) / 2  # 視窗中心點座標

    def step(self, action):
        # 將動作限制在動作界限內
        action = np.clip(action, *self.action_bound)

        # 依動作更新兩節手臂角度(角度隨時間微調)
        self.arm_info[:, 1] += action * self.dt

        # 角度模2π確保範圍在0~2π
        self.arm_info[:, 1] %= np.pi * 2

        # 取兩節手臂角度
        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]

        # 計算第一節手臂末端座標相對於中心的偏移量(x, y)
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        # 第二節手臂末端相對於第一節末端的偏移量(x, y)
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        # 計算第一節手臂末端絕對座標 = 中心座標 + 偏移
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy      # (x1, y1)
        # 第二節手臂末端絕對座標 = 第一節末端 + 第二節偏移
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

        # 取得當前狀態及第二節手臂末端到目標點的距離向量
        s, arm2_distance = self._get_state()

        # 計算本step獎勵
        r = self._r_func(arm2_distance)
        return s, r, self.get_point

    def reset(self):
        # 重置狀態，重置是否抓取到目標點
        self.get_point = False
        self.grab_counter = 0

        if self.mode == 'hard':
            # hard模式下，目標點隨機生成在指定區域(100~300範圍內)
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = pxy
        else:
            # easy模式，隨機初始化手臂角度
            arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
            self.arm_info[0, 1] = arm1rad
            self.arm_info[1, 1] = arm2rad

            # 計算手臂末端座標
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
            arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy      # (x1, y1)
            self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

            # 目標點回復至初始位置
            self.point_info[:] = self.point_info_init
        return self._get_state()[0]

    def render(self):
        # 若視窗尚未建立，建立Viewer視窗物件
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in)

        # 呼叫Viewer的render方法畫面更新
        self.viewer.render()

    def sample_action(self):
        # 隨機採樣動作
        return np.random.uniform(*self.action_bound, size = self.action_dim)

    def set_fps(self, fps = 30):
        # pyglet沒有set_fps_limit，使用schedule_interval定時呼叫_dummy維持固定FPS
        pyglet.clock.unschedule(self._dummy)
        pyglet.clock.schedule_interval(self._dummy, 1.0 / fps)

    def _dummy(self, dt):
        # 空函數用於定時觸發FPS控制
        pass

    def _get_state(self):
        # 取得手臂末端座標
        arm_end = self.arm_info[:, 2:4]

        # 手臂末端相對目標點的向量 (x1 - px, y1 - py, x2 - px, y2 - py)
        t_arms = np.ravel(arm_end - self.point_info)

        # 中心點相對目標點距離，作正規化(除以200)
        center_dis = (self.center_coord - self.point_info) / 200

        # 是否正在接觸目標點(grab_counter > 0 即接觸中)
        in_point = 1 if self.grab_counter > 0 else 0

        # 回傳狀態向量與第二節手臂末端相對目標點的距離向量
        return np.hstack([in_point, t_arms / 200, center_dis]), t_arms[-2:]

    def _r_func(self, distance):
        # 設定抓取持續時間閾值t
        t = 50

        # 計算距離向量的絕對距離
        abs_distance = np.sqrt(np.sum(np.square(distance)))

        # 獎勵值以距離負比率給予(距離越近獎勵越高)
        r = -abs_distance / 200

        # 若距離小於目標點半徑且尚未抓取成功，增加獎勵並計數持續抓取時間
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1

            # 持續抓取超過閾值，給予大量獎勵，設定抓取成功狀態
            if self.grab_counter > t:
                r += 10.
                self.get_point = True

        # 距離大於目標點半徑時，重置抓取計數器與狀態
        elif abs_distance > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r

class Viewer(pyglet.window.Window):
    color = {'background': [1] * 3 + [1]}  # 白色背景RGBA
    bar_thc = 5                            # 手臂粗細像素值

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in):
        super(Viewer, self).__init__(width, height, resizable = False, caption='Arm', vsync = False)
        self.set_location(x = 80, y = 10)                  # 設定視窗初始位置
        pyglet.gl.glClearColor(*self.color['background'])  # 設定背景色

        self.arm_info = arm_info      # 手臂狀態
        self.point_info = point_info  # 目標點座標
        self.mouse_in = mouse_in      # 滑鼠是否在視窗內
        self.point_l = point_l        # 目標點大小

        # 視窗中心點座標
        self.center_coord = np.array((min(width, height) / 2,) * 2)

        # pyglet批次繪製物件
        self.batch = pyglet.graphics.Batch()

        # 初始化頂點座標(空)
        arm1_box, arm2_box, point_box = [0] * 8, [0] * 8, [0] * 8

        # 顏色設定(RGB重複4次)
        c1, c2, c3 = (249, 86, 86) * 4, (86, 109, 249) * 4, (249, 39, 65) * 4

        # 建立目標點、手臂1、手臂2的四邊形物件
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))

    def render(self):
        pyglet.clock.tick()             # 時鐘滴答更新
        self._update_arm()              # 更新手臂與目標點位置
        self.switch_to()                # 切換繪圖視窗
        self.dispatch_events()          # 處理視窗事件(如鍵盤、滑鼠)
        self.dispatch_event('on_draw')  # 呼叫繪圖事件
        self.flip()                     # 顯示緩衝區畫面

    def on_draw(self):
        self.clear()       # 清空畫面
        self.batch.draw()  # 繪製批次內容(手臂與目標)

    def _update_arm(self):
        point_l = self.point_l

        # 計算目標點四邊形頂點(x,y)
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        # 計算第一節手臂座標(中心點->第一節末端)
        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))

        # 計算第二節手臂座標(第一節末端->第二節末端)
        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))

        # 計算第一節手臂的垂直方向角度(用於手臂寬度偏移)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]

        # 依角度計算手臂四邊形4個頂點座標
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)

        # 第二節手臂垂直方向角度
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]

        # 計算第二節手臂四邊形頂點
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)

        # 更新頂點資料至繪圖物件
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        # 使用鍵盤控制手臂角度
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1  # 第一節手臂角度增加
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1  # 第一節手臂角度減少
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1  # 第二節手臂角度增加
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1  # 第二節手臂角度減少
        elif symbol == pyglet.window.key.Q:
            # Q鍵設高FPS，達到1000FPS
            pyglet.clock.unschedule(self._dummy)
            pyglet.clock.schedule_interval(self._dummy, 1.0 / 1000)
        elif symbol == pyglet.window.key.A:
            # A鍵設低FPS，約30FPS
            pyglet.clock.unschedule(self._dummy)
            pyglet.clock.schedule_interval(self._dummy, 1.0 / 30)

    def on_mouse_motion(self, x, y, dx, dy):
        # 滑鼠移動時，目標點位置跟隨滑鼠移動
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        # 滑鼠進入視窗
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        # 滑鼠離開視窗
        self.mouse_in[0] = False

if __name__ == "__main__":
    env = ArmEnv(mode='easy')  # 初始化環境，可選模式'easy'或'hard'
    env.set_fps(60)            # 設定畫面刷新率為60FPS
    env.reset()                # 重置環境狀態

    # 使用while迴圈持續執行直到成功抓取目標點(get_point == True)
    while not env.get_point:
        action = env.sample_action()   # 隨機產生動作
        s, r, done = env.step(action)  # 環境依動作更新狀態並計算獎勵
        env.render()                   # 顯示當前手臂與目標狀態