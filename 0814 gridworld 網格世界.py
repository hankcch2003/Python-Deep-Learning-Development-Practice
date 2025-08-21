import io                                 # 用於文字輸出到緩衝區，可用於render函數
import sys                                # 用於系統相關操作，例如輸出到stdout
import time                               # 用於暫停程式，製作動畫效果
import numpy as np                        # 科學計算套件，處理矩陣運算
from IPython.display import clear_output  # 在Jupyter Notebook中清除輸出，製作動畫效果

# 定義離散環境基底
class DiscreteEnv:
    """
    簡化版離散環境基底，用作GridworldEnv繼承
    """
    def __init__(self, nS, nA, P, isd):
        self.nS = nS              # 狀態總數
        self.nA = nA              # 動作總數
        self.P = P                # 狀態轉移模型字典 {state: {action: [(prob, next_state, reward, done)]}}
        self.isd = isd            # 初始狀態分佈，決定reset()時初始狀態的機率
        self.s = None             # 當前狀態
        self.action_space = self  # 簡化版，讓 env.action_space.sample()可以隨機選動作

    def reset(self):
        # 隨機依據初始分佈選擇起始狀態
        self.s = np.random.choice(self.nS, p = self.isd)
        return self.s

    def step(self, a):
        """
        執行一個動作 a
        回傳(next_state, reward, done, info)
        """
        transitions = self.P[self.s][a]                  # 取得當前狀態採取動作a的所有轉移資訊
        prob, next_state, reward, done = transitions[0]  # 假設deterministic，只取第一個轉移
        self.s = next_state                              # 更新當前狀態
        return next_state, reward, done, {}              # 回傳下一狀態、獎勵、是否終止、額外資訊

    def sample(self):
        """
        隨機選擇一個動作
        """
        return np.random.randint(0, self.nA)            # 0 ~ nA-1隨機整數

# 定義GridworldEnv(動作對應編號)
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}  # 設定渲染模式，可選human(螢幕)或ansi(文字緩衝區)

    def __init__(self, shape = [4, 4]):
        self.shape = shape
        nS = np.prod(shape)   # 狀態總數 = 行數 * 列數
        nA = 4                # 動作總數，上下左右四個方向
        MAX_Y, MAX_X = shape  # 最大行數與列數，用於計算邊界

        P = {}                                           # 初始化狀態轉移字典
        grid = np.arange(nS).reshape(shape)              # 將狀態編號 0 ~ nS-1 轉成矩陣
        it = np.nditer(grid, flags=['multi_index'])      # 迭代器，用於遍歷每個格子

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index                        # 取得當前格子對應的行列座標
            P[s] = {a: [] for a in range(nA)}            # 初始化每個動作的轉移列表
            is_done = lambda s: s == 0 or s == (nS - 1)  # 終點條件，左上角和右下角為終點
            reward = 0.0 if is_done(s) else -1.0         # 終點獎勵0，其它步驟獎勵-1

            if is_done(s):
                # 如果是終點格子，所有動作都維持原地
                for a in range(nA):
                    P[s][a] = [(1.0, s, reward, True)]
            else:
                # 計算每個方向的下一格座標，遇邊界停留原地
                ns_up    = s if y == 0             else s - MAX_X
                ns_right = s if x == (MAX_X - 1)   else s + 1
                ns_down  = s if y == (MAX_Y - 1)   else s + MAX_X
                ns_left  = s if x == 0             else s - 1

                # 儲存轉移信息：機率、下一狀態、獎勵、是否終止
                P[s][UP]    = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN]  = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT]  = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()       # 移動到下一個格子

        isd = np.ones(nS) / nS  # 均勻初始分佈
        self.P = P
        super(GridworldEnv, self).__init__(nS, nA, P, isd)       # 呼叫父類別初始化

    def render(self, mode='human'):
        """
        將當前格子環境渲染到螢幕或文字緩衝區
        """
        grid = np.arange(self.nS).reshape(self.shape)            # 建立格子矩陣
        outfile = io.StringIO() if mode=='ansi' else sys.stdout  # human模式直接輸出到螢幕
        it = np.nditer(grid, flags = ['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            if self.s == s:
                output = " x "  # 當前位置
            elif s == 0 or s == self.nS - 1:
                output = " T "  # 終點
            else:
                output = " o "  # 空格
            if x == 0:
                output = output.lstrip()   # 左邊界不要多餘空白
            if x == self.shape[1] - 1:
                output = output.rstrip()   # 右邊界不要多餘空白
            outfile.write(output)
            if x == self.shape[1] - 1:
                outfile.write("\n")        # 每行結束換行
            it.iternext()

        if mode == 'ansi':
            return outfile.getvalue()     # 如果是文字模式，回傳文字

# 執行隨機策略動畫
env = GridworldEnv()                      # 建立Gridworld環境
state = env.reset()                       # 重置環境，取得初始狀態
done = False                              # 回合是否結束

# 隨機策略動畫
while not done:
    clear_output(wait = True)                       # 清除上一個輸出，產生動畫效果
    env.render()                                    # 顯示目前格子狀態
    action = env.action_space.sample()              # 隨機採取一個動作
    state, reward, done, info = env.step(action)    # 執行動作，更新狀態
    print(f"Action: {action}, Reward: {reward}\n")  # 顯示動作編號與獎勵
    time.sleep(0.3)                                 # 暫停0.3秒，控制動畫速度

print("Episode finished.")                          # 回合結束