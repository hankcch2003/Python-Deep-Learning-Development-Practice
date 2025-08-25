import numpy as np
from gym import Env, spaces
from gym.utils import seeding

# 從類別機率中抽樣一個索引
def categorical_sample(prob_n, np_random):

    """
    從一個類別分佈中抽樣(Categorical distribution)
    prob_n: 各類別的機率(總和為1)
    np_random: 隨機數生成器(由Gym封裝)
    """

    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)                   # 累加機率分佈
    return (csprob_n > np_random.rand()).argmax()  # 找到隨機值落在哪個機率區段

# 自定義離散環境類別，繼承自Gym的Env類別
class DiscreteEnv(Env):

    """
    離散型環境基礎類別，包含以下成員：
    - nS: 狀態數量
    - nA: 動作數量
    - P: 狀態轉移機率表(如下格式)
          P[s][a] = [(機率, 下一狀態, 獎勵, 是否結束), ...]
    - isd: 初始狀態機率分布
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P                 # 狀態轉移表
        self.isd = isd             # 初始狀態分佈
        self.lastaction = None     # 上一次的動作(for render/debug)
        self.nS = nS               # 狀態數
        self.nA = nA               # 動作數

        # 定義Gym格式的動作與狀態空間(使用離散空間)
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # 設定隨機種子並初始化隨機物件
        self.seed()

        # 初始狀態從isd中抽樣
        self.s = categorical_sample(self.isd, self.np_random)

    def seed(self, seed = None):

        """
        設定隨機種子(使得訓練可重現)
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        """
        將環境重設為初始狀態(從初始狀態分佈isd中抽樣)
        """

        self.s = categorical_sample(self.isd, self.np_random)
        self.lastaction = None
        return int(self.s)  # 回傳初始狀態編號

    def step(self, a):

        """
        執行一個動作，根據目前狀態與動作取得下一狀態與獎勵
        a: 動作(action)
        回傳: (新狀態, 獎勵, 是否結束, 額外資訊)
        """

        # 根據狀態與動作取得轉移列表
        transitions = self.P[self.s][a]

        # 根據機率抽樣一個轉移
        i = categorical_sample([t[0] for t in transitions], self.np_random)

        # 拆解轉移結果：機率、下一狀態、獎勵、是否結束
        p, s, r, d = transitions[i]
        self.s = s                          # 更新狀態
        self.lastaction = a                 # 記錄動作
        return (int(s), r, d, {"prob": p})  # 回傳標準Gym格式的資訊