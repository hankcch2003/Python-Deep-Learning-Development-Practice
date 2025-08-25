"""
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_v2_behavior()           # 禁用TensorFlow 2.x的行為，回到1.x風格

import numpy as np
import os
import shutil
import sys

# 將自訂環境路徑加入sys.path，方便引用自訂模組arm_env
sys.path.append(r'E:\Colab第三階段202505\ch18_20250814')
from arm_env import ArmEnv  # 匯入自訂的機械臂環境
np.random.seed(1)           # 固定numpy亂數種子，確保結果可重現
tf.set_random_seed(1)       # 固定tensorflow亂數種子

# 訓練相關參數設定
MAX_EPISODES = 200          # 最大訓練回合數
MAX_EP_STEPS = 200          # 每回合最大步數
LR_A = 1e-4                 # Actor網路學習率
LR_C = 1e-4                 # Critic網路學習率
GAMMA = 0.9                 # 折扣因子，衡量未來獎勵的重要性
REPLACE_ITER_A = 1100       # Actor目標網路參數更新頻率
REPLACE_ITER_C = 1000       # Critic目標網路參數更新頻率
MEMORY_CAPACITY = 5000      # 經驗回放記憶體容量
BATCH_SIZE = 16             # 每次訓練抽樣批次大小
VAR_MIN = 0.1               # 探索噪音的最小方差
RENDER = True               # 是否在訓練時渲染環境畫面
LOAD = True                 # 是否載入已訓練模型(此範例中未使用)
MODE = ['easy', 'hard']     # 環境難度模式
n_model = 1                 # 選擇難度索引(1代表hard)

# 建立環境物件
env = ArmEnv(mode = MODE[n_model])
STATE_DIM = env.state_dim        # 狀態維度
ACTION_DIM = env.action_dim      # 動作維度
ACTION_BOUND = env.action_bound  # 動作邊界(上下限)

# 定義TensorFlow的placeholder，存放狀態、獎勵、下一狀態
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape = [None, STATE_DIM], name='s')    # 當前狀態
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')                    # 獎勵
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape = [None, STATE_DIM], name='s_')  # 下一狀態

# Actor類別，負責生成動作
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim               # 動作維度
        self.action_bound = action_bound      # 動作上下界
        self.lr = learning_rate               # 學習率
        self.t_replace_iter = t_replace_iter  # 目標網路更新頻率
        self.t_replace_counter = 0            # 計數器，用於追蹤更新時機

        with tf.variable_scope('Actor'):
            # 評估網路，輸入狀態，輸出動作
            self.a = self._build_net(S, scope='eval_net', trainable = True)

            # 目標網路，輸入下一狀態，輸出動作，用於Critic目標計算
            self.a_ = self._build_net(S_, scope='target_net', trainable = False)

        # 取得Actor網路的參數集合
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.keras.initializers.GlorotUniform()  # 權重初始化器
            init_b = tf.constant_initializer(0.001)         # 偏差初始化器
            
            # 第一層全連接層，200個神經元，激活函數為relu6
            dense1 = tf.keras.layers.Dense(200, activation = tf.nn.relu6, kernel_initializer = init_w,
                                           bias_initializer = init_b, name='l1', trainable = trainable)
            net = dense1(s)

            # 第二層全連接層，200個神經元，激活函數為relu6
            dense2 = tf.keras.layers.Dense(200, activation = tf.nn.relu6, kernel_initializer = init_w,
                                           bias_initializer = init_b, name='l2', trainable = trainable)
            net = dense2(net)

            # 第三層全連接層，10個神經元，激活函數為relu
            dense3 = tf.keras.layers.Dense(10, activation = tf.nn.relu, kernel_initializer = init_w,
                                           bias_initializer = init_b, name='l3', trainable = trainable)
            net = dense3(net)

            with tf.variable_scope('a'):
                # 輸出層，輸出動作值，激活函數為tanh，輸出範圍[-1, 1]
                dense_a = tf.keras.layers.Dense(self.a_dim, activation = tf.nn.tanh,
                                                kernel_initializer = init_w, name='a',
                                                trainable = trainable)
                actions = dense_a(net)

                # 將動作值放大至動作邊界範圍內
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')
        return scaled_a

    # Actor網路的訓練步驟，使用梯度上升更新策略網路
    def learn(self, s):   # 批次更新(一次用一批資料訓練)
        self.sess.run(self.train_op, feed_dict = {S: s})

        # 每隔t_replace_iter步驟將目標網路參數更新為評估網路參數
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    # 選擇動作，給定當前狀態，輸出動作
    def choose_action(self, s):
        s = s[np.newaxis, :]                                 # 單筆狀態擴成batch形狀
        return self.sess.run(self.a, feed_dict = {S: s})[0]  # 回傳單筆動作

    # 將Critic計算出的動作梯度加入Actor的計算圖，用於策略梯度更新
    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys = self.a, xs = self.e_params, grad_ys = a_grads)

        with tf.variable_scope('A_train'):

            # 使用RMSProp優化器做梯度上升，學習率取負號
            opt = tf.train.RMSPropOptimizer(-self.lr)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


# Critic類別，負責評估狀態-動作值(Q值)
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess                      # TensorFlow的會話(Session)，負責執行計算圖
        self.s_dim = state_dim                # 狀態(state)的維度大小
        self.a_dim = action_dim               # 動作(action)的維度大小
        self.lr = learning_rate               # 學習率，用於調整網路權重更新幅度
        self.gamma = gamma                    # 折扣因子(Discount Factor)，衡量未來獎勵的重要性，範圍0~1
        self.t_replace_iter = t_replace_iter  # 目標網路參數更新的頻率(步數)
        self.t_replace_counter = 0            # 計數器，追蹤目前已執行多少步，用於判斷何時更新目標網路參數


        with tf.variable_scope('Critic'):
            # 評估網路，輸入當前狀態與動作，輸出Q值
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable = True)

            # 目標網路，輸入下一狀態與動作，輸出目標Q值
            self.q_ = self._build_net(S_, a_, 'target_net', trainable = False)

            # 取得參數集合
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            # 計算目標Q值：獎勵 + 折扣後的下一狀態Q值
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            # 計算TD誤差(均方差損失)
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            # 使用RMSProp優化器來最小化損失
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            # 計算Q值對動作的梯度，用於Actor更新策略
            self.a_grads = tf.gradients(self.q, a)[0]

    # Critic網路訓練步驟
    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        # 每隔t_replace_iter步驟更新目標網路參數
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    # Critic網路結構
    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.keras.initializers.GlorotUniform()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200

                # 狀態與動作分別乘權重後相加，再加偏差，通過relu6激活
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer = init_w, trainable = trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer = init_w, trainable = trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer = init_b, trainable = trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            # 第二層全連接層，200神經元，relu6激活
            dense2 = tf.keras.layers.Dense(200, activation = tf.nn.relu6, kernel_initializer = init_w,
                                           bias_initializer = init_b, name='l2', trainable = trainable)
            net = dense2(net)

            # 第三層全連接層，10神經元，relu激活
            dense3 = tf.keras.layers.Dense(10, activation = tf.nn.relu, kernel_initializer = init_w,
                                           bias_initializer = init_b, name='l3', trainable = trainable)
            net = dense3(net)

            with tf.variable_scope('q'):
                # 輸出Q值，不使用激活函數(線性輸出)
                dense_q = tf.keras.layers.Dense(1, kernel_initializer = init_w,
                                                bias_initializer = init_b, trainable = trainable)
                q = dense_q(net)   # Q(s, a)：表示在狀態s執行動作a所能得到的預期總回報(價值)
        return q

# 經驗回放記憶體，用於儲存與抽樣訓練數據
class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity                 # 記憶體最大容量(最多可存多少筆transition)
        self.data = np.zeros((capacity, dims))   # 預先建立一個(capacity, dims)的0陣列，用來存放transition資料
        self.pointer = 0                         # 儲存索引指標，記錄下一筆資料要儲存的位置

    # 儲存一筆轉移資料(狀態, 動作, 獎勵, 下一狀態)
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))  # 將s, a, r, s_水平堆疊成一筆完整的transition資料
        index = self.pointer % self.capacity     # 使用環狀緩衝區機制，確保超出容量時會覆寫舊資料
        self.data[index, :] = transition         # 將transition存入記憶體中對應的位置
        self.pointer += 1                        # 更新指標，指向下一筆要儲存的位置

    # 從記憶體中隨機抽樣n筆資料，用於訓練
    def sample(self, n):
        # 確保記憶體已填滿，才能進行抽樣
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size = n)  # 隨機選擇n個索引
        return self.data[indices, :]                         # 回傳對應索引的transition資料

sess = tf.Session()  # 建立TensorFlow會話

# 建立Actor和Critic物件
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)

# 將Critic計算出的動作梯度加入Actor的更新圖中
actor.add_grad_to_graph(critic.a_grads)

# 初始化所有變數
sess.run(tf.global_variables_initializer())

# 初始化經驗回放記憶體，dims為狀態 * 2 + 動作 + 1(獎勵)
memory = Memory(MEMORY_CAPACITY, dims = STATE_DIM * 2 + ACTION_DIM + 1)

var = 3.0  # 控制探索噪音的初始變異數
t1 = 0     # 計時器變數(目前未使用)，可用來記錄訓練所花費的時間或步數

# 主訓練迴圈
for i in range(MAX_EPISODES):
    s = env.reset()  # 重置環境，獲得初始狀態
    ep_reward = 0    # 回合累積獎勵
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()            # 渲染環境畫面
        a = actor.choose_action(s)  # Actor選擇動作

        # 加入高斯噪音以增加探索，並裁剪動作至合法範圍內
        a = np.clip(np.random.normal(a, var), ACTION_BOUND[0], ACTION_BOUND[1])
        s_, r, done = env.step(a)             # 執行動作，得到下一狀態、獎勵和是否結束
        memory.store_transition(s, a, r, s_)  # 將經驗存入回放記憶體

        # 訓練條件：當記憶體容量已滿才開始訓練(避免資料太少造成訓練不穩)
        if memory.pointer > MEMORY_CAPACITY:
           # 探索噪音的變異數(var)隨著訓練逐漸減少，但不低於VAR_MIN(保留最小隨機性)
           var = max([var * 0.9999, VAR_MIN])

           # 從經驗回放記憶體中隨機抽樣一個批次(batch size)的訓練資料
           b_M = memory.sample(BATCH_SIZE) 

           # 從抽樣資料中擷取「狀態s」
           b_s = b_M[:, :STATE_DIM]

           # 擷取「動作a」
           b_a = b_M[:, STATE_DIM : STATE_DIM + ACTION_DIM]

           # 擷取「獎勵r」
           b_r = b_M[:, -STATE_DIM - 1 : -STATE_DIM]

           # 擷取「下一狀態s_」
           b_s_ = b_M[:, -STATE_DIM:]

           # 使用這批資料更新Critic網路(更新Q(s,a)預測能力)
           critic.learn(b_s, b_a, b_r, b_s_)

           # 使用這批資料更新Actor網路(改進策略產生動作能力)
           actor.learn(b_s)

           # 狀態更新，為下一個時間步做準備
           s = s_

           # 將當前步驟的獎勵加總到本回合的總獎勵中
           ep_reward += r  

           # 若到達回合的最大步數(MAX_EP_STEPS)
           if j == MAX_EP_STEPS - 1:
               # 輸出訓練資訊(回合數、總獎勵、當前探索噪音)
               print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var)

               # 如果回合的總獎勵超過-300，就開啟畫面渲染(更高獎勵代表策略可能有學會)
               if ep_reward > -300:
                  RENDER = True

               # 結束當前訓練回合
               break