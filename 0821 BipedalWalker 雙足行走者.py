# pip install gymnasium
# pip install pygame

# 匯入gymnasium套件，為強化版的OpenAI Gym，用於建構與操作強化學習環境
import gymnasium as gym

# 建立Box2D環境BipedalWalker-v3，並啟用人類可視畫面(render_mode='human')
env = gym.make("BipedalWalker-v3", render_mode='human')

# 重置環境，開始新的一局，獲得初始觀測值observation
observation = env.reset()

# 執行300個時間步(避免無窮迴圈)
for _ in range(300):
    # 顯示目前畫面
    env.render()

    # 從動作空間中隨機選取一個動作(代表agent的行為)
    action = env.action_space.sample()

    # 執行動作，回傳新的觀測值、獎勵、終止旗標、截斷旗標與其他資訊
    observation,reward,terminated,truncated,info = env.step(action)

    # 若遊戲回合結束(成功或失敗)，則重置環境開始新的一局
    if terminated or truncated:
        observation,info = env.reset()

# 關閉環境並釋放資源
env.close()