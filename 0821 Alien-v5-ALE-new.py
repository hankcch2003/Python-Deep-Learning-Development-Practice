# pip install gymnasium
# pip install gymnasium[accept-rom-license]
# pip install gymnasium[atari]
# pip install pygame
# pip install ale_py

# 匯入gymnasium套件，為強化版的OpenAI Gym，用於建構與操作強化學習環境
import gymnasium as gym

# 匯入ale_py模組，它是Atari Learning Environment(ALE)的Python介面
import ale_py

# 註冊ale_py所支援的Atari環境至gymnasium
gym.register_envs(ale_py)

# 建立Atari遊戲環境Alien(ALE版本v5)，並啟用人類可視畫面(render_mode='human')
env = gym.make("ALE/Alien-v5", render_mode='human')

# 重置環境，開始新的一局，獲得初始觀測值observation
observation = env.reset()

# 執行300個時間步(避免無窮迴圈)
for _ in range(300):
    # 設定畫面更新速率為每秒60張影格(FPS)，保持畫面流暢
    env.metadata['render_fps'] = 60

    # 顯示目前畫面
    env.render()

    # 從動作空間中隨機選取一個動作(代表agent的行為)
    action = env.action_space.sample()

    # 執行動作，回傳新的觀測值、獎勵、終止旗標、截斷旗標與其他資訊
    observation, reward, terminated, truncated, info = env.step(action)

    # 若遊戲回合結束(成功或失敗)，則重置環境開始新的一局
    if terminated or truncated:
        observation = env.reset()

# 關閉環境並釋放資源
env.close()