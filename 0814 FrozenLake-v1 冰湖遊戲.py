# pip install gymnasium
# pip install gymnasium[accept-rom-license]
# pip install gymnasium[mujoco]
# pip install pygame
# https://gymnasium.farama.org/

# 匯入gymnasium套件，強化版的OpenAI Gym
import gymnasium as gym

# 建立FrozenLake環境，render_mode='human'代表使用圖形介面顯示遊戲畫面
env = gym.make("FrozenLake-v1", render_mode='human')

# 重置環境，初始化並回傳初始觀察值observation
observation = env.reset()

# 執行最多1000步的遊戲回合
for _ in range(1000):
    env.render()                        # 顯示當前環境狀態(視覺化)
    action = env.action_space.sample()  # 隨機採樣一個動作(此處代理人隨機決策)
    
    # 執行動作，並取得新的觀察值、獎勵、遊戲是否結束(terminated)、是否截斷(truncated)、其他資訊
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 若遊戲結束(角色掉進洞穴或成功到達終點)
    if terminated:
        # 重置環境開始新一局遊戲，並取得初始觀察值
        observation = env.reset()

# 遊戲結束，關閉環境釋放資源
env.close()