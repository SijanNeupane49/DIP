from env import DoubleInvertedPendulumCartEnv
from train import train_agent
from evaluate import evaluate_agent

env = DoubleInvertedPendulumCartEnv()
trained_agent = train_agent(env)
mean_reward, std_reward = evaluate_agent(trained_agent, env)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
