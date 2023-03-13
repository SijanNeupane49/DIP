import os
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from scipy.signal import StateSpace
from env import DoubleInvertedPendulumCartEnv


# Define the environment
env = DoubleInvertedPendulumCartEnv()

# Create the state space model

A, B, C, D = env.get_state_space_model()
ss = StateSpace(A, B, C, D)
ss_tensor = th.tensor(ss.A)

# Define the reference
Q = np.eye(6)
R = np.eye(1)
K, S, E = th.linalg.svd(ss_tensor)
L = 1.0 / (C @ th.pinverse(K) @ B)
K = L * th.pinverse(K) @ th.solve(E @ S @ Q @ E.T + R, E @ S @ K.T).T

# Define the action scaling function
def scale_action(action):
    return np.array([action[0] * 10.0])

# Create the vectorized environment
env = DummyVecEnv([lambda: DoubleInvertedPendulumOnCartEnv()])

# Create the SAC agent
agent = SAC(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    learning_rate=0.0003,
    buffer_size=1000000,
    learning_starts=10000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    gradient_steps=1,
    train_freq=1,
    action_noise=None,
    target_update_interval=1,
    tensorboard_log="./double_inverted_pendulum_on_cart_tensorboard/",
    use_sde=False,
    sde_sample_freq=-1,
    policy_kwargs={
        'net_arch': [256, 256]
    }
)

# Create the checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./double_inverted_pendulum_on_cart_checkpoints/',
    name_prefix='sac_double_inverted_pendulum_on_cart_checkpoint'
)

# Train the agent
agent.learn(
    total_timesteps=1000000,
    callback=checkpoint_callback,
    eval_env=env,
    eval_freq=10000,
    n_eval_episodes=10,
    eval_log_path="./double_inverted_pendulum_on_cart_eval_results/",
    reset_num_timesteps=False,
    action_noise=None,
    tb_log_name="double_inverted_pendulum_on_cart"
)
