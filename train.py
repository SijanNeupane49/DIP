import time
#import pygame
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


from env import DoubleInvertedPendulumCartEnv

env = DoubleInvertedPendulumCartEnv()




#env = DummyVecEnv([lambda: DoubleInvertedPendulumCartEnv()])

#model = SAC(MlpPolicy, env, verbose=1)
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
    save_freq=1000,
    save_path='./double_inverted_pendulum_on_cart_checkpoints/',
    name_prefix='sac_double_inverted_pendulum_on_cart_checkpoint'
)

# Train the agent
agent.learn(
    total_timesteps=10000,
    callback=checkpoint_callback,
    log_interval=4,
    tb_log_name= 'SAC',
    #eval_env=env,
    #eval_freq=10000,
    #n_eval_episodes=10,
    #eval_log_path="./double_inverted_pendulum_on_cart_eval_results/",
    reset_num_timesteps=False,
    progress_bar=True
    #action_noise=None,
    #tb_log_name="double_inverted_pendulum_on_cart"
)

obs = env.reset()
frames = []

for i in range(1000):
    action, _states = agent.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    frame = env.render()
    if frame is not None:#for rendering with matplotlib, to save frames
         frames.append(frame)
    env.render()
    #pygame.time.delay(2)
    time.sleep(0.01)
    if done:
         observation = env.reset()
    
env.close()


# Save the frames as a .gif file
frames[0].save('double_inverted_pendulum.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)

