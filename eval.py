# Evaluate the agent
best_model_path = os.path.join(checkpoint_callback.save_path, checkpoint_callback.best_model_path)
trained_agent = SAC.load(best_model_path, env=env)
mean_reward, std_reward = evaluate_policy(
    trained_agent,
    env,
    n_eval_episodes=10,
    render=False
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
