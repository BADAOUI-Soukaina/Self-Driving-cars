import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

from ENV_TRAIN_DQN import DrivingEnvMultiModelSAC  

LOG_DIR = "./logs/sac_offline_logs/"
SAVE_PATH = "./models/sac_offline_model"
EVAL_LOG_DIR = "./logs/sac_offline_eval_logs/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

def create_training_env():
    env = DrivingEnvMultiModelSAC()
    return Monitor(env, LOG_DIR)

env = make_vec_env(create_training_env, n_envs=1)

callback_stop_training = StopTrainingOnRewardThreshold(
    reward_threshold=100.0,
    verbose=1
)

eval_callback = EvalCallback(
    env,
    eval_freq=10000,
    n_eval_episodes=5,
    log_path=EVAL_LOG_DIR,
    best_model_save_path=SAVE_PATH + "_best",
    deterministic=True,
    render=False,
    callback_after_eval=callback_stop_training
)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=256,  
    tau=0.005,       
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1,
    tensorboard_log=LOG_DIR,
    ent_coef='auto'  
)

print("Démarrage de l'entraînement offline du SAC...")
print(f"Logs TensorBoard disponibles avec: tensorboard --logdir {LOG_DIR}")

try:
    model.learn(
        total_timesteps=2_000_000,
        callback=eval_callback
    )
    print("Entraînement terminé.")
except KeyboardInterrupt:
    print("Entraînement interrompu par l'utilisateur.")

model.save(SAVE_PATH)
print(f"Modèle final sauvegardé à: {SAVE_PATH}.zip")

env.close()
