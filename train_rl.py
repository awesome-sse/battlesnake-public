import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from snake_env import BattlesnakeRLEnv

def main():
    print("Создание параллельных сред...")
    # Запускаем 4 параллельных симулятора для быстрой сборки опыта
    env = make_vec_env(lambda: BattlesnakeRLEnv(), n_envs=16)

    # Авто-выбор девайса
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство для обучения: {device.upper()}")

    # Кастомная архитектура сети под Best-решение
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64], # Сеть стратегии (выбора ходов)
            vf=[256, 128, 64]  # Сеть оценки безопасности состояния
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None, # Обходим баг Windows
        device=device
    )

    print("Обучение началось. Ждем...")
    # 250k шагов — это минут 7-10 на твоем ПК. Модель будет готова!
    model.learn(total_timesteps=250_000)
    
    model.save("ppo_battlesnake_11x11")
    print("Успех! Модель ppo_battlesnake_11x11.zip сохранена.")

if __name__ == "__main__":
    main()