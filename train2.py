from stable_baselines3 import PPO
from snake_env import BattlesnakeRLEnv  # Ваша кастомная среда

# 1. Инициализируем среду
env = BattlesnakeRLEnv()

# 2. ЗАГРУЖАЕМ существующую модель вместо создания новой
# Мы передаем путь к старому zip-архиву и привязываем его к среде
model = PPO.load("ppo_battlesnake_11x11.zip", env=env)
print(">>> Старая модель успешно загружена. Начинаем дообучение... <<<")

# 3. Запускаем дообучение
# КРИТИЧЕСКИ ВАЖНО: reset_num_timesteps=False
# Этот флаг говорит модели продолжить график обучения (и LR scheduler), а не сбрасывать его в ноль
model.learn(total_timesteps=100000, reset_num_timesteps=False)

# 4. Сохраняем обновленную модель (можно под тем же или новым именем)
model.save("ppo_battlesnake_11x11_v2.zip")
print(">>> Дообучение завершено! Новая версия модели сохранена. <<<")