# Создайте файл export_ppo.py и запустите его локально
import json
import torch
from stable_baselines3 import PPO
from snake_env import BattlesnakeRLEnv

def main():
    # 1. Загружаем обученную модель
    model = PPO.load("ppo_battlesnake_11x11")
    policy = model.policy

    weights = {}

    # 2. Извлекаем слои mlp_extractor (архитектуру pi=[256, 128, 64])
    # Находим все полносвязные слои в сети стратегии
    layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    
    for i, layer in enumerate(layers):
        weights[f"W_hidden_{i}"] = layer.weight.detach().cpu().numpy().tolist()
        weights[f"b_hidden_{i}"] = layer.bias.detach().cpu().numpy().tolist()

    # 3. Извлекаем финальный слой, который выдает 4 действия (action_net)
    weights["W_action"] = policy.action_net.weight.detach().cpu().numpy().tolist()
    weights["b_action"] = policy.action_net.bias.detach().cpu().numpy().tolist()

    # 4. Сохраняем всё в компактный JSON
    with open("ppo_weights.json", "w") as f:
        json.dump(weights, f)
        
    print("🔥 Веса нейросети успешно сохранены в файл ppo_weights.json!")

if __name__ == "__main__":
    main()