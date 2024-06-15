import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class AvoidBulletsEnv(gym.Env):
    def __init__(self):
        super(AvoidBulletsEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 3 actions: left, stay, right
        self.observation_space = spaces.Box(low=0, high=5, shape=(4,), dtype=np.int32)
        self.state = None
        self.bullet_position = [0, 0]  # bullet (x, y)
        self.plane_position = 2  # Plane's x position in range [0, 5]

    def reset(self):
        self.bullet_position = [random.randint(0, 5), 0]
        self.plane_position = 2
        self.state = np.array([self.plane_position, 0, self.bullet_position[0], self.bullet_position[1]])
        return self.state

    def step(self, action):
        # Uçağın pozisyonunu güncelle
        if action == 0:  # sola hareket et
            self.plane_position = max(0, self.plane_position - 1)
        elif action == 2:  # sağa hareket et
            self.plane_position = min(5, self.plane_position + 1)

        # Merminin pozisyonunu güncelle
        self.bullet_position[1] += 1

        # Çarpışma kontrolü
        done = False
        if self.bullet_position[1] > 2:
            # Mermi en alta ulaştı, yeni bir mermi yukarıda başlar
            self.bullet_position = [random.randint(0, 5), 0]
        elif self.bullet_position[1] == 2 and self.bullet_position[0] == self.plane_position:
            done = True  # Çarpışma durumu

        # Ödülü hesapla
        reward = 1 if not done else -10

        # Durumu güncelle
        self.state = np.array([self.plane_position, 0, self.bullet_position[0], self.bullet_position[1]])
        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Basit bir terminal tabanlı görselleştirme
        grid = [[' ' for _ in range(6)] for _ in range(3)]
        if self.bullet_position[1] < 3:
            grid[self.bullet_position[1]][self.bullet_position[0]] = '*'
        grid[2][self.plane_position] = 'A'
        print('\n'.join(['|'.join(row) for row in grid]))
        print('-' * 13)

    def close(self):
        pass

# Ortamı Gymnasium'a kaydedin
gym.envs.registration.register(
    id='AvoidBullets-v1',
    entry_point='avoid_bullets_env:AvoidBulletsEnv',
)
