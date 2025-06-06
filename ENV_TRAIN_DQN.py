import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class DrivingEnvMultiModelSAC(gym.Env):
    def __init__(self):
        super().__init__()

        self.num_classes_best = 17
        self.num_classes_passage = 1
        self.num_classes_obs1 = 4

        obs_len = self.num_classes_best + self.num_classes_passage + self.num_classes_obs1 + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([139.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.current_speed_limit = 60.0
        self.current_speed_kmh = 0.0
        self.step_count = 0
        self.max_steps = 500

        self.speed_classes_map = {
            2: 100.0, 3: 120.0, 4: 20.0, 5: 30.0, 6: 40.0,
            7: 50.0, 8: 60.0, 9: 70.0, 10: 80.0, 11: 90.0
        }

        self.CLASS_TO_BEST_INDEX = {
            "Speed Limit 100": 2, "Speed Limit 120": 3, "Speed Limit 20": 4,
            "Speed Limit 30": 5, "Speed Limit 40": 6, "Speed Limit 50": 7,
            "Speed Limit 60": 8, "Speed Limit 70": 9, "Speed Limit 80": 10,
            "Speed Limit 90": 11, "Stop Sign": 12, "Traffic Light -Red-": 15,
            "Pedestrian Crossing": 16
        }

        self.CLASS_TO_PASSAGE_INDEX = {"crosswalk": 0}
        self.CLASS_TO_OBS1_INDEX = {"cone": 0, "barrier": 1, "car": 2, "truck": 3}

        self.stop_duration_required = 3.0
        self._stop_sign_timer_steps = 0
        self._red_light_timer_steps = 0
        self._timestep_duration_s = 0.032

        self.detections_hold_steps = 10
        self.detections_counter = 0

        self._current_detections_best = np.zeros(self.num_classes_best, dtype=np.float32)
        self._current_detections_passage = np.zeros(self.num_classes_passage, dtype=np.float32)
        self._current_detections_obs1 = np.zeros(self.num_classes_obs1, dtype=np.float32)

    def _generate_random_detections(self):
        detections_best = np.zeros(self.num_classes_best, dtype=np.float32)
        detections_passage = np.zeros(self.num_classes_passage, dtype=np.float32)
        detections_obs1 = np.zeros(self.num_classes_obs1, dtype=np.float32)

       
        for _ in range(random.randint(0, 3)):
            idx = random.randint(0, self.num_classes_best - 1)
            detections_best[idx] = 1.0

       
        if random.random() < 0.3:
            detections_passage[0] = 1.0

        
        for _ in range(random.randint(0, 2)):
            idx = random.randint(0, self.num_classes_obs1 - 1)
            detections_obs1[idx] = 1.0

        return detections_best, detections_passage, detections_obs1

    def _get_obs(self):
        
        norm_speed = self.current_speed_kmh / 139.0
        norm_speed_limit = self.current_speed_limit / 139.0

        obs = np.concatenate([
            self._current_detections_best,
            self._current_detections_passage,
            self._current_detections_obs1,
            np.array([norm_speed, norm_speed_limit], dtype=np.float32)
        ])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_speed_limit = 60.0
        self.current_speed_kmh = 0.0
        self._stop_sign_timer_steps = 0
        self._red_light_timer_steps = 0
        self.detections_counter = 0

        self._current_detections_best, self._current_detections_passage, self._current_detections_obs1 = self._generate_random_detections()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

    
        speed_cmd = np.clip(action[0], 0.0, 139.0)
        brake_cmd = np.clip(action[1], 0.0, 1.0)
        steer_cmd = np.clip(action[2], -1.0, 1.0)

        ACCEL_RATE = 10.0
        DECEL_RATE = 20.0
        BRAKE_RATE = 30.0

        if brake_cmd > 0.5:
            self.current_speed_kmh = max(0.0, self.current_speed_kmh - BRAKE_RATE)
        else:
            if self.current_speed_kmh < speed_cmd:
                self.current_speed_kmh = min(speed_cmd, self.current_speed_kmh + ACCEL_RATE)
            elif self.current_speed_kmh > speed_cmd:
                self.current_speed_kmh = max(speed_cmd, self.current_speed_kmh - DECEL_RATE)

        self.current_speed_kmh = np.clip(self.current_speed_kmh, 0.0, 139.0)

        # --- RECOMPENSE ---

        # 1. Respect de la limitation de vitesse
        if self.current_speed_kmh <= self.current_speed_limit:
            reward += 1.0
        else:
            reward -= 2.0

        # 2. Gestion panneau STOP
        stop_detected = self._current_detections_best[self.CLASS_TO_BEST_INDEX["Stop Sign"]] == 1
        if stop_detected:
            if self.current_speed_kmh < 1.0:
                self._stop_sign_timer_steps += 1
                reward += 2.0  
                if self._stop_sign_timer_steps * self._timestep_duration_s >= self.stop_duration_required:
                    reward += 5.0  
            else:
                reward -= 5.0 
                self._stop_sign_timer_steps = 0
        else:
            self._stop_sign_timer_steps = 0

        # 3. Gestion feu rouge
        red_light_detected = self._current_detections_best[self.CLASS_TO_BEST_INDEX["Traffic Light -Red-"]] == 1
        if red_light_detected:
            if self.current_speed_kmh < 1.0:
                self._red_light_timer_steps += 1
                reward += 2.0  # récompense arrêt au feu rouge
            else:
                reward -= 5.0  # pénalité
                self._red_light_timer_steps = 0
        else:
            self._red_light_timer_steps = 0

        # 4. Obstacles : pénalité si vitesse > 0 et obstacle détecté
        obstacles_detected = np.any(self._current_detections_obs1 == 1)
        if obstacles_detected and self.current_speed_kmh > 0:
            reward -= 3.0  

        # 5. Terminaison si trop d'étapes
        if self.step_count >= self.max_steps:
            truncated = True

        # Mise à jour des détections toutes les 10 étapes
        self.detections_counter += 1
        if self.detections_counter >= self.detections_hold_steps:
            self._current_detections_best, self._current_detections_passage, self._current_detections_obs1 = self._generate_random_detections()
            self.detections_counter = 0

        obs = self._get_obs()

        return obs, reward, terminated, truncated, {}
