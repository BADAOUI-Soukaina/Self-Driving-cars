import socket
import json
import numpy as np
from stable_baselines3 import SAC

MODEL_PATH = "D:/self_car_driving/models/sac_offline_model.zip"
HOST = '127.0.0.1'
PORT = 50007

print("Chargement du modèle SAC...")
model = SAC.load(MODEL_PATH)
print("Modèle SAC chargé.")

NUM_CLASSES_BEST = 17
NUM_CLASSES_PASSAGE = 1
NUM_CLASSES_OBS1 = 4
EXPECTED_OBS_SHAPE = (24,)

def process_received_obs(received_obs_list, expected_shape):
    # Remplacer les nan par 0.0 pour éviter problème de conversion numpy
    cleaned_list = [0.0 if (isinstance(x, float) and (x != x)) else x for x in received_obs_list]
    obs = np.array(cleaned_list, dtype=np.float32)
    
    if obs.shape != expected_shape:
        if obs.size == expected_shape[0]:
            obs = obs.reshape(expected_shape)
        else:
            print(f"Observation malformée, taille reçue: {obs.size}, attendue: {expected_shape[0]}")
            obs = np.zeros(expected_shape, dtype=np.float32)
    return obs

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Serveur SAC en écoute sur {HOST}:{PORT} ...")
    
    conn, addr = s.accept()
    with conn:
        print(f"Connexion acceptée de {addr}")
        data_buffer = b""
        
        while True:
            try:
                packet = conn.recv(4096)
                if not packet:
                    print("Connexion fermée par le client.")
                    break
                data_buffer += packet
                if b'\n' in data_buffer:
                    parts = data_buffer.split(b'\n')
                    raw_msg = parts[0]
                    data_buffer = b'\n'.join(parts[1:])
                    
                    try:
                        obs_json = json.loads(raw_msg.decode('utf-8').strip())
                    except json.JSONDecodeError:
                        print("Erreur JSON reçue, on attend la prochaine.")
                        continue
                    
                    if "observation" not in obs_json:
                        print("Clé 'observation' manquante.")
                        continue
                    
                    obs = process_received_obs(obs_json["observation"], EXPECTED_OBS_SHAPE)
                    print(f"Observation reçue (taille {len(obs)}): {obs.tolist()}")
                    
                    action, _ = model.predict(obs, deterministic=True)
                    
                    if not isinstance(action, np.ndarray) or action.shape != (3,):
                        action = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                    
                    speed_cmd = float(np.clip(action[0], 0.0, 139.0))
                    brake_cmd = float(np.clip(action[1], 0.0, 1.0))
                    steer_cmd = float(np.clip(action[2], -1.0, 1.0))
                    
                    response = {
                        "speed": speed_cmd,
                        "brake": brake_cmd,
                        "steer": steer_cmd,
                        "reset_signal": False
                    }
                    
                    response_str = json.dumps(response) + "\n"
                    conn.sendall(response_str.encode('utf-8'))
                    
            except Exception as e:
                print(f"Exception serveur SAC : {e}")
                break

print("🛑 Serveur SAC arrêté.")
