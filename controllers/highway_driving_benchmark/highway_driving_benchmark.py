print("Contrôleur Python lancé")

# --- Imports ---
try:
    from ultralytics import YOLO
    from vehicle import Driver
    import numpy as np
    import cv2
    import os
    import time
    import socket
    import json
except Exception as e:
    print(f"Erreur pendant l'import : {e}")
    exit(1)

# --- Chargement des modèles YOLO ---
try:
    model_best = YOLO("D:/self_car_driving/best.pt")
    model_passage = YOLO("D:/self_car_driving/passage.pt")
    model_obs1 = YOLO("D:/self_car_driving/obs1.pt")
except Exception as e:
    print(f"Erreur chargement modèles YOLO : {e}")
    exit(1)

# --- Initialisation Webots ---
driver = Driver()
timestep = int(driver.getBasicTimeStep())
maxSpeed = 139
driver.setSteeringAngle(0.0)

# Capteurs de distance
sensors = {}
for name in [
    'front', 'front right 0', 'front right 1', 'front right 2',
    'front left 0', 'front left 1', 'front left 2',
    'rear', 'rear left', 'rear right', 'right', 'left'
]:
    sensor = driver.getDevice('distance sensor ' + name)
    sensor.enable(timestep)
    sensors[name] = sensor

# GPS
gps = driver.getDevice('gps')
gps.enable(timestep)

# Caméras
camera = driver.getDevice('camera')
camera.disable()

camera2 = driver.getDevice('camera(1)')
camera2.enable(timestep)
camera2.recognitionEnable(timestep)

# Dossier d'enregistrement d'images
save_dir = r"D:\self_car_driving\images"
os.makedirs(save_dir, exist_ok=True)
image_counter = 0

# Dernier label détecté (pour éviter doublons successifs)
last_detected_label = None
current_frame_labels = set()

# --- Connexion socket avec l'agent SAC ---
HOST = '127.0.0.1'
PORT = 50007
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Connexion au serveur SAC établie.")
except Exception as e:
    print(f"Erreur de connexion socket : {e}")
    exit(1)

# --- Liste complète des classes (22) ---
all_classes = [
    # model_best (17 classes)
    "Pedestrian Crossing", "Radar", "Speed Limit -100-", "Speed Limit -120-",
    "Speed Limit -20-", "Speed Limit -30-", "Speed Limit -40-", "Speed Limit -50-",
    "Speed Limit -60-", "Speed Limit -70-", "Speed Limit -80-", "Speed Limit -90-",
    "Stop Sign", "Traffic Light -Green-", "Traffic Light -Off-", "Traffic Light -Red-",
    "Traffic Light -Yellow-",
    # model_passage (1 classe)
    "crosswalk",
    # model_obs1 (4 classes)
    "Cone", "Safety-barrier", "Safety-bollard", "Safety-cone"
]

print(f"Nombre de classes détectables: {len(all_classes)}")  # Devrait afficher 22

# --- Fonctions utiles ---

def build_observation_vector(detected_labels, current_speed, speed_limit):
    vector = [1 if cls in detected_labels else 0 for cls in all_classes]
    norm_speed = current_speed / maxSpeed
    norm_speed_limit = speed_limit / maxSpeed
    vector.append(norm_speed)
    vector.append(norm_speed_limit)
    print(f"Observation vector length: {len(vector)}")  # Devrait afficher 24
    return vector

def get_current_speed():
    # Vitesse actuelle (en m/s), convertie en km/h pour cohérence si besoin
    # Ici on prend juste driver.getCurrentSpeed() tel quel
    return driver.getCurrentSpeed()

def get_speed_limit_from_labels(detected_labels):
    limits_map = {
        "Speed Limit -20-": 20,
        "Speed Limit -30-": 30,
        "Speed Limit -40-": 40,
        "Speed Limit -50-": 50,
        "Speed Limit -60-": 60,
        "Speed Limit -70-": 70,
        "Speed Limit -80-": 80,
        "Speed Limit -90-": 90,
        "Speed Limit -100-": 100,
        "Speed Limit -120-": 120
    }
    for label in detected_labels:
        if label in limits_map:
            return limits_map[label]
    return maxSpeed

def get_observations():
    detected = current_frame_labels.copy()
    current_speed = get_current_speed()
    speed_limit = get_speed_limit_from_labels(detected)
    obs_vector = build_observation_vector(detected, current_speed, speed_limit)
    return obs_vector

def communicate_with_agent(observation):
    try:
        sock.sendall(json.dumps({"observation": observation}).encode('utf-8') + b'\n')
        data = sock.recv(1024).decode('utf-8')
        action = json.loads(data)
        print(f"Action reçue du SAC : {action}")  # Affichage action reçue
        return action
    except Exception as e:
        print(f"Erreur communication agent : {e}")
        return {"speed": 0, "steering": 0, "brake": 0}

# --- Boucle principale ---
while driver.step() != -1:
    try:
        image2 = camera2.getImage()
        current_frame_labels.clear()
        if image2 is not None:
            width2 = camera2.getWidth()
            height2 = camera2.getHeight()
            image_array2 = np.frombuffer(image2, np.uint8).reshape((height2, width2, 4))
            image_rgb2 = cv2.cvtColor(image_array2, cv2.COLOR_BGRA2RGB)

            has_detection_cam2 = False
            detection_configs = [
                (model_best, (0, 255, 0), 0.8),
                (model_passage, (255, 255, 0), 0.6),
                (model_obs1, (255, 0, 255), 0.9),
            ]

            for model, color, conf_threshold in detection_configs:
                results = model.predict(image_rgb2, save=False, conf=conf_threshold)
                for r in results:
                    for box in getattr(r, 'boxes', []):
                        cls_id = int(box.cls)
                        label = model.names[cls_id]

                        if label == last_detected_label:
                            continue

                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        text = f'{label} {conf:.2f}'

                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        text_x = center_x - text_width // 2
                        text_y = center_y + text_height // 2

                        cv2.rectangle(image_rgb2, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image_rgb2, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        has_detection_cam2 = True
                        current_frame_labels.add(label)

            if current_frame_labels:
                last_detected_label = list(current_frame_labels)[0]

            # Afficher la vitesse actuelle dans l'image caméra
            current_speed = get_current_speed()
            speed_text = f"Speed: {current_speed:.2f} m/s"
            cv2.putText(image_rgb2, speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Détection Caméra 2", cv2.cvtColor(image_rgb2, cv2.COLOR_RGB2BGR))

            if has_detection_cam2:
                filename = os.path.join(save_dir, f"cam2_frame_{image_counter:05d}.png")
                cv2.imwrite(filename, cv2.cvtColor(image_rgb2, cv2.COLOR_RGB2BGR))
                print(f"Image détectée et sauvegardée : {filename}")
                image_counter += 1

        # --- Observation -> action via agent SAC ---
        obs_vector = get_observations()
        print(f"Observation envoyée au SAC (taille {len(obs_vector)}): {obs_vector}")

        action = communicate_with_agent(obs_vector)

        speed = max(0.0, min(maxSpeed, action.get("speed", 0)))
        steering = max(-1.0, min(1.0, action.get("steering", 0)))
        brake = max(0.0, min(1.0, action.get("brake", 0)))

        print(f"Actions appliquées -> Speed: {speed:.2f}, Steering: {steering:.2f}, Brake: {brake:.2f}")

        driver.setCruisingSpeed(speed)
        driver.setSteeringAngle(steering)
        driver.setBrakeIntensity(brake)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Erreur dans la boucle principale : {e}")
        break

cv2.destroyAllWindows()
sock.close()
