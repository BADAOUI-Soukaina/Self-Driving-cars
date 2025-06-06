# Self-Driving-cars# 🚗 

Ce projet simule une voiture autonome dans Webots. Elle est capable de détecter des objets avec YOLOv8, suivre la route avec la caméra, et apprendre à conduire grâce à un agent d’apprentissage par renforcement(SAC).

## 🛠 Technologies utilisées

- Webots
- Python 3
- OpenCV
- YOLOv8 (Ultralytics)
- Sockets (communication entre Webots et Python)
- OpenAI Gym
- SAC

## 📸 Fonctionnalités

- Détection des feux tricolores, panneaux STOP, limitations de vitesse, obstacles, etc.
- Suivi de ligne avec la caméra
- Communication en temps réel entre Webots et Python
- Environnement personnalisé pour entraîner un agent intelligent

## ▶ Lancer le projet

1. Ouvrir Webots et lancer la simulation
2. Exécuter le fichier Python controller.py
3. Lancer l'environnement env_webots.py

## 📁 Fichiers importants

- controller.py : code du contrôleur Webots
- env_webots.py : environnement Gym pour le RL
- YOLO_models/ : modèles entraînés YOLO
- world/ : simulation Webots

## 📌 Objectif

Créer une voiture autonome qui détecte son environnement et prend des décisions intelligentes pour rouler de façon autonome.
