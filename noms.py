from ultralytics import YOLO

# Chemins des modèles
path_model_best = "D:/self_car_driving/best.pt"
path_model_passage = "D:/self_car_driving/passage.pt"
path_model_obs1 = "D:/self_car_driving/obs1.pt"

def print_model_classes(model, model_name):
    print(f"\nClasses détectées par {model_name} :")
    classes = model.names
    # Si c'est un dict, trier par clé, sinon afficher liste
    if isinstance(classes, dict):
        for class_id in sorted(classes.keys()):
            print(f"  {class_id}: {classes[class_id]}")
    else:
        for i, class_name in enumerate(classes):
            print(f"  {i}: {class_name}")

def main():
    try:
        model_best = YOLO(path_model_best)
        model_passage = YOLO(path_model_passage)
        model_obs1 = YOLO(path_model_obs1)
    except Exception as e:
        print(f"Erreur lors du chargement des modèles : {e}")
        return

    print_model_classes(model_best, "model_best")
    print_model_classes(model_passage, "model_passage")
    print_model_classes(model_obs1, "model_obs1")

if __name__ == "__main__":
    main()
