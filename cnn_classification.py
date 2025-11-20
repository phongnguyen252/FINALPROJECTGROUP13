import os
import json
import numpy as np
import tensorflow as tf

BASE_DIR = "D:/IR_challenge"
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_food_classifier.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

class CNNFoodClassifier:
    def __init__(self, model_path=MODEL_PATH, class_path=CLASS_NAMES_PATH):
        self.model = tf.keras.models.load_model(model_path)
        with open(class_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

    def predict_image(self, image_path, img_size=(128, 128)):
        '''Dự đoán lớp của một ảnh'''
        img = tf.keras.utils.load_img(image_path, target_size=img_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        preds = self.model.predict(img_array, verbose=0)

        idx = np.argmax(preds[0])
        confidence = float(preds[0][idx])
        label = self.class_names[idx]
        return {
            "path": image_path,
            "predicted_class": label,
            "confidence": confidence
        }
