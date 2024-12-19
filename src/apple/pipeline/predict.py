import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.keras"))

        # Load and preprocess image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(299, 299))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  
        test_image = np.expand_dims(test_image, axis=0)

        # Get probabilities
        probabilities = model.predict(test_image)[0]  
        probabilities = probabilities.tolist()  

        # Check the maximum probability
        max_prob = max(probabilities)
        class_index = np.argmax(probabilities)

        # Map class index to label
        labels = ['Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy']
        if max_prob > 0.7:
            prediction = labels[class_index]
        else:
            prediction = "I cannot detect the disease with confidence."

        # Return only the prediction, no probabilities
        return {"prediction": prediction}
