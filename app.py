from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from apple.pipeline.predict import PredictionPipeline
from apple.utils.common import decodeImage

# Set environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"  
        self.classifier = PredictionPipeline(self.filename)

# Initialize the ClientApp
clapp = ClientApp()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    """
    Trigger the training process.
    """
    try:
        os.system("python main.py")
        return jsonify({"message": "Training successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if not request.is_json:
            return jsonify({"error": "Invalid JSON format"}), 400

        image = request.json.get('image', None)
        if image is None:
            return jsonify({"error": "No image data provided"}), 400

        # Decode and save the image
        decodeImage(image, clapp.filename)

        # Perform prediction
        result = clapp.classifier.predict()

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
