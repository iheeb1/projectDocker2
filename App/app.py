from symbol import tfpdef
from flask import Flask
from flask import render_template, request
app=Flask(__name__)

from werkzeug.utils import secure_filename

import os
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#from PIL import Image
#from keras.applications.vgg19 import VGG19
#from keras.preprocessing import image
#from keras.applications.vgg19 import preprocess_input
#from keras.models import Model
import numpy as np
#from tensorflow.keras.utils import load_img
import scipy.io.wavfile as wavfile
import numpy
import os.path
from os import walk
from scipy import stats
import numpy as np
import librosa 
import numpy as np
from scipy.stats import norm
import pickle
import matplotlib.pyplot as plt
# Import the libraries
import matplotlib.pyplot as plt
from sklearn import svm
#from keras.preprocessing.image import img_to_array
#from keras.applications.vgg19 import preprocess_input

UPLOAD_FOLDER = os.getcwd()  
ALLOWED_EXTENSIONS = {'wav', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    print("no existing")

@app.route("/")
def home():
    return "Hello, Flask!"
    
@app.route('/upload')
def upload():
   return render_template("form.html")



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 
           

	
@app.route('/uploaderSVM', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        audio_base64 = request.form['audio_base64']

        audio_data = base64.b64decode(audio_base64)
        filename = 'uploaded_audio.wav'  
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(audio_data)

        signal, rate = librosa.load(filepath)
        S = librosa.feature.melspectrogram(signal, sr=rate, n_fft=2048, hop_length=512, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        S_DB = S_DB.flatten()[:1200]

        # Load the SVM model
        clf = pickle.load(open('SVM.pkl', 'rb'))

        # Make predictions
        ans = clf.predict([S_DB])[0]
        music_class = str(ans)

        print(music_class)
        return music_class

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode("utf-8")
    

@app.route('/uploaderVGG19', methods=['POST'])
def classify_image():
    data = request.get_json()

    if data and "image_data" in data:
        # Decode base64 data
        encoded_image_data = data["image_data"]
        decoded_image_data = base64.b64decode(encoded_image_data)

        # Save decoded data to a temporary image file
        temp_image_file = '/Nouvarch/shared_volume/temp_image.jpg'
        with open(temp_image_file, 'wb') as temp_file:
            temp_file.write(decoded_image_data)

        # Load and preprocess the image
        img = image.load_img(temp_image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale the pixel values to the range [0, 1]

        # Make predictions
        predictions = model.predict(img_array)

        # Decode the predictions
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        predicted_genre = genres[np.argmax(predictions)]

        response_data = {"received_message": "Image file received and processed successfully",
                         "response": f"Predicted Genre: {predicted_genre}"}
    else:
        response_data = {"received_message": "No image file received", "response": "Error"}

    return jsonify(response_data)
          
          
    
               
      
		
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  
   
