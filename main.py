import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['SECRET_KEY'] = 'tomkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

save_path = os.path.join('static', 'assets')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods = ['GET', 'POST'])
def index():
    image = None
    pokemon = None
    if request.method == "POST":
        f = request.files['file']
        if (allowed_file(f.filename)==True):
            image = os.path.join(save_path, "new_pokemon.png")
            f.save(image)
            predict = cv2.imread(image, cv2.IMREAD_COLOR)
            predict = cv2.resize(predict, (100, 100))
            predict = cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)
            model = load_model('./static/models/model.keras')
            model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
            predict = np.array(predict)/255
            predict = tf.reshape(predict, [-1,100,100,3])
            y = model.predict(predict)
            pokemon = np.argmax(y)
            return render_template('index.html', image=image, pokemon=pokemon, filename=None)
    return render_template('index.html', image=image, pokemon=pokemon, filename=None)

if __name__=="__main__":
    app.run(debug=True)