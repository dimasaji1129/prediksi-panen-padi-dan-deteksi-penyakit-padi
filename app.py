import numpy as np
from flask import Flask, render_template, request,jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os
import nltk
import pickle
import json
import random
import streamlit as st
import matplotlib.pyplot as plt
import pymysql



app = Flask(__name__)  
app.static_folder = 'static'

nltk.download('popular')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

#loadModel deteksi
disease_model = load_model("deteksi.h5")
model = pickle.load(open("model.pkl","rb"))

# Load the classification model
classification_model = load_model('model/models_baru.h5')

# Load the intents and words
intents = json.loads(open('model/data.json').read())
words = pickle.load(open('model/texts_baru.pkl', 'rb'))
classes = pickle.load(open('model/labels_baru.pkl', 'rb'))

# Function to preprocess the image for the model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img
def preprocess_image_mobile(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


# Deteksi penyakit pada gambar
def detect_disease(image_path):
    img = preprocess_image(image_path)
    predictions = disease_model.predict(img)
    classes = ['BLB', 'BPH', 'Brown_Spot', 'False_Smut', 'Healthy_Plant', 'Hispa', 'Neck_Blast', 'Sheath_Blight_Rot', 'Stemborer']
    result_label = classes[np.argmax(predictions)]
    return result_label

def detect_disease_mobile(image_path):
    img = preprocess_image_mobile(image_path)
    predictions = disease_model.predict(img)
    classes = ['BLB', 'BPH', 'Brown_Spot', 'False_Smut', 'Healthy_Plant', 'Hispa', 'Neck_Blast', 'Sheath_Blight_Rot', 'Stemborer']
    result_label = classes[np.argmax(predictions)]
    return result_label
# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to create a bag of words array
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

# Function to predict the class
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    # Assuming the model is a deep learning model
    p = p.reshape(1, -1)
    res = model.predict(p)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Function to get a random response
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function for chatbot response
def chatbot_response(msg):
    ints = predict_class(msg, classification_model)
    res = get_response(ints, intents)
    return res

# Route for home
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/sentimen')
def sentimen():
    return render_template('sentimen.html')
# Route for index
@app.route("/index")
def index():
    return render_template('index.html')

# Route for disease detection
@app.route("/pendeteksi", methods=["GET", "POST"])
def pendeteksi():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("deteksi.html", error="No image provided")
        image_file = request.files["image"]
        if image_file.filename == "":
            return render_template("deteksi.html", error="No selected image file")
        # Save the uploaded image to a temporary location
        temp_image_path = "temp_image.jpg"
        image_file.save(temp_image_path)
        # Perform disease detection
        result_label = detect_disease(temp_image_path)
        # Get disease explanation based on label
        explanations = {
             'BLB':' Bacterial Leaf Blight, penyakit yang disebabkan oleh bakteri Xanthomonas oryzae pv. oryzae pada tanaman padi. Penyakit ini mempengaruhi daun tanaman padi dan menyebabkan kerugian hasil panen',
        'BPH': 'Brown Planthopper, serangga yang menyerang tanaman padi dan dapat menyebabkan kerusakan yang signifikan. Serangga ini dapat menularkan penyakit virus dan menyebabkan kematian tanaman pada infestasi yang parah',
        'Brown_Spot': 'Penyakit daun coklat, penyakit yang disebabkan oleh jamur Cochliobolus miyabeanus pada tanaman padi. Penyakit ini mempengaruhi daun tanaman padi dan menyebabkan bercak coklat kecil dengan halo kuning',
        'False_Smut': 'Penyakit busuk palsu, penyakit yang disebabkan oleh jamur Ustilaginoidea virens pada tanaman padi. Penyakit ini mempengaruhi biji padi dan menyebabkan biji tersebut berubah menjadi bola busuk besar berwarna hijau kecoklatan',
        'Healthy_Plant': 'Tanaman sehat, tidak terinfeksi oleh penyakit atau hama',
        'Hispa': 'Serangga Hispa, serangga yang menyerang tanaman padi dan dapat menyebabkan kerusakan pada daun. Serangga ini dapat menyebabkan pola makan "jendela" khas dan pengeringan daun. Infestasi yang berat dapat menyebabkan kerugian hasil panen',
        'Neck_Blast': 'Penyakit leher blast, penyakit yang disebabkan oleh jamur Magnaporthe oryzae pada tanaman padi. Penyakit ini mempengaruhi leher dan malai tanaman padi, menyebabkan busuk leher dan kehilangan sebagian atau seluruh biji',
        'Sheath_Blight_Rot': 'Penyakit busuk sarung, penyakit yang disebabkan oleh jamur Rhizoctonia solani pada tanaman padi. Penyakit ini mempengaruhi sarung dan daun tanaman padi, menyebabkan lesi dan pembusukan',
        'Stemborer': 'Penggerek batang, serangga yang menyerang batang tanaman padi dan dapat menyebabkan kerusakan yang signifikan. Serangga ini juga dapat menularkan penyakit dan melemahkan tanaman'
        }
        result_explanation = explanations.get(result_label, 'Unknown disease')
        # Remove the temporary image files
        os.remove(temp_image_path)
        # Render the result on the HTML template
        return render_template("deteksi.html", result_label=result_label,result_explanation=result_explanation)
    else:
        # Handle GET requests to /pendeteksi
        return render_template("deteksi.html")
    

from flask import jsonify  # Impor modul jsonify

@app.route("/mobile", methods=["GET", "POST"])
def mobile():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No image provided"})
        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No selected image file"})
        # Save the uploaded image to a temporary location
        temp_image_path = "temp_image.jpg"
        image_file.save(temp_image_path)
        # Perform disease detection
        result_label = detect_disease_mobile(temp_image_path)
        # Get disease explanation based on label
        explanations = {
             'BLB':' Bacterial Leaf Blight, penyakit yang disebabkan oleh bakteri Xanthomonas oryzae pv. oryzae pada tanaman padi. Penyakit ini mempengaruhi daun tanaman padi dan menyebabkan kerugian hasil panen',
        'BPH': 'Brown Planthopper, serangga yang menyerang tanaman padi dan dapat menyebabkan kerusakan yang signifikan. Serangga ini dapat menularkan penyakit virus dan menyebabkan kematian tanaman pada infestasi yang parah',
        'Brown_Spot': 'Penyakit daun coklat, penyakit yang disebabkan oleh jamur Cochliobolus miyabeanus pada tanaman padi. Penyakit ini mempengaruhi daun tanaman padi dan menyebabkan bercak coklat kecil dengan halo kuning',
        'False_Smut': 'Penyakit busuk palsu, penyakit yang disebabkan oleh jamur Ustilaginoidea virens pada tanaman padi. Penyakit ini mempengaruhi biji padi dan menyebabkan biji tersebut berubah menjadi bola busuk besar berwarna hijau kecoklatan',
        'Healthy_Plant': 'Tanaman sehat, tidak terinfeksi oleh penyakit atau hama',
        'Hispa': 'Serangga Hispa, serangga yang menyerang tanaman padi dan dapat menyebabkan kerusakan pada daun. Serangga ini dapat menyebabkan pola makan "jendela" khas dan pengeringan daun. Infestasi yang berat dapat menyebabkan kerugian hasil panen',
        'Neck_Blast': 'Penyakit leher blast, penyakit yang disebabkan oleh jamur Magnaporthe oryzae pada tanaman padi. Penyakit ini mempengaruhi leher dan malai tanaman padi, menyebabkan busuk leher dan kehilangan sebagian atau seluruh biji',
        'Sheath_Blight_Rot': 'Penyakit busuk sarung, penyakit yang disebabkan oleh jamur Rhizoctonia solani pada tanaman padi. Penyakit ini mempengaruhi sarung dan daun tanaman padi, menyebabkan lesi dan pembusukan',
        'Stemborer': 'Penggerek batang, serangga yang menyerang batang tanaman padi dan dapat menyebabkan kerusakan yang signifikan. Serangga ini juga dapat menularkan penyakit dan melemahkan tanaman'
        }
        result_explanation = explanations.get(result_label, 'Unknown disease')
        # Remove the temporary image files
        os.remove(temp_image_path)
        # Render the result as JSON
        return jsonify({"result_label": result_label, "result_explanation": result_explanation})
    else:
        # Handle GE
        # T requests to /pendeteksi
        return render_template("deteksimobile.html")

# Route for prediction
@app.route("/prediksi", methods=["POST"])
def prediksi():
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    return render_template("index.html", prediction_text="{},kg".format(prediction))

#Route Prediksi Mobile
@app.route("/prediksi_mobile", methods=["POST"])
def prediksi_mobile():
    try:
        # Get the JSON data from the Flutter app
        request_data = request.get_json()

        # Extract features from JSON data
        luas_lahan = float(request_data.get('luas_lahan', 0.0))
        luas_panen = float(request_data.get('luas_panen', 0.0))
        jumlah_bibit = int(request_data.get('jumlah_bibit', 0))

        # Perform prediction using your model
        float_features = [luas_lahan, luas_panen, jumlah_bibit]
        feature = np.array(float_features).reshape(1, -1)
        prediction = model.predict(feature)

        # Format the prediction response
        prediction_text = "{} kg".format(prediction[0])

        # Return the prediction as JSON
        return jsonify({'prediction_text': prediction_text})

    except Exception as e:
        # Handle exceptions
        return jsonify({'error': str(e)})

        
# Route for chat
@app.route("/chat")
def chat():
    return render_template("chatbot.html")

# Route for getting chatbot response
@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    response = chatbot_response(user_text)
    return jsonify(response)

@app.route('/ulasan')
def ulasan():
    return render_template('ulasan.html')

# database konfigurasi
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'review',
}

# Function to insert data into MySQL
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Insert data into the table
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors_headers)

# Route to handle form submission
@app.route('/submit', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(port=5000)