from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy
import google.generativeai as genai
import os
import numpy as np
import pandas as pd
import pickle
import requests
from prettytable import PrettyTable

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure key for production use

# Configure Gemini API directly
api_key = "AIzaSyC0ANqOJkp2lH3wWNzfrIDbumD6x4Otkac"  # Replace with your actual Gemini API key
genai.configure(api_key=api_key)

# Set up the model
model = genai.GenerativeModel('gemini-1.5-pro')

# News API configuration
NEWS_API_KEY = "d329f014918a4271b5d49c3f9b66a707"  # Replace with your NewsAPI.org API Key
NEWS_URL = f"https://newsapi.org/v2/top-headlines?category=health&apiKey={NEWS_API_KEY}"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Load datasets and model
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper functions
def helper(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    desc = " ".join(desc) if len(desc) > 0 else "No description available"

    pre = precautions[precautions['Disease'] == disease]
    pre_list = pre.iloc[0].values[1:].tolist() if not pre.empty else ["No precautions available"]

    med = medications[medications['Disease'] == disease]['Medication'].values
    med_list = list(med) if len(med) > 0 else ["No medications available"]

    diet = diets[diets['Disease'] == disease]['Diet'].values
    diet_list = list(diet) if len(diet) > 0 else ["No diet recommendations available"]

    wrkout = workout[workout['disease'] == disease]['workout'].values
    workout_list = list(wrkout) if len(wrkout) > 0 else ["No workout recommendations available"]

    return desc, pre_list, med_list, diet_list, workout_list

# Symptom and disease mappings
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    input_vector = input_vector.reshape(1, -1)
    prediction = svc.predict(input_vector)[0]
    return diseases_list.get(prediction, "Disease not found")

# HTML template with inline CSS
with open('templates/askAI.html', 'r', encoding='utf-8') as file:
    HTML_TEMPLATE = file.read()

# Routes
@app.route("/")
def login():
    users=User.query.all()
    table=PrettyTable()
    table.field_names = ["id", "Username"]
    for user in users:
        table.add_row([user.id, user.username])
    print(table)
    return render_template("login.html")

@app.route("/index")
def index():
    if 'logged_in' in session and session['logged_in']:
        return render_template("index.html")
    return redirect(url_for('login'))

@app.route("/login", methods=["POST"])
def handle_login():
    username = request.form.get('username')
    password = request.form.get('password')
    user = User.query.filter_by(username=username, password=password).first()

    if user:
        session['logged_in'] = True
        return redirect(url_for('index'))
    else:
        flash("Invalid username or password.", "danger")
        return redirect(url_for('login'))

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')

    if password != confirm_password:
        flash("Passwords do not match!", "danger")
        return redirect(url_for('login'))

    if User.query.filter_by(username=username).first():
        flash("Username already exists.", "danger")
        return redirect(url_for('login'))

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    flash("Registration successful! You can now log in.", "success")
    return redirect(url_for('login'))

@app.route("/login-signup")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route("/predict", methods=["POST"])
def predict():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))

    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '')

        if not symptoms or symptoms.lower() == "symptoms":
            return render_template("index.html", message="Please enter valid symptoms.")

        user_symptoms = [s.strip().lower() for s in symptoms.split(',') if s.strip()]
        predicted_disease = get_predicted_value(user_symptoms)

        dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

        return render_template("index.html",
                               predicted_disease=predicted_disease,
                               dis_des=dis_des,
                               my_precautions=precautions,
                               medications=medications,
                               my_diet=rec_diet,
                               workout=workout)
    return render_template("index.html")

# Initialize database (move this code directly after the Flask app initialization)
with app.app_context():
    db.create_all()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and get responses from Gemini."""
    data = request.json
    user_message = data.get('message', '')
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Format chat history for Gemini
        formatted_history = []
        for entry in chat_history:
            if entry.get('role') == 'user':
                formatted_history.append({"role": "user", "parts": [entry.get('content')]})
            else:
                formatted_history.append({"role": "model", "parts": [entry.get('content')]})

        # Create a chat session
        chat = model.start_chat(history=formatted_history)

        # Generate response
        response = chat.send_message(user_message)

        return jsonify({
            "response": response.text,
            "success": True
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/askAI")
def ask():
    return render_template("askAI.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/developer")
def developer():
    return render_template("hitw.html")

@app.route("/blog")
def blog():
    return render_template("blog.html")

# New medical news routes
@app.route('/medical-news')
def get_news():
    response = requests.get(NEWS_URL)
    data = response.json()
    return jsonify(data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Disable Flask's dotenv loading to avoid the encoding error
    app.run(debug=True, load_dotenv=False)