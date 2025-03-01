import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the dataset
dataset = pd.read_csv('Training.csv')
X = dataset.drop('prognosis', axis=1)  # Features
y = dataset['prognosis']              # Target

# 2. Encode prognosis labels
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# 4. Train SVC Model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Save the trained model
pickle.dump(svc, open('svc.pkl', 'wb'))

# 5. Load Required Data Files
description = pd.read_csv("description.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv('medications.csv')
workout = pd.read_csv("workout_df.csv")
diets = pd.read_csv("diets.csv")

# 6. Helper Function
def helper(disease):
    desc = description[description['Disease'] == disease]['Description'].values[0]
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
    med = medications[medications['Disease'] == disease]['Medication'].values[0]
    diet = diets[diets['Disease'] == disease]['Diet'].values[0]
    wrkout = workout[workout['disease'] == disease]['workout'].values[0]
    return desc, pre, med, diet, wrkout

# 7. Function for Prediction
def get_predicted_value(user_symptoms):
    # Create a zero-vector DataFrame with the same column names as X_train
    input_vector = pd.DataFrame(np.zeros((1, len(X.columns))), columns=X.columns)
    
    # Set the symptoms provided by the user to 1
    for symptom in user_symptoms:
        if symptom in X.columns:
            input_vector[symptom] = 1

    # Predict the disease
    prediction = svc.predict(input_vector)[0]  # Pass the DataFrame with feature names
    return le.inverse_transform([prediction])[0]  # Decode back to disease name

# 8. Main Code: User Input
if __name__ == "__main__":
    print("========== Medical Recommendation System ==========")
    symptoms = input("Enter your symptoms separated by commas: ")
    user_symptoms = [s.strip() for s in symptoms.split(',')]
    
    try:
        # Predict Disease
        predicted_disease = get_predicted_value(user_symptoms)
        desc, pre, med, diet, wrkout = helper(predicted_disease)

        # Display Results
        print("\n================= Predicted Disease ==============")
        print(predicted_disease)
        print("================= Description ====================")
        print(desc)
        print("================= Precautions ====================")
        for i, p in enumerate(pre, start=1):
            print(f"{i}. {p}")
        print("================= Medications ====================")
        print(med)
        print("================= Recommended Diet ===============")
        print(diet)
        print("================= Recommended Workout ============")
        print(wrkout)

    except Exception as e:
        print(f"Error: {e}")
