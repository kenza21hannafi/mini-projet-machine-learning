#importation des bib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay
import streamlit as st
import pickle

#chargement de données
gender_data = pd.read_csv("C:/Users/user/Desktop/2eme/data mining/gender_submission.csv")
train_data = pd.read_csv("C:/Users/user/Desktop/2eme/data mining/train.csv")
test_data = pd.read_csv("C:/Users/user/Desktop/2eme/data mining/test.csv")

# Prétraitement des données
def pretraitement_data(df):
    # Retirer les colonnes inutiles
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # Traitement des valeurs manquantes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Convertir les attributs catégorielles en numériques
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    return df

train_data = pretraitement_data(train_data)
test_data = pretraitement_data(test_data)

# Séparation des données
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Prédiction et évaluation
y_predict = model.predict(X_test)

KNN_accuracy = accuracy_score(y_test, y_predict)
print("Précision de KNN : ", KNN_accuracy)



# Sauvegarder le modèle pour Streamlit
with open("knn_titanic_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Interface utilisateur avec Streamlit
st.title("Prédiction de survie des passagers du Titanic")
pclass = st.selectbox("Classe du passager (Pclass)", [1, 2, 3])
sex = st.selectbox("Sexe", ["Homme", "Femme"])
age = st.slider("Âge", 0, 80, 25)
fare = st.number_input("Prix du billet (Fare)", min_value=0.0, value=50.0)
embarked = st.selectbox("Port d'embarquement", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

# Charger le modèle
with open("knn_titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Préparation des données utilisateur
sex = 1 if sex == "Femme" else 0
embarked = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}[embarked]

# Préparer les colonnes d'embarquement comme dans le jeu de données d'entraînement
user_data = pd.DataFrame([[pclass, age, fare, sex, 
                           embarked == 0, embarked == 1, embarked == 2]], 
                         columns=['Pclass', 'Age', 'Fare', 'Sex_male', 'Embarked_0', 'Embarked_1', 'Embarked_2'])

# S'assurer que les colonnes de `user_data` sont dans le même ordre que celles d'entraînement
user_data = user_data.reindex(columns=X.columns, fill_value=0)

# Prédire la survie
if st.button("Prédire la survie"):
    prediction = model.predict(user_data)
    result = "Survécu" if prediction[0] == 1 else "Non survécu"
    st.success(f"Résultat : {result}")

