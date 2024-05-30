import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def iqr(data):
    """
    Elimina outlierii din coloana `Age` folosind metoda Interquartile Range (IQR).
    :param data: Date de intrare
    :return: Date curatate fara outlieri in coloana `Age`
    """
    ages = data["Age"]
    q1 = np.percentile(data["Age"], 25)  # Primul quartil (Q1)
    q3 = np.percentile(data["Age"], 75)  # Al treilea quartil (Q3)
    iqr_var = q3 - q1  # Intervalul interquartil
    low = q1 - 1.5 * iqr_var  # Limita inferioara
    high = q3 + 1.5 * iqr_var  # Limita superioara

    return data[(ages >= low) & (ages <= high)]  # Returneaza datele fara outlieri

def fill_empty(data):
    """
    Completeaza valorile lipsa din coloana `Age` cu media varstelor.
    :param data: Date de intrare
    :return: Date cu valorile lipsa completate
    """
    ages = data["Age"]
    age_mean = ages.mean()  # Calculeaza media varstelor
    ages.fillna(age_mean, inplace=True)  # Completeaza valorile lipsa cu media
    return data

def z_score(data):
    """
    Elimina valorile extreme din coloana `Age` folosind scorul z.
    :param data: Date de intrare
    :return: Date curatate folosind scorul z
    """
    data["Age_z_score"] = zscore(data["Age"])  # Calculeaza scorul z pentru fiecare valoare din `Age`
    max_z_value = 3  # Valoarea maxima a scorului z acceptata
    return data[(data["Age_z_score"] > -max_z_value) & (data["Age_z_score"] < max_z_value)]  # Returneaza datele fara valorile extreme

def verify(data):
    """
    Verifica tipul de date al coloanei `Age` si afiseaza o histograma a distributiei varstelor.
    :param data: Date de intrare
    """
    if data["Age"].dtype != "int64" and data["Age"].dtype != "float64":
        return  # Verifica daca tipul de date al coloanei `Age` este numeric

    plt.title("Age")
    data["Age"].hist()  # Afiseaza o histograma a distributiei varstelor
    plt.show()

def train(training_data):
    """
    Antreneaza un model Random Forest Classifier folosind datele de antrenament curatate.
    :param training_data: Date de antrenament curatate
    :return: Modelul antrenat Random Forest Classifier
    """
    model = RandomForestClassifier()
    training_data = training_data.copy()
    training_data = training_data.drop(["Name", "Ticket", "Cabin", "Age_z_score"], axis=1)  # Elimina coloanele irelevante

    gender_map = {"male": 1, "female": 2}  # Mapare pentru coloana `Sex`
    embarked_map = {"C": 1, "Q": 2, "S": 3}  # Mapare pentru coloana `Embarked`

    training_data["Sex"] = training_data["Sex"].replace(gender_map)  # Transforma valorile categorice in numerice
    training_data["Embarked"] = training_data["Embarked"].replace(embarked_map)

    X = training_data.drop("Survived", axis=1)  # Caracteristicile de antrenament
    y = training_data["Survived"]  # Etichetele de antrenament

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # imparte datele in seturi de antrenament si testare

    model.fit(X_train, y_train)  # Antreneaza modelul
    training_predictions = model.predict(X_test)  # Realizeaza predictii pe setul de testare

    accuracy = accuracy_score(y_test, training_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Afiseaza acuratetea modelului

    importance = model.feature_importances_  # Importanta caracteristicilor
    indices = np.argsort(importance)[::-1]
    names = [X.columns[i] for i in indices]

    plt.figure()
    plt.title(f"RandomForestClassifier Feature Importance")
    plt.bar(range(X.shape[1]), importance[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)  # Afiseaza importanta caracteristicilor
    plt.savefig(f"RandomForestClassifier_feature_importance.png")
    plt.show()

    return model  # Returneaza modelul antrenat

def predict(__real_data, model):
    """
    Foloseste modelul antrenat pentru a prezice supravietuirea pasagerilor din setul de date de test.
    :param __real_data: Date reale de test
    :param model: Modelul antrenat Random Forest Classifier
    """
    real_data: pd.DataFrame = __real_data.copy()

    gender_map = {"male": 1, "female": 2}  # Mapare pentru coloana `Sex`
    embarked_map = {"C": 1, "Q": 2, "S": 3}  # Mapare pentru coloana `Embarked`

    real_data = real_data.drop(["Name", "Ticket", "Cabin"], axis=1)  # Elimina coloanele irelevante
    real_data["Sex"] = real_data["Sex"].replace(gender_map)  # Transforma valorile categorice in numerice
    real_data["Embarked"] = real_data["Embarked"].replace(embarked_map)

    real_x = real_data.drop("Survived", axis=1)  # Caracteristicile de test
    real_y = real_data["Survived"]  # Etichetele de test

    real_predictions = model.predict(real_x)  # Realizeaza predictiile

    accuracy = accuracy_score(real_y, real_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Afiseaza acuratetea predictiilor

    real_data["Survived"] = real_predictions  # Adauga predictiile in datele reale
    real_data.to_csv(f"RandomForestClassifier_predictions.csv")  # Salveaza predictiile intr-un fisier CSV


def main():
    data = pd.read_csv("train.csv")
    data.set_index("PassengerId", inplace=True)

    verify(data)  # Verifica distributia initiala a varstelor

    data = fill_empty(data)  # Completeaza valorile lipsa
    data = iqr(data)  # Elimina outlierii

    data = z_score(data)  # Elimina valorile extreme

    verify(data)  # Verifica distributia varstelor dupa curatare

    model = train(data)  # Antreneaza modelul
    test_data = pd.read_csv("test.csv")
    correct_predictions = pd.read_csv("gender_submission.csv")

    test_data.set_index("PassengerId", inplace=True)
    correct_predictions.set_index("PassengerId", inplace=True)

    test_data = pd.merge(test_data, correct_predictions,
                         on="PassengerId")  # Fuzioneaza datele de test cu predictiile corecte


    predict(test_data, model)  # Realizeaza predictiile folosind modelul antrenat



if __name__ == "__main__":
    main()  # Ruleaza functia principala
