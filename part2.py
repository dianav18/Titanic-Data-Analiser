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



def main():
    data = pd.read_csv("train.csv")
    data.set_index("PassengerId", inplace=True)

    verify(data)  # Verifica distributia initiala a varstelor

    data = fill_empty(data)  # Completeaza valorile lipsa
    data = iqr(data)  # Elimina outlierii

    data = z_score(data)  # Elimina valorile extreme

    verify(data)  # Verifica distributia varstelor dupa curatare


if __name__ == "__main__":
    main()  # Ruleaza functia principala
