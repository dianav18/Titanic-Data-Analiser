import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
#TASK 1

df = pd.read_csv('train.csv')
num_columns = df.shape[1]
print("Number of columns:", num_columns)
print("\nData type for every coloumn:")
print(df.dtypes)
print("\nNumber of absent values for every column:")
print(df.isnull().sum())
num_rows = df.shape[0]
print("\nThe total number of lines:", num_rows)
duplicates = df.duplicated().any()
print("\nThere are duplicated lines:", duplicates)

# TASK 2

survived_rate = df['Survived'].mean() * 100
not_survived_rate = 100 - survived_rate
class_counts = df['Pclass'].value_counts(normalize=True) * 100
gender_counts = df['Sex'].value_counts(normalize=True) * 100
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs[0, 0].bar(['Survived', 'Not Survived'], [survived_rate, not_survived_rate], color=['blue', 'red'])
axs[0, 0].set_title('Survival Rates')
axs[0, 0].set_ylabel('Percentage')
class_counts.plot(kind='bar', ax=axs[0, 1], color='green')
axs[0, 1].set_title('Passenger Class Distribution')
axs[0, 1].set_ylabel('Percentage')
gender_counts.plot(kind='bar', ax=axs[1, 0], color='purple')
axs[1, 0].set_title('Gender Distribution')
axs[1, 0].set_ylabel('Percentage')
plt.tight_layout()

plt.show()

#TASK 3

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
numeric_columns = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
for i, col in enumerate(numeric_columns):
    ax = axs[i//3, i%3]
    df[col].hist(ax=ax, bins=10, color='skyblue', edgecolor='black')
    ax.set_title(f'Histogram for {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()

#TASK 4

missing_data = df.isnull().sum()
missing_columns = missing_data[missing_data > 0]
missing_proportions = (missing_columns / len(df)) * 100
print("Coloumns with missing values:")
print(missing_proportions)
missing_by_survival = df.groupby('Survived').apply(lambda x: x.isnull().sum() / len(x) * 100)
print("\nThe procent for every missing value:")
print(missing_by_survival[missing_columns.index])

#TASK 5

bins = [0, 20, 40, 60, np.inf]
labels = [0, 1, 2, 3]
df['AgeCategory'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
age_category_counts = df['AgeCategory'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
age_category_counts.plot(kind='bar', color='teal')
plt.title('Number of Passengers by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Number of Passengers')
plt.xticks(ticks=np.arange(len(labels)), labels=['0-20', '21-40', '41-60', '61+'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#TASK 6

male_survival = df[(df['Sex'] == 'male') & (df['Survived'] == 1)]['AgeCategory'].value_counts().sort_index()
total_males = df[df['Sex'] == 'male']['AgeCategory'].value_counts().sort_index()
male_survival_rate = (male_survival / total_males) * 100
plt.figure(figsize=(10, 6))
male_survival_rate.plot(kind='bar', color='orange')
plt.title('Survival Rate of Males by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Survival Rate (%)')
plt.xticks(ticks=np.arange(len(labels)), labels=['0-20', '21-40', '41-60', '61+'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#TASK 7

total_passengers = len(df)
num_children = (df['Age'] < 18).sum()
percent_children = (num_children / total_passengers) * 100
children_survival_rate = df[df['Age'] < 18]['Survived'].mean() * 100
adults_survival_rate = df[df['Age'] >= 18]['Survived'].mean() * 100
plt.figure(figsize=(8, 5))
plt.bar(['Children', 'Adults'], [children_survival_rate, adults_survival_rate], color=['blue', 'green'])
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print(f"Procentul de copii la bord: {percent_children:.2f}%")

#TASK 8

mean_age_survived = df[df['Survived'] == 1]['Age'].mean()
mean_age_not_survived = df[df['Survived'] == 0]['Age'].mean()
df.loc[(df['Age'].isnull()) & (df['Survived'] == 1), 'Age'] = mean_age_survived
df.loc[(df['Age'].isnull()) & (df['Survived'] == 0), 'Age'] = mean_age_not_survived
most_common_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(most_common_embarked, inplace=True)
print("Valori lipsă după completare pentru 'Age':", df['Age'].isnull().sum())
print("Valori lipsă după completare pentru 'Embarked':", df['Embarked'].isnull().sum())
df.head()

#TASK 9
df['Title'] = df['Name'].apply(lambda name: re.search(r'\b(\w+)\.', name).group(1) if re.search(r'\b(\w+)\.', name) else 'Unknown')
def check_title_sex(row):
    title_sex_mapping = {
        'Mr': 'male',
        'Miss': 'female',
        'Mrs': 'female',
        'Master': 'male',
        'Don': 'male',
        'Dona': 'female',
        'Lady': 'female',
        'Sir': 'male',
        'Countess': 'female',
        'Dr': 'neutral',
        'Rev': 'neutral',
        'Mme': 'female',
        'Ms': 'female',
        'Major': 'male',
        'Capt': 'male',
        'Jonkheer': 'male'
    }
    expected_sex = title_sex_mapping.get(row['Title'], 'neutral')
    if expected_sex == 'neutral':
        return True
    return row['Sex'] == expected_sex
df['Title_Sex_Correct'] = df.apply(check_title_sex, axis=1)
correct = df['Title_Sex_Correct'].sum()
incorrect = len(df) - correct
print(f"Număr de titluri corecte: {correct}")
print(f"Număr de titluri incorecte: {incorrect}")
title_counts = df['Title'].value_counts()

plt.figure(figsize=(10, 6))
title_counts.plot(kind='bar', color='c')
plt.title('Distribuția titlurilor de noblete în funcție de numărul de persoane')
plt.xlabel('Titlu')
plt.ylabel('Număr de persoane')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#TASK 10

df['Alone'] = ((df['SibSp'] == 0) & (df['Parch'] == 0)).astype(int)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Alone', hue='Survived', multiple='stack', discrete=True, palette='pastel')
plt.title('Survival Based on Being Alone on Titanic')
plt.xlabel('Alone (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Alone', 'Alone'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
sns.catplot(data=df.head(100), x='Pclass', y='Fare', hue='Survived', kind='swarm', height=5, aspect=1.5)
plt.title('Fare vs. Class vs. Survival for the First 100 Records')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.grid(True)
plt.show()
