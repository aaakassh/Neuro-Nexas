# Titanic Survival Prediction Model
import joblib
import pandas as pd

# 1 Load the saved model and scaler
model = joblib.load('logistic_regression_titanic_model.joblib')
scaler = joblib.load('titanic_scaler.joblib')


# 2 Load the cleaned Titanic dataset (assuming it's already preprocessed)
titanic_df = pd.read_csv('tested.csv') 

# 3 Perform the same preprocessing steps:

# Fill missing Fare and Age
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

# Drop Cabin
titanic_df.drop('Cabin', axis=1, inplace=True)

# Encode Sex
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
embarked_dummies = pd.get_dummies(titanic_df['Embarked'], prefix='Embarked')
titanic_df = pd.concat([titanic_df, embarked_dummies], axis=1)

# Drop unused columns
titanic_df.drop(['Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

# 4 Prepare features for prediction
features = titanic_df.drop(['PassengerId', 'Survived'], axis=1)

# 5 Scale features
features_scaled = scaler.transform(features)

# 6 Predict for all passengers
predictions = model.predict(features_scaled)

# 7 Add predictions back to dataframe
titanic_df['Predicted_Survived'] = predictions

# 8 Display result
print(titanic_df[['PassengerId', 'Survived', 'Predicted_Survived']].head(20))

# OUTPUT

    PassengerId  Survived  Predicted_Survived
0           892         0                   0
1           893         1                   1
2           894         0                   0
3           895         0                   0
4           896         1                   1
5           897         0                   0
6           898         1                   1
7           899         0                   0
8           900         1                   1
9           901         0                   0
10          902         0                   0
11          903         0                   0
12          904         1                   1
13          905         0                   0
14          906         1                   1
15          907         1                   1
16          908         0                   0
17          909         0                   0
18          910         1                   1
19          911         1                   1
