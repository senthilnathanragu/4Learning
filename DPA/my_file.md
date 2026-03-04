import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = "http://raw.githubusercontent.com/senthilnathan-guvi/public_data/refs/heads/main/jobs.csv"
df = pd.read_csv(data)

X = df.drop(columns=["career_path"])
y = df['career_path']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score

joblib.dump(model, 'job_predictor.joblib')

my_model = joblib.load('job_predictor.joblib')
predictoins = my_model.predict(X_test)
predictoins
