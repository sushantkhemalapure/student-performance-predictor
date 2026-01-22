import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("database.csv")

# Convert result to numbers
data["result"] = data["result"].map({"fail": 0, "pass": 1})

# Features and label
X = data[["study_hours", "attendance", "previous_score"]]
y = data["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test prediction
study_hours = int(input("Enter study hours per day: "))
attendance = int(input("Enter attendance percentage: "))
previous_score = int(input("Enter previous exam score: "))

prediction = model.predict([[study_hours, attendance, previous_score]])

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
