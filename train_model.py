import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


data = pd.read_csv("hallucination_dataset.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")


features = []
labels = data["label"]


for _, row in data.iterrows():

    answer_embedding = model.encode([row["ai_answer"]])
    evidence_embedding = model.encode([row["evidence"]])

    score = (
        answer_embedding @ evidence_embedding.T
    )[0][0]

    features.append([score])


X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)


classifier = LogisticRegression()

classifier.fit(
    X_train,
    y_train
)


predictions = classifier.predict(X_test)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

matrix = confusion_matrix(
    y_test,
    predictions
)


print("\n--- MODEL RESULTS ---")

print("Dataset size:", len(data))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1-Score:", round(f1 * 100, 2), "%")


print("\nConfusion Matrix:")
print(matrix)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Supported",
            "Hallucinated"
        ]
    )
)


joblib.dump(
    classifier,
    "hallucination_model.pkl"
)

print("\nModel saved as hallucination_model.pkl")