import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


class StrategySuccessClassifier:
    def __init__(self, data_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, data_file)

        self.data = pd.read_csv(file_path)
        self.model = RandomForestClassifier(random_state=42)

    def preprocess_data(self):
        X = self.data[['PitStopLap', 'TyreChoice', 'RacePosition', 'LapTime']]
        y = self.data['StrategySuccess']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, cm, report

if __name__ == "__main__":
    data_file = "example_strategy_data.csv"
    print(f"Expected file path: {os.path.abspath(data_file)}")

    classifier = StrategySuccessClassifier(data_file)
    X_train, X_test, y_train, y_test = classifier.preprocess_data()

    classifier.train_model(X_train, y_train)

    accuracy, cm, report = classifier.evaluate_model(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    example_data = pd.DataFrame({
        'PitStopLap': [15, 30],
        'TyreChoice': [1, 0],  # 1: Soft, 0: Hard
        'RacePosition': [5, 12],
        'LapTime': [90.5, 95.2]
    })
    predictions = classifier.predict(example_data)
    for i, pred in enumerate(predictions):
        print(f"Example {i + 1} predicted strategy success: {'Yes' if pred == 1 else 'No'}")