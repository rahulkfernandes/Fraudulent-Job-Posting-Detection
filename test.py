import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from project import my_model
sys.path.insert(0, '../..')
from sklearn.metrics import classification_report

def test(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(classification_report(y_test, predictions, digits=4))

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("./data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    test(data)
    runtime = (time.time() - start) / 60.0
    print(runtime)