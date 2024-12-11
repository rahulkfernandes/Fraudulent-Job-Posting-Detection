import time
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from project import my_model
sys.path.insert(0, '../..')
import numpy as np
from scipy.stats import norm
from sklearn.metrics import classification_report

def test(data):
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("\nClassification Report (Test Set):")
    report = classification_report(y_test, predictions, digits=4)
    print(report)
    f1 = report[str(1)]['f1-score']
    return f1

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("./data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")

    f1_scores = np.array([])
    for i in range(0,25):
        print(f'=====================Iteration {i}====================')
        f1 = test(data)
        print(f'F1 Score for iteration {i} =', f1)
        f1_scores = np.append(f1_scores, f1)
    mean = f1_scores.mean()
    std = f1_scores.std()
    print('F1 score stats for target class 1:')
    print('Mean =', mean)
    print('Median =', np.median(f1_scores))
    print('Min =', np.min(f1_scores))
    print('Max =', np.max(f1_scores))
    print('STD =', std)
    print('Var =', np.var(f1_scores))

    percentile_10 = np.percentile(f1_scores, 10)
    percentile_25 = np.percentile(f1_scores, 25)
    print('Below 10th percentile =', percentile_10)
    print('Below 25th percentile =', percentile_25)

    prob_below_0_75 = norm.cdf(0.75, loc=mean, scale=std)
    print('prob_below_0_75 for MLP, LR & RF =', prob_below_0_75)
    # print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print(runtime)