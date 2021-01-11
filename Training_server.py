import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def main():

    df = pd.read_csv("/Users/ruben/Documents/ITC/Exercises/Week_6/heart.csv")

    X = df.drop(columns=["target", 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                         'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    Y = df["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=df['target'])

    print(X_train.head())

    clf = LogisticRegression()
    clf.fit(X_train, Y_train)

    Y_predict = clf.predict(X_test)

    acc = accuracy_score(Y_test, Y_predict)
    conf_mat = confusion_matrix(Y_test, Y_predict)

    print("The accuracy of the prediction is:", acc)
    print("The confusion matrix of the prediction is:", "\n", conf_mat)

    tn, fp, fn, tp = conf_mat.ravel()

    print(f"Number of True positives:", tp)
    print(f"Number of True negatives:", tn)
    print(f"Number of False positive:", fp)
    print(f"Number of False negatives:", fn)

    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

if __name__ == '__main__':
    main()