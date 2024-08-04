import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(model_file):

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def load_mnist_test_data(test_file):

    test_data = pd.read_csv(test_file, header=None)
    test_data = test_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_test = test_data.iloc[:, 1:].values / 255.0  
    y_test = test_data.iloc[:, 0].values.astype(np.int32) 

    return X_test, y_test

def test_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test data:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
   

    plt.savefig("confusion_matrix.png")

    print("Predicted Labels:", y_pred)
    print("Actual Labels:", y_test)

def main():

    model_file = 'mnist_knn_model1.pkl' 
    test_file = 'mnist_test.csv'  

    knn_model = load_model(model_file)
    X_test, y_test = load_mnist_test_data(test_file)

    test_model(knn_model, X_test, y_test)

if __name__ == "__main__":
    main()

