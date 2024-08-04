import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def load_mnist_data(train_file):
    
    train_data = pd.read_csv(train_file, header=None, skiprows=1)
    train_data = train_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train = train_data.iloc[:, 1:].values / 255.0 
    y_train = train_data.iloc[:, 0].values.astype(np.int32)  

    return X_train, y_train


def train_model(X_train, y_train):

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def main():
    train_file = 'mnist_train.csv'

    X_train, y_train = load_mnist_data(train_file)

    knn_model = train_model(X_train, y_train)

    model_file = 'mnist_knn_model1.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(knn_model, f)

    print("Trained model saved to:", model_file)

if __name__ == "__main__":
    main()

