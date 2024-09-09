from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':

    # Load the iris dataset
    iris = load_iris()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    # Create an MLP without scaling the data
    mlp_no_scaling = MLPClassifier(hidden_layer_sizes=(15,), max_iter=500, random_state=42)
    mlp_no_scaling.fit(X_train, y_train)

    # Evaluate the MLP without scaling
    score_no_scaling = mlp_no_scaling.score(X_test, y_test)
    print(f"Accuracy without scaling: {score_no_scaling}")

    # Create an MLP with scaling the data between 0 and 1
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_scaling = MLPClassifier(hidden_layer_sizes=(13,), max_iter=700, random_state=42)
    mlp_scaling.fit(X_train_scaled, y_train)

    # Evaluate the MLP with scaling
    score_scaling = mlp_scaling.score(X_test_scaled, y_test)
    print(f"Accuracy without scaling: {score_no_scaling}")
    print(f"Accuracy with scaling: {score_scaling}")