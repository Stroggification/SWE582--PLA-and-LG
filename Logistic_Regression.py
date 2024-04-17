import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(training_data, learning_rate=0.01, num_epochs=1000):
    X = training_data[0]
    y = training_data[1]
    num_samples, num_features = X.shape
    w = np.zeros(num_features)  # Initialize weights
    losses = []

    for epoch in range(num_epochs):
        # Forward pass
        y_pred = sigmoid(np.dot(X, w))
        # Calculate loss
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        losses.append(loss)
        # Gradient descent
        gradient = np.dot(X.T, (y_pred - y)) / num_samples
        w -= learning_rate * gradient

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    return w
def min_max_scaling(data):
    
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data

def predict_logistic_regression(model, data):
    return sigmoid(np.dot(data, model))

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    data = df.iloc[:, :7].values
    class_column = df.iloc[:, 7].values

# assign binary labels to the classes "Osmancık as 1 and Cammeo as 0"
    labels = np.where(class_column == "Osmancik", 0, 1)
# Convert labels list to numpy array
    labels = np.array(labels)
    print("Labels array size", labels.size)
    print("labels array:", labels) 
    print("Shape of the data array:", data.shape)
    print("Shape of the labels array:", labels.shape)
    rnd_x = np.array(data)
    rnd_y = labels  
    # rnd_x = np.array([[0, 1], [0.6, 0.6], [1, 0], [1, 1], [0.3, 0.4], [0.2, 0.3], [0.1, 0.4], [0.5, -0.1]])
    # rnd_y = np.array([1, 1, 1, 1, 0, 0, 0, 0]) 
    rnd_x= min_max_scaling(rnd_x)
    rnd_data = [rnd_x, rnd_y]
    trained_model = train_logistic_regression(rnd_data)
    print("Trained model:", trained_model)

    predictions = predict_logistic_regression(trained_model, rnd_x)
    for i in range(len(rnd_x)):
        print("{} -> {:.2f}".format(rnd_x[i], predictions[i]))

    

