import numpy as np
from random import choice
import matplotlib.pyplot as plt




def train_perceptron(training_data):
    '''
    Train a perceptron model given a set of training data
    :param training_data: A list of data points, where training_data[0]
    contains the data points and training_data[1] contains the labels.
    Labels are +1/-1.
    :return: learned model vector
    '''
    X = training_data[0]
    y = training_data[1]
    model_size = X.shape[1]
    w = np.zeros(model_size)
    # w = [1, 2, -2]
    iteration = 0
    while True:
        misclassified = []
        for i, x in enumerate(X):
            if np.dot(X[i], w) * y[i] <= 0:
                misclassified.append(i)
        if not misclassified:
            break
    #chose random element
        idx = choice(misclassified)
        w += y[idx] * X[idx]
        print("Weight vectors", w)
        iteration += 1
    print("Converged after {} iterations".format(iteration))
    return w

def print_prediction(model,data):
    '''
    Print the predictions given the dataset and the learned model.
    :param model: model vector
    :param data:  data points
    :return: nothing
    '''
    result = np.matmul(data,model)
    predictions = np.sign(result)
    for i in range(len(data)):
        print("{}: {} -> {}".format(data[i][:2], result[i], predictions[i]))

def plot_decision_boundary(X, y, w):
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=y)
    
    # Plotting decision boundary
    x_vals = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    y_vals = (-w[0] * x_vals - w[2]) / w[1]
    plt.plot(x_vals, y_vals, 'r--')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Points and Decision Boundary')
    plt.show()

if __name__ == '__main__':
    
    data_small = np.load('data_small.npy')
    data_large = np.load('data_large.npy')

    # for the given data set the bias is the first index change [data, data, bias] to [bias, data,data]
    rnd_x_small = np.array([arr[::-1] for arr in data_small])
    rnd_x_large = np.array([arr[::-1] for arr in data_large])

    rnd_y_small = np.load('label_small.npy')
    rnd_data_small = [rnd_x_small,rnd_y_small]
    rnd_y_large = np.load('label_large.npy')
    rnd_data_large = [rnd_x_large,rnd_y_large]

    #print model for small dataset
    trained_model_small = train_perceptron(rnd_data_small)
    print("Trained model:", trained_model_small)
    print_prediction(trained_model_small, rnd_x_small)
    plot_decision_boundary(rnd_x_small[:,:2], rnd_y_small, trained_model_small)
    print(rnd_x_small)

    #print model for large dataset
    trained_model_large = train_perceptron(rnd_data_large)
    print("Trained model:", trained_model_large)
    print_prediction(trained_model_large, rnd_x_large)
    plot_decision_boundary(rnd_x_large[:,:2], rnd_y_large, trained_model_large)
    print(rnd_x_large)
   
