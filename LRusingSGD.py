import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculate_loss(weight, x, y):
    total = 0
    for i in range(len(x)):
        log_value = math.log(1 + np.exp((-1 * y[i] * np.dot(weight.T, x[i]))))
        total = total + log_value
    return total / len(x)

def sigmoid_function(s):
    return 1 / (1 + np.exp(-s))

def calculate_gradients(x, y, y_predicted):
    d_weight = (1 / x.shape[0]) * np.dot(y_predicted - y, x)
    return d_weight

def update_weights(weights, d_weight, learning_rate):
    weights = weights - learning_rate * d_weight
    return weights

def predict(weights, x):
    s = np.dot(x, weights)
    y_predicted = sigmoid_function(s)
    for i in range(len(y_predicted)):
        if y_predicted[i] > 0.5: y_predicted[i] = 1
        else: y_predicted[i] = 0
    return y_predicted

def plot_figures(losses, data, predicted_y):
    losses = np.array(losses)
    plt.plot(losses[:, 0], losses[:, 1])
    plt.xlabel("epoch number")
    plt.ylabel("loss")
    plt.show()

    data_frame = pd.DataFrame(data[:, 1:]) # column added for bias term is excluded
    data_frame['class'] = predicted_y
    data_frame.columns = ['x', 'y', 'class']
    sns.lmplot(data=data_frame, x='x', y='y', hue='class', fit_reg=False, legend=True, legend_out=True)
    plt.show()

def time_based_decay(last_learning_rate, decay_rate, i):
    new_learning_rate = last_learning_rate / (1. + decay_rate * i)
    return new_learning_rate

def main():
    # Loading data files
    x_train_initial = np.load('train_features.npy')
    y_train = np.load('train_labels.npy')
    x_test_initial = np.load('test_features.npy')
    y_test = np.load('test_labels.npy')

    # Adding one column of 1 to x data's for bias
    x_train = np.ones((x_train_initial.shape[0], x_train_initial.shape[1] + 1))
    x_train[:, 1:] = x_train_initial
    x_test = np.ones((x_test_initial.shape[0], x_test_initial.shape[1] + 1))
    x_test[:, 1:] = x_test_initial

    # Preprocessing
    # for y labels: change from (-1,1) to (0,1)
    for i in range(len(y_train)):
        if y_train[i] == -1:
            y_train[i] = 0
    for i in range(len(y_test)):
        if y_test[i] == -1:
            y_test[i] = 0

    row_number, column_number = x_train.shape

    # Initializations
    weights = np.zeros(column_number) # first element of w is used for bias
    epoches = 300
    learning_rate = 0.01
    decay_rate = learning_rate/epoches
    losses = []


    # Main loop: logistic regression
    for i in range(epoches):
        learning_rate = time_based_decay(learning_rate, decay_rate, i)
        random_indexes = list(range(row_number))  # ordered list of up to row_number (number of data point) elements
        np.random.shuffle(random_indexes)         # ordered list is randomly shuffled

        # Loop for stochastic gradient descent
        for j in range(row_number):
            data_x, data_y = x_train[random_indexes[j]], y_train[random_indexes[j]] # get a random data point
            s = np.dot(data_x, weights)
            y_predicted = sigmoid_function(s)
            d_weights = calculate_gradients(data_x, data_y, y_predicted)
            weights = update_weights(weights, d_weights, learning_rate)
        loss = calculate_loss(weights, x_train, y_train)
        losses.append([i,loss])

    y_pred_test = predict(weights, x_test)
    y_pred_train = predict(weights, x_train)

    print("*" * 50)
    print("Logistic Regression using Stochastic Gradient Descent")
    print("*" * 50)

    correct_predictions = 0
    for i in range(len(y_train)):
        if y_train[i] == y_pred_train[i]:
            correct_predictions = correct_predictions + 1
        i = i + 1
    print("Accuracy on train data:  ", (correct_predictions / i) * 100)


    correct_predictions = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred_test[i]:
            correct_predictions = correct_predictions + 1
        i = i + 1
    print("Accuracy on test data :  ", (correct_predictions / i) * 100)

    plot_figures(losses, x_test, y_pred_test)

if __name__ == "__main__":
    main()
