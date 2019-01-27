import numpy as np
import json
import string
import operator
from collections import *

class OrderedCounter(Counter, OrderedDict):
    pass

"""
 Brainstormed features:
 - sentiment analysis
"""

def get_features(top_words, data):
    # Word count feature
    # 100 data points
    x_train = np.zeros((10000, 164))
    y_train = np.zeros(10000)
    i = 0
    for data_point in data:
        features = np.zeros(164)
        word_list = process_string(data_point['text'])
    
        for word in word_list:
            if word in top_words:
                index = top_words.index(word)
                features[index] += 1

        features[160] = data_point["controversiality"]
        features[161] = data_point["children"]
        features[162] = 1 if data_point["is_root"] else 0
        features[163] = 1

        x_train[i] = features
        y_train[i] = data_point["popularity_score"]
        i = i+1

    # TODO: at least two more features: these can be 
    # based on the text data, transformations of the other numeric features, or interaction terms. 
    return x_train, y_train

def calculate_closed_form(X, Y):
    coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    return coeffs

def calculate_gradient_descent(X, y, init_weights, beta=10**-9, alpha=10**-6, epsilon=4*10**-5):
    error = 10
    new_weights = init_weights

    while error > epsilon:
        weights = new_weights

        # the error was here, Prof mentioned in class that the learning rate
        # should be scaled by 1/n where n = # training points
        gradient = X.T.dot(X.dot(weights) - y)
        new_weights = weights - 2* alpha/(1+beta) * gradient
        error = np.linalg.norm((new_weights - weights),2)
        # print(error)

    return new_weights

def read_json_file():
    file = open('./data/proj1_data.json', 'r')
    data= file.read()
    file.close()
    json_data = json.loads(data)
    return json_data

def process_string(str):
    return str.lower().split(' ')

def count_top_words(data):
    print("Finding top words")
    word_freq = OrderedCounter()
    for line in data:
        word_list = process_string(line['text'])
        word_freq.update(word_list)
    most_common = dict(word_freq.most_common(160))
    return list(most_common.keys())

def split_data(data, first_split, second_split, third_split):
    train, validation, test = [], [], []

    for i in range(0, first_split):
        train.append(data[i])
    print(("Created training dataset. Sample from training data: {}").format(train[0]))

    for i in range(first_split, second_split):
        validation.append(data[i])
    print(("Created validation dataset. Sample from validation data: {}").format(validation[0]))

    for i in range(second_split, third_split):
        test.append(data[i]) 
    print(("Created test dataset. Sample from validation data: {}").format(test[0]))

    return train, validation, test

def calculate_mean_squared_error(predicted_scores, actual_scores):
    total_error = 0
    for x, y in np.nditer([predicted_scores, actual_scores]):
        squared_difference = (x - y)**2 
        total_error += squared_difference
    mean_squared_error = total_error / len(predicted_scores)
    return mean_squared_error

def main():
    data = read_json_file()

    """
    Use the first 10,000 points for training, the next 1,000 for validation, and the final 1,000 for testing.
    """
    train, validation, test = split_data(data, 10000, 11000, 12000)

    top_words = count_top_words(train)
    
    x_train, y_train = get_features(top_words, train)

    closed_form_weights = calculate_closed_form(x_train, y_train)
    print("Closed Form weights: \n", closed_form_weights)

    weights = np.zeros(164)
    gradient_descent_weights = calculate_gradient_descent(x_train, y_train, weights, beta=0.1)
    print("Gradient Descent weights: \n", gradient_descent_weights)
    
    print("Evaluating weights on validation dataset (closed form)")
    x_validate, y_validate = get_features(top_words, validation)
    predicted_scores_validate = x_validate.dot(closed_form_weights)
    actual_scores_validate = y_validate

    error = calculate_mean_squared_error(predicted_scores_validate, actual_scores_validate)
    print("Mean squared error is " + str(error) + " for closed form weights.")

    print("Evaluating weights on validation dataset (gradient descent)")
    predicted_scores_validate = x_validate.dot(gradient_descent_weights)
    actual_scores_validate = y_validate

    error = calculate_mean_squared_error(predicted_scores_validate, actual_scores_validate)
    print("Mean squared error is " + str(error) + " for gradient descent weights.")

    

if __name__ == '__main__':
    main()