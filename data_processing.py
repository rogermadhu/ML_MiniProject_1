import numpy as np
import json
import string
import operator
from collections import *
from statsmodels.regression.linear_model import OLS

class OrderedCounter(Counter, OrderedDict):
    pass

# array is 


"""
 Brainstormed features:
 - sentiment analysis
"""

def get_features(top_words, data):
    # Word count feature
    # 100 data points
    x_train = np.zeros((10000, 161))
    y_train = np.zeros(10000)
    i = 0
    for data_point in data:
        features = np.zeros(161)
        word_list = process_string(data_point['text'])
    
        for word in word_list:
            if word in top_words:
                index = top_words.index(word)
                features[index] += 1
        features[160] = 1
        x_train[i] = features
        y_train[i] = data_point["popularity_score"]
        i = i+1

    # TODO: at least two more features: these can be 
    # based on the text data, transformations of the other numeric features, or interaction terms. 
    return x_train, y_train

def calculate_closed_form(X, Y):
    coeffs = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    return coeffs

# def calculate_gradient_descent(X, Y, coeff, bias, learning_rate):
#     # def update_weights(radio, sales, weight, bias, learning_rate):
#     coeff_deriv = 0
#     bias_deriv = 0
#     comments = len(X.shape[0])

#     for i in range(comments):
#             # Calculate partial derivatives
#             # -2x(y - (mx + b))
#         coeff_deriv += -2 * X[i] * (Y[i] - (coeff * X[i] + bias))

#             # -2(y - (mx + b))
#         bias_deriv += -2 * (Y[i] - (coeff * X[i] + bias))

#         # We subtract because the derivatives point in direction of steepest ascent
#     coeff -= (coeff_deriv / comments) * learning_rate
#     bias -= (bias_deriv / comments) * learning_rate

#     return coeff, bias

def hypothesis(x, theta):
	return np.dot(
			np.transpose(theta),
			x
		)

def gradient_descent(x, y, weights, learning_rate, decay_rate, iterations=1500):
	for iteration in range(iterations):
        # for each of the weights
		for j in range(len(weights)):
			gradient = 0
			for i in range(learning_rate):
				gradient += (hypothesis(x[i], weights) - y[i]) * x[i][j]
		gradient *= 1/learning_rate
		weights[j] = weights[j] -  (decay_rate * gradient)
		print(weights)
	return weights	

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

def main():
    data = read_json_file()

    """
    Use the first 10,000 points for training, the next 1,000 for validation, and the final 1,000 for testing.
    """
    train, validation, test = split_data(data, 10000, 11000, 12000)

    top_words = count_top_words(train)
    
    x_train, y_train = get_features(top_words, train)
    coeffs1 = calculate_closed_form(x_train, y_train)
    print("coeffs1")
    print(coeffs1)
    # coeffs2 = gradient_descent(x_train, y_train, np.zeros(coeffs1.shape), 500, 0.1)
    # print("coeffs2")
    # print(coeffs2)
    # create a linear model and extract the parameters
    coeffs_lm = OLS(y_train, x_train).fit().params
    print('test coeffs')
    print(coeffs_lm)

if __name__ == '__main__':
    main()