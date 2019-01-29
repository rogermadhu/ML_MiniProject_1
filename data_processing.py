import numpy as np
import json
import string
import operator
from collections import *
import matplotlib.pyplot as plt
from nltk import sent_tokenize, word_tokenize

class OrderedCounter(Counter, OrderedDict):
    pass

"""
 Brainstormed features:
 - sentiment analysis
"""

def get_features(top_words, data):
    num_data_points = len(data)
    num_features = len(top_words) + 4

    x_train = np.zeros((num_data_points, num_features))
    y_train = np.zeros(num_data_points)
    i = 0
    for data_point in data:
        features = np.zeros(num_features)
        word_list = process_string(data_point['text'])
    
        for word in word_list:
            if word in top_words:
                index = top_words.index(word)
                features[index] += 1

        features[num_features - 4] = data_point["controversiality"]
        features[num_features - 3] = data_point["children"]
        features[num_features - 2] = 1 if data_point["is_root"] else 0
        features[num_features - 1] = 1 # bias term

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

def count_top_words(data, num_top_words):
    word_freq = OrderedCounter()
    for line in data:
        word_list = process_string(line['text'])
        word_freq.update(word_list)
    most_common = dict(word_freq.most_common(num_top_words))
    return list(most_common.keys())

def count_words(data):
    words_count = list()
    for line in data:
        count = len(line['text'].split())
        words_count.append(count)
    return list(words_count)

def avg_word_size(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

def avg_words_len(data):
    avg = list()
    for line in data:
        count = avg_word(line['text'])
        avg.append(count)
    return list(avg)

def find_urls(data):
    import re
    count_links = list()
    for line in data:
        count_links.append(len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line['text'])))
    return list(count_links)

def extract_keywords(data):
    import RAKE
    from RAKE import SmartStopList

    stoplist = SmartStopList()
    Rake_obj = RAKE.Rake(stoplist)

    keywords_count = list()
    for line in data:
        keywords = Rake_obj.run(line['text'])
        keywords_count.append(len(keywords))
    return keywords_count

def avg_keyword(data):
    import RAKE
    from RAKE import SmartStopList

    stoplist = SmartStopList()
    Rake_obj = RAKE.Rake(stoplist)
    keywords_avg = list()
    for line in data:
        line_len = len(line['text'].split())
        keywords = Rake_obj.run(line['text'])
        keywords_count = len(keywords)
        if keywords_count > 0:
            keywords_avg.append(float(line_len/keywords_count))
        else:
            keywords_avg.append(0)
    return keywords_avg

def split_data(data, first_split, second_split, third_split):
    train, validation, test = [], [], []

    for i in range(0, first_split):
        train.append(data[i])

    for i in range(first_split, second_split):
        validation.append(data[i])

    for i in range(second_split, third_split):
        test.append(data[i]) 

    return train, validation, test

def calculate_mean_squared_error(predicted_scores, actual_scores):
    total_error = 0
    for x, y in np.nditer([predicted_scores, actual_scores]):
        squared_difference = (x - y)**2 
        total_error += squared_difference
    mean_squared_error = total_error / len(predicted_scores)
    return mean_squared_error

def evaluate_model(top_words, weights, dataset):
    x_validate, y_validate = get_features(top_words, dataset)
    predicted_scores_validate = x_validate.dot(weights)
    actual_scores_validate = y_validate
    error = calculate_mean_squared_error(predicted_scores_validate, actual_scores_validate)
    return error 

def main():
    data = read_json_file()

    """
    Use the first 10,000 points for training, the next 1,000 for validation, and the final 1,000 for testing.
    """
    train, validation, test = split_data(data, 10000, 11000, 12000)

    top_words = count_top_words(train, 160) # top 160 words
    
    x_train, y_train = get_features(top_words, train)

    # """
    # Calculate closed form and gradient descent weights.
    # """
    # closed_form_weights = calculate_closed_form(x_train, y_train)
    # closed_form_error = evaluate_model(top_words, closed_form_weights, train)
    # print("Mean squared error is " + str(closed_form_error) + " for closed form weights on the training set.")

    # weights = np.zeros(164)
    # gradient_descent_weights = calculate_gradient_descent(x_train, y_train, weights, beta=0.1)
    # gradient_descent_error = evaluate_model(top_words, gradient_descent_weights, train)
    # print("Mean squared error is " + str(gradient_descent_error) + " for gradient descent weights on the training set.")
 
    # # print out weights, include in report
    # """
    # Evaluate closed form and gradient descent weights.
    # """
    # closed_form_error = evaluate_model(top_words, closed_form_weights, validation)
    # print("Mean squared error is " + str(closed_form_error) + " for closed form weights on the validation set.")

    # gradient_descent_error = evaluate_model(top_words, gradient_descent_weights, validation)
    # print("Mean squared error is " + str(gradient_descent_error) + " for gradient descent weights on the validation set.")

    # """
    # Create and evaluate models with no text features, top 60 words, and top 160 words
    # """
    # # no text features
    # no_text_x_train, no_text_y_train = get_features([], train)
    # no_text_closed_form_weights = calculate_closed_form(no_text_x_train, no_text_y_train)
    # no_text_training_error = evaluate_model([], no_text_closed_form_weights, train)
    # no_text_validation_error = evaluate_model([], no_text_closed_form_weights, validation)
    # print("MSE is " + str(no_text_training_error) + " for no-text features model on the training set. (closed form)" )
    # print("MSE is " + str(no_text_validation_error) + " for no-text features model on the validation set. (closed form)" )

    # # top 60 words
    # top_words = count_top_words(train, 60)
    # top60_x_train, top60_y_train = get_features(top_words, train)
    # top60_closed_form_weights = calculate_closed_form(top60_x_train, top60_y_train)
    # top60_training_error = evaluate_model(top_words, top60_closed_form_weights, train)
    # top60_validation_error = evaluate_model(top_words, top60_closed_form_weights, validation)
    # print("MSE is " + str(top60_training_error) + " for 60-word features model on the training set. (closed form)" )
    # print("MSE is " + str(top60_validation_error) + " for 60-word features model on the validation set. (closed form)" )

    # # top 160 words
    # top_words = count_top_words(train, 160)
    # top160_x_train, top60_y_train = get_features(top_words, train)
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top60_y_train)
    # top160_training_error = evaluate_model(top_words, top160_closed_form_weights, train)
    # top160_validation_error = evaluate_model(top_words, top160_closed_form_weights, validation)
    # print("MSE is " + str(top160_training_error) + " for 160-word features model on the training set. (closed form)" )
    # print("MSE is " + str(top160_validation_error) + " for 160-word features model on the validation set. (closed form)" )

    cnt_words = count_words(train)
    avg_word_siz = avg_word_size(train)
    avg_words_length = avg_words_len(train)
    fd_urls = find_urls(train)
    ext_keywords = extract_keywords(train)
    avg_kywd = avg_keyword(train)

    

if __name__ == '__main__':
    main()