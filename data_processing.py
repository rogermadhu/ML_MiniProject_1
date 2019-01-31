import numpy as np
import json
import string
import operator
from collections import *
import matplotlib.pyplot as plt
from textblob import TextBlob
import RAKE
from RAKE import SmartStopList

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

def get_features_extra(top_words, data, features_list, f_size):
    
    num_data_points = len(data)
    num_features = len(top_words) + 4 + f_size

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
        temp_f_size = f_size
        if len(features_list) > 0:
            for feats in features_list:
                if feats == "count_words":
                    features[num_features - 4 - temp_f_size] = count_words(data_point["text"])
                    temp_f_size += -1
                elif feats == "avg_words_len":
                    features[num_features - 4 - temp_f_size] = avg_words_len(data_point["text"])
                    temp_f_size += -1
                elif feats == "find_urls":
                    features[num_features - 4 - temp_f_size] = find_urls(data_point["text"])
                    temp_f_size += -1
                elif feats == "extract_keywords":
                    features[num_features - 4 - temp_f_size] = extract_keywords(data_point["text"])
                    temp_f_size += -1
                elif feats == "avg_keyword":
                    features[num_features - 4 - temp_f_size] = avg_keyword(data_point["text"])
                    temp_f_size += -1
                elif feats == "sentiment":
                    features[num_features - 4 - temp_f_size] = sentiment_analyzer(data_point["text"])
                    temp_f_size += -1
                elif feats == "bigrams":
                    features[num_features - 4 - temp_f_size] = bigrams(data_point["text"])
                    temp_f_size += -1
                elif feats == "sc":
                    features[num_features - 4 - temp_f_size] = specialchar_children(data_point["text"], data_point["children"])
                    temp_f_size += -1
                elif feats == "kc":
                    features[num_features - 4 - temp_f_size] = keyword_children(data_point["text"], data_point["children"])
                    temp_f_size += -1
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

def count_words(str):
    count = len(str.split())
    return count

def avg_words_len(str):
  words = str.split()
  return (sum(len(word) for word in words)/len(words))

def find_urls(str):
    import re
    num_links = len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str))
    return num_links

def extract_keywords(str):
    stoplist = SmartStopList()
    Rake_obj = RAKE.Rake(stoplist)
    keywords_count = len(Rake_obj.run(str))
    return keywords_count
#
# def avg_word_len(str):
#     words = str.split()
#     total = 0
#     for w in words:
#         total = total+len
#     stoplist = SmartStopList()
#     Rake_obj = RAKE.Rake(stoplist)
#     line_len = len(str.split())
#     keywords = Rake_obj.run(str)
#     keywords_count = len(keywords)
#     if keywords_count > 0:
#         return float(line_len/keywords_count)
#     else:
#         return 0

def avg_keyword(str):
    stoplist = SmartStopList()
    Rake_obj = RAKE.Rake(stoplist)
    line_len = len(str.split())
    keywords = Rake_obj.run(str)
    keywords_count = len(keywords)
    if keywords_count > 0:
        return float(line_len/keywords_count)
    else:
        return 0

def sentiment_analyzer(str):
    blob = TextBlob(str)
    sentiment = blob.sentiment
    return sentiment.polarity

def sentiment_subjectivity(str):
    blob = TextBlob(str)
    sentiment = blob.sentiment
    return sentiment.subjectivity

def bigrams(str):
    bigrams = TextBlob(str).ngrams(2)
    return len(bigrams)

def keyword_children(text, children):
    # Keyword to children weight
    # How the number of keywords in the text creates further discussion through children
    # without the -1 the validation MSE doesn't change by a lot, using -1 we add bias towards comments which has 1 children, ignore comments with 1 children
    # function adds more weight to the comments with more than 1 children
    if (extract_keywords(text) > 1) and (children > 0):
        if children > 2:
            return children-1
        return 1
    return 0

def specialchar_children(text, children):
    # Amount of special characters lead to more popularity and causes more children
    import re
    ret = re.split(r'[\(\n)\\_+\[\]\:|<,\.\.\./]', text)
    if len(ret) > 1:
        if children > 2:
            return 2
    return 0

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

def evaluate_model_extra_feats(top_words, weights, dataset, features_lst, feat_size):
    x_validate, y_validate = get_features_extra(top_words, dataset, features_lst, feat_size)
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
    
    #x_train, y_train = get_features(top_words, train)

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
    # top160_x_train, top160_y_train = get_features(top_words, train)
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model(top_words, top160_closed_form_weights, train)
    # top160_validation_error = evaluate_model(top_words, top160_closed_form_weights, validation)
    # print("MSE is " + str(top160_training_error) + " for 160-word features model on the training set. (closed form)" )
    # print("MSE is " + str(top160_validation_error) + " for 160-word features model on the validation set. (closed form)" )

    # """
    # Task 3: 3. Using Closed-Form approach Introducing combination of six new features proposed improve performance on the validation set
    # List of Features ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
    # Just copy paste in features_lst
    # """

#################################################################################################################################
    # # top 60 words + extra features => best model
    best_features_lst = ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword", "sc", "kc"]
    # for word in all_features_lst:
    top_words = count_top_words(train, 60)
    top60_x_train, top60_y_train = get_features_extra(top_words, train, best_features_lst, len(best_features_lst))

    top60_closed_form_weights = calculate_closed_form(top60_x_train, top60_y_train)
    top60_training_error = evaluate_model_extra_feats(top_words, top60_closed_form_weights, train, best_features_lst, len(best_features_lst))
    top60_validation_error = evaluate_model_extra_feats(top_words, top60_closed_form_weights, validation, best_features_lst, len(best_features_lst))
    top60_test_error = evaluate_model_extra_feats(top_words, top60_closed_form_weights, test, best_features_lst, len(best_features_lst))

    print("MSE is " + str(top60_training_error) + ' for 60-word + [count_words, avg_words_len, find_urls, extract_keywords, avg_keyword, sc, kc] features model on the training set. (closed form)' )
    print("MSE is " + str(top60_validation_error) + ' for 60-word + [count_words, avg_words_len, find_urls, extract_keywords, avg_keyword, sc, kc] features model on the validation set. (closed form)' )
    print("MSE is " + str(top60_test_error) + ' for 60-word + [count_words, avg_words_len, find_urls, extract_keywords, avg_keyword, sc, kc] features model on the test set. (closed form)' )

#################################################################################################################################

    # # top 160 words
    # top_words = count_top_words(train, 160)
    # features_lst = ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 160-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]' )
    # print("MSE is " + str(validation_error) + ' for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]' )
    # print('')

    # top_words = count_top_words(train, 160)
    # features_lst = ["count_words","avg_words_len","find_urls","extract_keywords"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 160-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]' )
    # print("MSE is " + str(validation_error) + ' for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]' )
    # print('')

    # top_words = count_top_words(train, 160)
    # features_lst = ["count_words","avg_words_len","find_urls"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 160-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls"]' )
    # print("MSE is " + str(validation_error) + ' for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls"]' )
    # print('')

    # top_words = count_top_words(train, 160)
    # features_lst = ["count_words","avg_words_len"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 160-word features model on the training set. (closed form) ["count_words","avg_words_len"]' )
    # print("MSE is " + str(validation_error) + ' for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len"]' )

    # top_words = count_top_words(train, 160)
    # features_lst = ["extract_keywords","avg_keyword"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 160-word features model on the training set. (closed form) ["extract_keywords","avg_keyword"]' )
    # print("MSE is " + str(validation_error) + ' for 160-word features model on the validation set. (closed form) ["extract_keywords","avg_keyword"]' )

    # top_words = count_top_words(train, 160)
    # features_lst = ["count_words","extract_keywords"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 160-word features model on the training set. (closed form) ["count_words","extract_keywords"]' )
    # print("MSE is " + str(validation_error) + ' for 160-word features model on the validation set. (closed form) ["count_words","extract_keywords"]' )
    
    # print("starting 60")

    # top_words = count_top_words(train, 60)
    # features_lst = ["kc", "sc"]
    # top60_x_train, top60_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top60_closed_form_weights = calculate_closed_form(top60_x_train, top60_y_train)
    # top60_training_error = evaluate_model_extra_feats(top_words, top60_closed_form_weights, train, features_lst, len(features_lst))
    # top60_validation_error = evaluate_model_extra_feats(top_words, top60_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top60_training_error)
    # validation_error = str(top60_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["sentiment"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["sentiment"]' )

    # # top 60 words
    # top_words = count_top_words(train, 60)
    # features_lst = ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]' )
    # print('')

    # top_words = count_top_words(train, 60)
    # features_lst = ["count_words","avg_words_len","find_urls","extract_keywords"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]' )
    # print('')

    # top_words = count_top_words(train, 60)
    # features_lst = ["count_words","avg_words_len","find_urls"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls"]' )
    # print('')

    # top_words = count_top_words(train, 60)
    # features_lst = ["count_words","avg_words_len"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["count_words","avg_words_len"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len"]' )

    # top_words = count_top_words(train, 60)
    # features_lst = ["extract_keywords","avg_keyword"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["extract_keywords","avg_keyword"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["extract_keywords","avg_keyword"]' )

    # top_words = count_top_words(train, 60)
    # features_lst = ["sentiment"]
    # top160_x_train, top160_y_train = get_features_extra(top_words, train, features_lst, len(features_lst))
    # top160_closed_form_weights = calculate_closed_form(top160_x_train, top160_y_train)
    # top160_training_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, train, features_lst, len(features_lst))
    # top160_validation_error = evaluate_model_extra_feats(top_words, top160_closed_form_weights, validation, features_lst, len(features_lst))
    # training_error = str(top160_training_error)
    # validation_error = str(top160_validation_error)
    # print("MSE is " + str(training_error) + ' for 60-word features model on the training set. (closed form) ["extract_keywords","avg_keyword"]' )
    # print("MSE is " + str(validation_error) + ' for 60-word features model on the validation set. (closed form) ["extract_keywords","avg_keyword"]' )

"""
MSE is 1.0458993305987312 for 160-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
MSE is 0.985558231296495 for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
Difference: 0.0603410993022351

MSE is 1.0466443274695258 for 160-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]
MSE is 0.988606687424817 for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]
Difference: 0.0580376400447030

MSE is 1.04682623890494 for 160-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls"]
MSE is 0.9893830598686777 for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls"]
Difference: 0.0574431790362631

MSE is 1.0468268069157576 for 160-word features model on the training set. (closed form) ["count_words","avg_words_len"]
MSE is 0.9893884011669069 for 160-word features model on the validation set. (closed form) ["count_words","avg_words_len"]
Difference: 0.0574384057488441

MSE is 1.0461456589456841 for 160-word features model on the training set. (closed form) ["extract_keywords","avg_keyword"]
MSE is 0.9864798298937568 for 160-word features model on the validation set. (closed form) ["extract_keywords","avg_keyword"]
Difference: 0.0596658290519239

MSE is 1.0467082484792412 for 160-word features model on the training set. (closed form) ["count_words","extract_keywords"]
MSE is 0.989115373691621 for 160-word features model on the validation set. (closed form) ["count_words","extract_keywords"]
Difference: 0.0596658290519239

starting 60
MSE is 1.0593746004140236 for 60-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
MSE is 0.9798744754406972 for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords","avg_keyword"]
Difference: 0.0795001249733231

MSE is 1.0601548886533734 for 60-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]
MSE is 0.9821912344357807 for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls","extract_keywords"]
Difference: 0.0779636542175900

MSE is 1.0602082743190757 for 60-word features model on the training set. (closed form) ["count_words","avg_words_len","find_urls"]
MSE is 0.9826271472126821 for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len","find_urls"]
Difference: 0.0775811271063880

MSE is 1.0602088279189974 for 60-word features model on the training set. (closed form) ["count_words","avg_words_len"]
MSE is 0.9826229317339635 for 60-word features model on the validation set. (closed form) ["count_words","avg_words_len"]
Difference: 0.0775858961850271

MSE is 1.059458282917423 for 60-word features model on the training set. (closed form) ["extract_keywords","avg_keyword"]
MSE is 0.9797631310632505 for 60-word features model on the validation set. (closed form) ["extract_keywords","avg_keyword"]
Difference: 0.0796951518541700

MSE is 1.0601800211863246 for 60-word features model on the training set. (closed form) ["count_words","extract_keywords"]
MSE is 0.9825485864626804 for 60-word features model on the validation set. (closed form) ["count_words","extract_keywords"]
Difference: 0.0776314347236400
"""

if __name__ == '__main__':
    main()
