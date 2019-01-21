import numpy as np
import json
import string
import operator
from collections import *

class OrderedCounter(Counter, OrderedDict):
    pass

# array is 


"""
 Brainstormed features:
 - sentiment analysis
"""

def add_features(top_words, data):
    # Word count feature
    # 100 data points 
    for data_point in data:
        word_count_feature = np.zeros(160)
        word_list = process_string(data_point['text'])
    
        for word in word_list:
            if word in top_words:
                index = top_words.index(word)
                word_count_feature[index] += 1
        data_point["word_count_feature"] = word_count_feature

    # TODO: at least two more features: these can be 
    # based on the text data, transformations of the other numeric features, or interaction terms. 
    return data

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
    # print(top_words)
    
    data_with_features = add_features(top_words, train)
    print(data_with_features[3])

if __name__ == '__main__':
    main()