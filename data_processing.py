import numpy as np
import json
import string
import operator
from collections import Counter

def read_json_file():
    file = open('./data/proj1_data.json', 'r')
    data= file.read()
    file.close()
    json_data = json.loads(data)
    return json_data

def process_string(str):
    return str.split(' ').lower()

def count_top_words(train_data):
    word_freq = Counter()
    for line in train_data:
        word_list = process_string(line['text'])
        word_freq.update(word_list)
    return dict(word_freq.most_common(160))

def split_data(jData, start, stop):
    temp = []
    for i in range(start, stop):
        temp.append(jData[i])
    return temp

def main():
    data = read_json_file()

    """
    Use the first 10,000 points for training, the next 1,000 for validation, and the final 1,000 for testing.
    """
    train_data = split_data(data, 0, 10000)
    print("Created training dataset. Sample from training data: {}").format(train_data[0])

    validate_data = split_data(data, 10000, 11000)
    print("Created validation dataset. Sample from validation data: {}").format(validate_data[0])

    test_data = split_data(data, 11000, 12000)
    print("Created testing dataset. Sample from validation data: {}").format(test_data[0])

    print("Finding top words")
    word_frequency = count_top_words(train_data)
    # print(word_frequency)

if __name__ == '__main__':
    main()