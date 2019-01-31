import json
import csv
from collections import *

class OrderedCounter(Counter, OrderedDict):
    pass

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
    most_common = word_freq.most_common(num_top_words)
    return most_common

def write_words(most_common):
    with open('most_common_words_count.txt', 'w+') as file_count, open('most_common_words.txt', 'w+') as file_no_count:
        for word, count in most_common:
            file_count.write(word + " occurred " + str(count) + " times\n")
            file_no_count.write(word + "\n")

def main():
    data = read_json_file()
    print(len(data))
    most_common = count_top_words(data, 160)
    print(len(most_common))
    write_words(most_common)


if __name__ == '__main__':
    main()

