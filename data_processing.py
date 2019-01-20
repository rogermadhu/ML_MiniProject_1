import numpy as np
import json
import string
import operator

def read_json_file():
    file = open('./data/proj1_data.json', 'r')
    data= file.read()
    file.close()
    json_data = json.loads(data)
    return json_data

def to_lower(str):
    return str.lower()

def split(str):
    return str.split(' ')

def preprocess_text(jsonContent, topMostFreqWords):
    if len(jsonContent) > 0:
        textDic = {}
        for x in jsonContent:
            wordsList = split(to_lower(x['text']))
            for word in wordsList:
                if word not in textDic:
                    textDic[word] = 1
                else:
                    textDic[word]+= 1
        sortedWordsList = sorted(textDic.items(), key=operator.itemgetter(1), reverse=True)
        
        retDic = {}
        for i in range(0, topMostFreqWords):
            retDic[sortedWordsList[i][0]] = sortedWordsList[i][1]
            # print(sortedWordsList[i])
    return retDic

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

    print("Preprocessing training data")
    preprocess_text(train_data, 160)

if __name__ == '__main__':
    main()