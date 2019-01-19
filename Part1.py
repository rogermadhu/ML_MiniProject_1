import numpy as np
import json
import string
import operator

def ReadFile():
    file = open('./data/proj1_data.json', 'r')
    data= file.read()
    file.close()
    Jsondata = json.loads(data)
    return Jsondata

def ToLower(str):
    return str.lower()

def Split(str):
    return str.split(' ')

def PreProcessText(jsonContent, topMostFreqWords):
    if len(jsonContent) > 0:
        textDic = {}
        for x in jsonContent:
            wordsList = Split(ToLower(x['text']))
            for word in wordsList:
                if word not in textDic:
                    textDic[word] = 1
                else:
                    textDic[word]+= 1
        sortedWordsList = sorted(textDic.items(), key=operator.itemgetter(1), reverse=True)
        
        retDic = {}
        for i in range(0, topMostFreqWords):
            retDic[sortedWordsList[i][0]] = sortedWordsList[i][1]
            print(sortedWordsList[i])
    return retDic

def SplitData(jData, start, stop):
    temp = []
    for i in range(start, stop):
        temp.append(jData[i])
    return temp


jData = ReadFile()

jDataTrain = []
jDataTrain = SplitData(jData, 0, 10000)
jDataValidate = []
jDataValidate = SplitData(jData, 10000, 11000)
jDataTest = []
jDataTest = SplitData(jData, 11000, 12000)



PreProcessText(jDataTrain, 160)