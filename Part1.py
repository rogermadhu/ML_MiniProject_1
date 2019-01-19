import numpy as np
import json
import string

def readFile():
    file = open('./data/proj1_data.json', 'r')
    data= file.read()
    file.close()
    Jsondata = json.loads(data)
    return Jsondata

    totalData = len(Jsondata)
    print(totalData)
    # for x in Jsondata:
    #     print(x, x['text'])

def toLower(str):
    return str.lower()

def split(str):
    return str.split(' ')

readFile()