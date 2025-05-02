import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import codecs
import numpy as np
from argparse import ArgumentParser
import argparse
import re
cv = CountVectorizer(analyzer='word')   

def get_data(dims, lang):
    lang_train = cv.fit_transform(codecs.open(f'{dims}/traintest/{lang}/{lang}-train.txt', 'r', 'utf8'))
    lang_test = cv.fit_transform(codecs.open(f'{dims}/traintest/{lang}/{lang}-test.txt', 'r', 'utf8'))

    return lang_train, lang_test

def get_results(dims):
    gold_train = []
    gold_test= []
    with open(f'{dims}/traintest/gold-train.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            gold_train.append(l)
    f.close()

    with open(f'{dims}/traintest/gold-test.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            gold_test.append(l)
    f.close()

    return np.array(gold_train), np.array(gold_test)

def get_inputs(file_path, column):
    input_file = np.loadtxt(file_path, dtype=str, delimiter=';', skiprows=0, usecols=(column,), encoding='utf8')
    original_input = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for l in lines:
            original_input.append(l.replace('\n', ''))
    f.close()

    return cv.transform(input_file), original_input

def write_result(prediction, as_strings, dims, original_input, file_name):
    multi_strings = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    bin_strings = ['false', 'true']
    header = 'prediction'

    def format_item(item):
        s = f'{item}'.replace('\n', '')
        if as_strings: 
            if dims == 'multi':
                return multi_strings[int(s)-1]
            else:
                return bin_strings[int(s)]
        if i == 0:
            return header
        return s

    res = []
    for i, item in enumerate(prediction):
        res.append(f'{original_input[i]};{format_item(item)};\n')

    with open(f'../output/result-{file_name}.csv', 'w', encoding='utf8') as file:
        for r in res:
            file.write(r)
    file.close()
    print(f'Result written in output/result-{file_name}.csv')

def predict(lang, dims, file_path, column, as_strings):
    file_name = file_path.split('/')[-1].split('.')[0]
    lang_train, lang_test = get_data(dims, lang)
    gold_train, gold_test = get_results(dims)
    input_file, original_input = get_inputs(file_path, column)

    classifier = LogisticRegression(random_state=0, solver="liblinear", max_iter=1000)
    model = classifier.fit(lang_train, gold_train)
    model = model.fit(lang_test, gold_test)
    prediction = model.predict(input_file)
    write_result(prediction, as_strings, dims, original_input, file_name)

def main():
    parser = ArgumentParser()
    # Code borrowed and modified from https://github.com/cynarr/sentimentator/blob/master/data_import.py
    def check_lang(l, pattern=re.compile(r'^[a-zA-Z]{2}$')):
        if not pattern.match(l):
            raise argparse.ArgumentTypeError('Use a lowercase two-character alphabetic language code. Available codes: en, fi, fr, it.')
        return l

    def check_dim(d):
        dims = ['multi', 'bin']
        if d not in dims:
            raise argparse.ArgumentTypeError('Use "multi" or "bin" for the classification type.')
        return d
    
    parser.add_argument('LANG', help='', type=check_lang)
    parser.add_argument('DIMS', help='', type=check_dim)
    parser.add_argument('FILE', help='path to input file (csv)', type=str)
    parser.add_argument('COL', help='index of column containing data. Starts from 0', type=int)
    parser.add_argument('AS_STRINGS', help='Boolean value to indicate should result be shown as string', type=bool)


    args = parser.parse_args()

    predict(args.LANG, args.DIMS, args.FILE, args.COL, args.AS_STRINGS)


if __name__ == "__main__":
    main()
