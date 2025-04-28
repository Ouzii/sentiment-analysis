import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import codecs
import numpy as np
from argparse import ArgumentParser
import argparse
import re

def predict(lang, dims, file_path, column, as_strings):
    file_name = file_path.split('/')[-1].split('.')[0]
    print(file_path)
    cv = CountVectorizer(analyzer='word')   
    results = []
    with open(f'{dims}/traintest/gold-train.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            results.append(l)
    f.close()
    input_file = np.loadtxt(file_path, dtype=str, delimiter=';', skiprows=0, usecols=(column,), encoding='utf8')
    original_input = []
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for l in lines:
            original_input.append(l.replace('\n', ''))
    f.close()
    features = cv.fit_transform(codecs.open(f'{dims}/traintest/{lang}/{lang}-train.txt', 'r', 'utf8'))
    input_values = cv.transform(input_file) 
    results = np.array(results)
    classifier = LogisticRegression(random_state=0, solver="liblinear", max_iter=1000)
    model = classifier.fit(features, results)
    prediction = model.predict(input_values)
    res = []
    for i, item in enumerate(prediction):
        s = f'{item}'
        s = s.replace('\n', '')
        if as_strings: 
            if dims == 'multi':
                s = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'][int(s)-1]
            else:
                s = ['false', 'true'][int(s)]
        if i == 0:
            s = 'prediction'
        res.append(f'{original_input[i]};{s};\n')
    with open(f'../output/result-{file_name}.csv', 'w', encoding='utf8') as file:
        for r in res:
            file.write(r)
    file.close()

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
