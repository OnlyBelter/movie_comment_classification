from nltk.stem.porter import PorterStemmer
import os
import re
import unicodedata
import json


root_dir = r'data'
training_file = 'train.txt'
removes = ['', '']
replaces = {'\/': ' ', 'is n\'t': 'isnot'}

def pre_process(root_d, input_f, output_f):
    y = {}
    X = {}
    word_count = {}
    i = 0
    stemmer = PorterStemmer()
    with open(os.path.join(root_d, input_f)) as f_handle:
        for each_line in f_handle:
            # each_line = '2 If you \'re paying attention , the `` big twists '' are pretty easy to guess - but that does n\'t make the movie any less entertaining .'
            each_line = each_line.strip()
            y[i] = each_line[0]
            x = each_line[1:].strip().lower()
            x = re.sub(r'[\']?\d+[st]*', 'number', x)
            x = re.sub(r'\\/', ' ', x)
            x = re.sub(r'n\'t', 'not', x)
            x = re.sub(r'\'re', 'are', x)
            x = re.sub(r'it \'s', 'it is', x)
            x = re.sub(r'that \'s', 'that is', x)
            x = re.sub(r'there \'s', 'there is', x)
            x = re.sub(r'\?', 'question_mark', x)
            x = re.sub(r'!', 'exclamation_mark', x)
            x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore')
            x_list = x.decode('utf-8').split(' ')
            x_list_stemmed = [stemmer.stem(word) for word in x_list]  # stemming
            X[i] = x_list_stemmed
            i += 1
    with open(os.path.join(root_d, 'y.json'), 'a') as f_handle:
        json.dump(y, f_handle, indent=2)
    with open(os.path.join(root_d, 'X.json'), 'a') as f_handle:
        json.dump(X, f_handle, indent=2)

pre_process(root_dir, training_file, 'process_result.txt')

