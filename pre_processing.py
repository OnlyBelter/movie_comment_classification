from nltk.stem.porter import PorterStemmer
import os
import re


root_dir = r'data'
training_file = 'train.txt'
removes = ['', '']
replaces = {'\/': ' '}

def pre_process(root_d, input_f, output_f):
    y = {}
    X = {}
    word_count = {}
    i = 0
    stemmer = PorterStemmer()
    with open(os.path.join(root_d, input_f)) as f_handle:
        for each_line in f_handle:
            each_line = each_line.strip()
            y[i] = each_line[0]
            x = each_line[1:].strip().lower()
            x = re.sub(r'\d+', 'number', x)
            x_list = x.split(' ')
            x_list_stemmed = [stemmer.stem(word) for word in x_list]  # stemming



