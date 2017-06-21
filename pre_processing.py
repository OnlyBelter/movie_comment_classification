from nltk.stem.snowball import EnglishStemmer
import os
import re
import unicodedata
import json
import numpy as np

root_dir = r'data'
training_file = 'train.txt'

def pre_process(root_d, input_f, output_f):
    y = {}
    X = {}
    word_count = {}
    word2inx = {}
    inx2word = []
    i = 0
    stemmer = EnglishStemmer()
    with open(os.path.join(root_d, input_f)) as f_handle:
        for each_line in f_handle:
            # each_line = '2 If you \'re paying attention , the `` big twists '' are pretty easy to guess - but that does n\'t make the movie any less entertaining .'
            each_line = each_line.strip()
            y[i] = each_line[0]
            x = each_line[1:].strip().lower()
            x = re.sub(r'[\']?\d+[st]*', 'number', x)
            x = re.sub(r'\\/', ' ', x)
            x = re.sub(r'ca n\'t', 'can not', x)
            x = re.sub(r'n\'t', 'not', x)
            x = re.sub(r'\'re', 'are', x)
            x = re.sub(r'\'m', 'am', x)
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
    # with open(os.path.join(root_d, 'y.json'), 'a') as f_handle:
    #     json.dump(y, f_handle, indent=2)
    # with open(os.path.join(root_d, 'X.json'), 'a') as f_handle:
    #     json.dump(X, f_handle, indent=2)
    # count
    for m in X:
        each_sentence = X[m]
        for word in each_sentence:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    # with open(os.path.join(root_d, 'word_count.json'), 'a') as f_handle:
    #     json.dump(word_count, f_handle, indent=2)
    inx = 0
    for m in word_count:
        if (not re.search(r'^\W+$', m)) and (word_count[m] >= 10) and (len(m) > 1):
            word2inx[m] = inx  # these words will become word vector
            inx2word.append((inx, m))
            inx += 1
    # with open(os.path.join(root_d, 'inx2word.json'), 'a') as f_handle:
    #     json.dump(inx2word, f_handle, indent=2)
    # sentence to index and convert y to np.array
    X_inx = {}
    X_new = {}
    y_array = np.zeros([len(y), 1], dtype=np.int)
    for i, m in enumerate(sorted(list(X.keys()))):
        each_sentence = X[m]
        X_inx[m] = []
        X_new[m] = []
        y_array[i] = y[i]
        for word in each_sentence:
            if word in word2inx:
                X_new[m].append(word)
                X_inx[m].append(word2inx[word])
        # print(' '.join(X_new[m]))
    # convert sentence to vector
    X_matrix = np.zeros([len(X_inx), len(word2inx)])
    total_sentence_inx = sorted(list(X_inx.keys()))
    # print(total_sentence_inx)
    for i in total_sentence_inx:
        X_matrix[i][X_inx[i]] = 1
    # delete the sentences that have no word
    sen_inx = np.where(X_matrix.sum(1)!=0)
    X_matrix = X_matrix[sen_inx]
    y_array = y_array[sen_inx]
    # np.save('X', X_matrix)
    # np.save('y', y_array)
    return {'X': X_matrix, 'y': y_array}

# root_d = root_dir
# input_f = training_file
# output_f = 'process_result2.txt'

processed_data = pre_process(root_dir, training_file, 'process_result2.txt')
print(processed_data['y'])
