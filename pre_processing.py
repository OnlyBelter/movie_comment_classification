from nltk.stem.snowball import EnglishStemmer
import os
import re
import unicodedata
import json
import numpy as np

def count_num_in_sent(sent_inx_list):
    """

    :param sent_inx_list:  [1685, 1477, 19, 775, 1391, 223, 686]
    :return: index and number with ordered, {'inxs': [], 'num': []}
    """
    sent_inx_list = sorted(sent_inx_list)
    inx2count = {}
    inxs = []
    num = []
    for inx in sent_inx_list:
        if inx in inx2count:
            inx2count[inx] += 1
        else:
            inx2count[inx] = 1
    for inx in sorted(inx2count.keys()):
        inxs.append(inx)
        num.append(inx2count[inx])
    return {'inxs': inxs, 'num': num}


def pre_process(root_d, input_f, min_freq, count_in_sent=False):
    """
    :param root_d:
    :param input_f:
    :param min_freq: minimum frequency, an integer
    :param count_in_sent: if need to count words in sentence
    :return:
    """
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
            # X[i] = x_list
            i += 1
    # y_json_path = os.path.join(root_d, 'y_train.json')
    # X_json_path = os.path.join(root_d, 'X_train.json')
    # if not os.path.exists(y_json_path):
    #     with open(y_json_path, 'a') as f_handle:
    #         json.dump(y, f_handle, indent=2)
    # if not os.path.exists(X_json_path):
    #     with open(X_json_path, 'a') as f_handle:
    #         json.dump(X, f_handle, indent=2)
    # count
    for m in X:
        each_sentence = X[m]
        for word in each_sentence:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    # word_count_path = os.path.join(root_d, 'word_count_in_train.json')
    # if not os.path.exists(word_count_path):
    #     with open(word_count_path, 'a') as f_handle:
    #         json.dump(word_count, f_handle, indent=2)
    inx = 0
    for m in sorted(word_count.keys()):
        if (not re.search(r'^\W+$', m)) and (word_count[m] >= min_freq) and (len(m) > 1):
            word2inx[m] = inx  # these words will become word vector
            inx2word.append((inx, m))
            inx += 1
    # inx2word_path = os.path.join(root_d, 'inx2word.json')
    # if not os.path.exists(inx2word_path):
    #     with open(inx2word_path, 'a') as f_handle:
    #         json.dump(inx2word, f_handle, indent=2)
    # used in test
    word2inx_path = os.path.join(root_d, 'word2inx.json')
    if not os.path.exists(word2inx_path):
        with open(word2inx_path, 'a') as f_handle:
            json.dump(word2inx, f_handle, indent=2)

    # sentence to index and convert y to np.array
    X_inx = {}
    X_new = {}
    y_array = np.zeros([len(y)], dtype=np.int)
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
    X_matrix = np.zeros([len(X_inx), len(word2inx)], dtype=np.int)
    total_sentence_inx = sorted(list(X_inx.keys()))
    # print(total_sentence_inx)
    inx_num = {}
    for i in total_sentence_inx:
        if count_in_sent:
            inx_num = count_num_in_sent(X_inx[i])
            X_matrix[i][inx_num['inxs']] = inx_num['num']
        else:
            X_matrix[i][X_inx[i]] = 1
        # if i == 1:
        #     print(X_inx[i])
            # print(inx_num)
    # delete the sentences that have no word
    sen_inx = np.where(X_matrix.sum(1)!=0)
    X_matrix = X_matrix[sen_inx]
    y_array = y_array[sen_inx]
    # np.save('X', X_matrix)
    # np.save('y', y_array)
    return {'X': X_matrix, 'y': y_array}


def pre_process_dev_test(root_d, input_f, min_freq, word2inx_path, count_in_sent=False, data_type='dev'):
    """

    :param root_d:
    :param input_f:
    :param min_freq: minimum frequency, an integer
    :param count_in_sent: if need to count words in sentence
    :return:
    """
    y = {}
    X = {}
    i = 0
    stemmer = EnglishStemmer()
    with open(os.path.join(root_d, input_f)) as f_handle:
        for each_line in f_handle:
            # each_line = '2 If you \'re paying attention , the `` big twists '' are pretty easy to guess - but that does n\'t make the movie any less entertaining .'
            each_line = each_line.strip()
            if data_type == 'dev':
                y[i] = each_line[0]
                x = each_line[1:].strip().lower()
            else:
                y[i] = 0
                x = each_line[:].strip().lower()
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
    print(i)
    if data_type == 'dev':
        y_json_path = os.path.join(root_d, 'y_dev.json')
        X_json_path = os.path.join(root_d, 'X_dev.json')
        if not os.path.exists(y_json_path):
            with open(y_json_path, 'a') as f_handle:
                json.dump(y, f_handle, indent=2)
        if not os.path.exists(X_json_path):
            with open(X_json_path, 'a') as f_handle:
                json.dump(X, f_handle, indent=2)

    # sentence to index and convert y to np.array
    X_inx = {}
    X_new = {}
    y_array = np.zeros([len(y)], dtype=np.int)
    if os.path.exists(word2inx_path):
        with open(word2inx_path, 'r') as f_handle:
            word2inx = json.load(f_handle)
    else:
        print("There is no 'word2inx.json' file, please run train.py first")
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
    X_matrix = np.zeros([len(X_inx), len(word2inx)], dtype=np.int)
    total_sentence_inx = sorted(list(X_inx.keys()))
    # print(total_sentence_inx)
    for i in total_sentence_inx:
        if count_in_sent:
            inx_num = count_num_in_sent(X_inx[i])
            X_matrix[i][inx_num['inxs']] = inx_num['num']
        else:
            X_matrix[i][X_inx[i]] = 1
        # if i == 1:
        #     print(X_inx[i])
            # print(inx_num)
    # delete the sentences that have no word
    if data_type == 'test':
        return {'X': X_matrix}
    sen_inx = np.where(X_matrix.sum(1) != 0)
    X_matrix = X_matrix[sen_inx]
    y_array = y_array[sen_inx]
    if data_type == 'dev':
        return {'X': X_matrix, 'y': y_array}


# root_d = root_dir
# input_f = training_file
# output_f = 'process_result2.txt'


