try:
    from .pre_processing import pre_process
except:
    from pre_processing import pre_process
from sklearn.ensemble import RandomForestClassifier
import os
import json
from sklearn.externals import joblib


def trans2sentence(inx2word_path, x):
    with open(inx2word_path, 'r') as f_handle:
        inx2word = json.load(f_handle)
    inx2word_dic = {i[0]: i[1] for i in inx2word}
    print(x)
    x_inx = []
    if x[0] == 1:
        x_inx.append(0)
    for i in range(len(x)):
        # print(inx2word_dic[i])
        if i*x[i] != 0:
            x_inx.append(i)
    print(x_inx)
    sent = [inx2word_dic[i] for i in x_inx]
    return ' '.join(sent)


# 随机森林模型, 本身就是多类分类器
def random_forest_cla(training_data, model_name):
    X_train = training_data['X']
    y_train = training_data['y']
    forest_clf = RandomForestClassifier(random_state=42)
    forest_clf.fit(X_train, y_train)
    joblib.dump(forest_clf, model_name)


if __name__ == '__main__':
    print('Start to load training data...')
    data_dir_path = r'data'
    training_file = 'train.txt'
    training_data = pre_process(data_dir_path, training_file, min_freq=10, count_in_sent=False)
    print('Training data\'s shape is', training_data['X'].shape)
    print('Start to fit model...')
    saved_model_name = 'saved_model.pkl'
    random_forest_cla(training_data, model_name=saved_model_name)
    print('Model saved in:', os.path.abspath(saved_model_name))