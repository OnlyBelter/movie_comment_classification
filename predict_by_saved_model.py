from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix  # 用于评价模型
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
try:
    from .pre_processing import pre_process_dev_test
except:
    from pre_processing import pre_process_dev_test
import os


def validate_model(X_dev, y_dev):
    forest_clf = joblib.load('saved_model.pkl')
    y_test_pred = []
    for i in X_dev:
        y_test_pred += list(forest_clf.predict([i]))
    # print('y_pred:', y_test_pred)
    print('The precision of classification in dev is:')
    print(np.where(y_dev == y_test_pred)[0].size / y_dev.size)
    print('#------------')

    # 模型的评价
    print('Cross validation scores in dev:')
    print(cross_val_score(forest_clf, X_dev, y_dev, cv=3, scoring='accuracy'))
    print('#------------')
    y_pred = cross_val_predict(forest_clf, X_dev, y_dev, cv=3)
    conf_mx = confusion_matrix(y_dev, y_pred)
    print('Confusion matrix:')
    print(conf_mx)
    print('#------------')
    # plt.matshow(conf_mx)
    # plt.show()
    #--- plot normalized confusion matrix
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    nor_conma_file = 'normalized_confusion_matrix.png'
    if not os.path.exists(nor_conma_file):
        plt.savefig(nor_conma_file)
    print('Normalized confusion matrix image saved in', os.path.abspath(nor_conma_file))


def classify_test(X_test, test_data_path, data_root_dir):
    forest_clf = joblib.load('saved_model.pkl')
    y_test_pred = []
    for i in X_test:
        # print(forest_clf.predict([i]))
        y_test_pred += list(forest_clf.predict([i]))
        # print(forest_clf.predict_proba([i]))
    print('y_pred:', len(y_test_pred))
    all_test_data = []
    with open(os.path.join(data_root_dir, test_data_path), 'r') as f_handle:
        all_test_data = f_handle.readlines()
    result_file = 'test_result.txt'
    if not os.path.exists(result_file):
        for i in range(len(y_test_pred)):
            with open(result_file, 'a') as f_handle:
                f_handle.write(str(y_test_pred[i]) + ' ' + all_test_data[i].strip() + '\n')


if __name__ == '__main__':
    data_dir_path = r'data'
    dev_file = 'dev.txt'
    test_file = 'test-release.txt'
    word2inx_file = 'word2inx.json'
    word2inx_file_path = os.path.join(data_dir_path, word2inx_file)
    validate = True
    if validate:
        print('Start to load dev data...\n')
        dev_data = pre_process_dev_test(data_dir_path, dev_file,
                                        word2inx_path=word2inx_file_path,
                                        min_freq=5,
                                        count_in_sent=False,
                                        data_type='dev')
        validate_model(X_dev=dev_data['X'], y_dev=dev_data['y'])
    print('Start to load test data...\n')
    test_data_mat = pre_process_dev_test(data_dir_path, test_file,
                                         word2inx_path=word2inx_file_path,
                                         min_freq=5,
                                         count_in_sent=False,
                                         data_type='test')
    print('There are {} totally'.format(test_data_mat['X'].shape[0]))
    classify_test(test_data_mat['X'], test_data_path=test_file, data_root_dir=data_dir_path)
