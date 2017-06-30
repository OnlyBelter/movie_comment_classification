# movie_comment_classification
This is for my web mining class final project this term(2017 Spring)

- 该模型用于对电影评论进行分类（0-4共5类），训练数据位于`data/train.txt`
- 使用随机森林模型




# Requirement
Python 3.5


# Dependencies
- nltk（用于提取词干）
- scikit-learn（建模及评估）
- unicodedata（用于过滤non-ascii编码）
- numpy


# Usage
- training: `python train.py`
  - 这一步使用train.txt训练并保存模型
- validate and predict: `python predict_by_saved_model.py`
  - 这一步使用dev.txt验证模型的准确率，并预测test-release.txt

# Result
- The precision of classification in dev:  0.3518
- 结果文件为`test_result.txt`
- Normalized confusion matrix heatmap
<div align="center">
    <img src="https://github.com/OnlyBelter/learn_neuralTalk/blob/master/demo_images/002_ski.png?raw=true">
</div>