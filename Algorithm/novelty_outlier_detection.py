# coding:utf-8
import numpy as np
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import argparse
import csv
import sys
import os
import urllib
import http.client
import json
import math
import time
import threading


parser = argparse.ArgumentParser(description="Novelty and Outlier Detection")
parser.add_argument('--train_data', default='/root/data/mixcontent/zhangqi/one_class/dataset/segment_train_data.txt', help='训练数据文件')
parser.add_argument('--test_data', default='/root/data/mixcontent/zhangqi/one_class/dataset/test_true_data.txt', help='测试数据文件')
parser.add_argument('--segment_corpus_data', default=None, help='需要去除异常数据的语料（已分词）')


# stopwords_file = '/root/data/mixcontent/zhangqi/tensorflowapp/opinionMining/cluster/source/stop_words.txt'
stopwords_file = '/root/data/mixcontent/zhangqi/one_class/stop_words.txt'


def load_stopwords(stopwords_file):
  """加载停用词"""
  # return map(lambda word:word.strip(), open(stopwords_file).readlines())
  return [word.strip() for word in open(stopwords_file) if word]

# tfidf  Vecorizer
tfidfVecorizer = TfidfVectorizer(stop_words=load_stopwords(stopwords_file))



def sklearn_evaluation(y_true, y_pred, labels=None):
  """获取评估数据: precision, recall, f1-score, confusion matrix."""

  classification_report = metrics.classification_report(y_true, y_pred, labels=labels)
  return float(metrics.accuracy_score(y_true, y_pred)), \
       classification_report, metrics.confusion_matrix(y_true, y_pred, labels=labels)


def load_train_data(data_path, segment=True):
    """"""
    # with open(data_path) as f:
    # for line in f:
    #   #分词处理 
    #   segment_doc = ' '.join(segment_api(line.strip()))
    #   # sentence_list.append(segment_sentence)
    #   corpora_documents.append(segment_doc)
    if segment:
      return [' '.join(segment_api(line.strip())) for line in open(data_path) if line.strip()]
    else:
      return [line.strip() for line in open(data_path) if line]

def load_json_corpus(json_file):
  """"""
  count_error = 0
  with open(json_file) as f:
    # json_list = map(lambda x:json.loads(x), json_file)
    json_list = [json.loads(line) for line in f]
    print('data count:', len(json_list))
    # corpus = [' '.join(segment_api(data["content"].strip())) for data in json_list if data["content"].strip()]
  corpus = []
    
  for ix, data in enumerate(json_list):
    try:
      if data["content"].strip():
        corpus.append(' '.join(segment_api(data["content"].strip())))
        time.sleep(0.06)
        count+=1
    except Exception as e:
      print('sentence length :{}, error {}'.format(len(data["content"])), e)
      count_error+=1
    if ix % 1000 == 0:
      print('segment process %s' % ix)
      sys.stdout.flush()
  with open('segment_corpus.txt', 'w') as f:
    f.writelines([doc for doc in corpus])

  print('success count: {} error count: {} all count: {}'.format(len(corpus),count_error),len(json_list))
  sys.stdout.flush()
  return corpus

def load_test_data(data_path, segment=True):
  """"""
  test_data = []
  test_label = []
  if segment:
    for line in open(data_path):
      if line.strip():
        label, doc = line.strip().split(',', 1)
        doc = ' '.join(segment_api(line.strip()))
        test_data.append(doc)
        test_label.append(label)

    return test_data, test_label
  else:
    for line in open(data_path):
      if line.strip():
        label, doc = line.strip().split(',', 1)
        test_data.append(doc)
        test_label.append(label)
    return test_data, test_label
    

def Tfidf_transform(corpusa_doccuments):
  """"""
  #若要过滤停用词，可在初始化模型时设置
  # tfidfVecorizer
  return tfidfVecorizer.fit_transform(corpusa_doccuments)

def TfidfTransform(corpusa_doccuments):
  """转换tfidf语料"""
  return tfidfVecorizer.transform(corpusa_doccuments)
  
def resultsToCsv(data_path, novelty_results, outlier_results):
  """模型预测正确与离群结果写入csv"""
  try:
    with open(os.path.join(data_path,"novelty_results.csv"),"w") as f_nov:
      writer = csv.writer(f_nov)
      writer.writerows(novelty_results)
    with open(os.path.join(data_path,"outlier_results.csv"),"w") as f_out:
      writer = csv.writer(f_out)
      writer.writerows(outlier_results)
  except Exception as e:
    raise e


# fit the model
class OneClassSVMClassifier:
  """docstring  OneClassSVMClassifier"""
  def __init__(self, save_path):
    """arg"""
    # 训练误差，在（0,1]之间
    self.nu = 0.4005
    # 核函数
    self.kernel = 'rbf'
    # 核函数参数
    # self.gamma = 0.1
    # 保存路径
    self.save_path = os.path.join(save_path,'OneClassSVM')
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    # 分类器初始化
    self.classifier = svm.OneClassSVM(nu=self.nu, kernel=self.kernel)

  # fit one class model 
  def fit_model(self, train_data_matrix):
    """模型训练"""
    novelty_results = []
    outlier_results = []
    self.classifier.fit(train_data_matrix)
    # scores_pred = self.classifier.decision_function(train_data_matrix)
    y_pred_label = self.classifier.predict(train_data_matrix)
    n_outlier = y_pred_label[y_pred_label == -1].size
    print('train outlier  {}/{} = {}'.format(n_outlier,len(y_pred_label),1-(n_outlier/len(y_pred_label))))

  def predictResults(self, corpus_matrix, documents):
    """模型预测"""
    novelty_results = []
    outlier_results = []
    y_pred_label = self.classifier.predict(corpus_matrix)
    n_outlier = y_pred_label[y_pred_label == -1].size
    print('train outlier  {}/{} = {}'.format(n_outlier,len(y_pred_label),1-(n_outlier/len(y_pred_label))))
    # split novelty and outlier
    novelty_indexs = np.where(pred_label == 1)[0]
    outlier_indexs = np.where(pred_label == -1)[0]
    for ix, doc in enumerate(documents):
      if ix in novelty_indexs:
        novelty_results.append([ix,doc])
      elif ix in outlier_indexs:
        outlier_results.append([ix,doc])
      if ix % 1000==0:
        print("process sentence %d"% ix)
        sys.stdout.flush()
    return novelty_results, outlier_results  

  def test_model(self, test_data_matrix, test_true_label):
    """"""
    y_pred_label = self.classifier.predict(test_data_matrix)
    accuracy, classification_report, confusion_matrix = sklearn_evaluation(test_true_label, y_pred_label)
    print('Accuracy: {} \nClassification Report:\n{} \nconfusion_matrix:\n{}'.format(accuracy, classification_report,confusion_matrix))


  def predictToCSV(self, corpus_matrix, documents):
    # results write in csv files
    novelty_results, outlier_results = self.predictResults(corpus_matrix, documents)
    resultsToCsv(self.save_path,novelty_results,outlier_results)

  
  def optimize_parameters(self, train_data_matrix, test_data_matrix, test_true_label, n=10):
    """"根据测试数据参数优化"""
    nu = np.linspace(0.05,0.2,20)
    # nu = [0.005,0.05,0.1,0.3,0.4005,0.45]
    # nu = np.linspace(start=1e-2, stop=1e-4, num=n)
    gamma = np.linspace(start=1e-4, stop=0.8, num=n)
    kernels = [ 'linear', 'poly', 'rbf']
    # kernels = [ 'rbf']
    opt_diff = 1.0
    opt_nu = None
    opt_gamma = None
    fw = open(os.path.join(self.save_path, "model_optimize_parameters.txt"), "a")
    # 
    print('optimize_parameters')
    for kernel in kernels:
      for i in range(len(nu)):
          self.classifier = svm.OneClassSVM(kernel=kernel, nu=nu[i])
          self.classifier.fit(train_data_matrix)
          y_pred_label = self.classifier.predict(test_data_matrix)
          n_errors_test = (y_pred_label!=test_true_label).sum()
          accuracy, classification_report, confusion_matrix = sklearn_evaluation(test_true_label, y_pred_label)
          if accuracy>0.93:
            fw.write('kernel :{} nu:{}\n'.format(kernel,nu[i]))
            print('kernel :{} nu:{}'.format(kernel,nu[i]))
            fw.write('Accuracy: {} \nClassification Report:\n{} \nconfusion_matrix:\n{}'.format(accuracy, classification_report,confusion_matrix))
            print('Accuracy: {} \nClassification Report:\n{}\n'.format(accuracy, classification_report))
            diff = n_errors_test/len(y_pred_label)
          sys.stdout.flush()
      # K 折交叉验证
      # scores = cross_validation.cross_val_score(self.classifier,test_data_matrix,test_true_label, scoring='roc_auc', cv=cv) 
      # print('train cv = 5, mean scores {}'.format(scores.mean()))
      # fw.close()


    # 参数优化训练最优模型
    self.nu = opt_nu
    # self.gamma = opt_gamma
    # restart fit model
    # self.fit_model_write_results(train_data_matrix, train_documents, cv)

class IsolationForest_Classifier:
  """docstring for IsolationForest_Class"""
  def __init__(self, save_path):
    """"""
    # The number of base estimators
    self.n_estimators = 100
    # 保存路径
    self.save_path = os.path.join(save_path,'IsolationForest')
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    # 训练误差
    self.contamination = 0.1
    # 分类器
    self.classifier = IsolationForest(n_estimators=self.n_estimators,contamination=self.contamination)

  def fit_model(self, train_data_matrix, train_documents, cv=5):
    """训练模型"""
    novelty_results = []
    outlier_results = []
    self.classifier.fit(train_data_matrix)
    scores_pred = self.classifier.decision_function(train_data_matrix)
    y_pred_label = self.classifier.predict(train_data_matrix)
    n_outlier = y_pred_label[y_pred_label == -1].size
    print('train outlier  {}/{}'.format(n_outlier,len(y_pred_label)))

    # split novelty and outlier
    novelty_indexs = np.where(y_pred_label == 1)[0]
    outlier_indexs = np.where(y_pred_label == -1)[0]
    for ix, doc in enumerate(train_documents):
      if ix in novelty_indexs:
        novelty_results.append([ix,doc])
      elif ix in outlier_indexs:
        outlier_results.append([ix,doc])
      if ix % 1000==0:
        print("process sentence %d"% ix)

  def predictResults(self,docs_matrix):
    """模型预测"""
    novelty_results = []
    outlier_results = []
    pre_label = self.classifier.predict(docs_matrix)
    novelty_indexs = np.where(y_pred_label == 1)[0]
    outlier_indexs = np.where(y_pred_label == -1)[0]


  def test_model(self, train_data_matrix, test_data_matrix, test_true_label):
    self.classifier.fit(train_data_matrix)
    y_pred_label = self.classifier.predict(test_data_matrix)
    n_outlier = (y_pred_label!=test_true_label).sum()
    diff = n_outlier/len(y_pred_label)
    print('train outlier  {}/{} ,accuracy:{}'.format(n_outlier,len(y_pred_label),1-diff))

  def fit_model_write_results(self, train_data_matrix, train_documents, cv=5):
    _, novelty_results, outlier_results = self.fit_model(train_data_matrix, train_documents, cv)
    # results write in csv files
    print('the results save in {}'.format(self.save_path))
    resultsToCsv(self.save_path,novelty_results,outlier_results)

    
  def optimize_parameters(self, train_data_matrix, test_data_matrix, test_true_label, train_documents, cv=5, n=10):
    """"根据测试数据参数优化"""
    n_estimators = np.linspace(start=50, stop=500, num=n)
    contamination = np.linspace(start=1e-3, stop=0.8, num=n)
    opt_diff = 1.0
    opt_n_estimators  = None
    opt_contamination = None
    fw = open(os.path.join(self.save_path, "model_optimize_parameters.txt"), "a")
    for i in range(len(n_estimators)):
      for j in range(len(contamination)):
        classifier = IsolationForest(n_estimators=int(n_estimators[i]), contamination=contamination[j], n_jobs=-1)
        classifier.fit(train_data_matrix)
        y_pred_label = classifier.predict(test_data_matrix)
        accuracy, classification_report, confusion_matrix = sklearn_evaluation(test_true_label, y_pred_label)
        print('n_estimators: {} contamination: {}'.format(n_estimators[i],contamination[j]))
        print('Accuracy: {} \nClassification Report:{}'.format(accuracy, classification_report))
        fw.write('n_estimators: {} contamination: {}'.format(n_estimators,contamination))
        fw.write('Accuracy: {} \nClassification Report:\n{} \nconfusion_matrix:\n{}'.format(accuracy, classification_report,confusion_matrix))
        n_errors_test = (y_pred_label!=test_true_label).sum()
        diff = n_errors_test/len(y_pred_label)
        sys.stdout.flush()
        # fw.write(",".join([str(nu[i]), str(gamma[j]), str(diff), str(n_errors_test)]) + "\n")
    # 参数优化训练最优模型
    self.n_estimators = opt_n_estimators
    self.contamination = opt_contamination
    print('best accuracy {}, n_estimators: {}, contamination: {}'.format(1-opt_diff,self.n_estimators,self.contamination))
    # restart fit model
    # self.fit_model_write_results(train_data_matrix, train_documents, cv)
  
  def optimize_GridSearchCV(self, train_data_matrix, test_data_matrix, test_true_label, n=20):
    """"""
    self.classifier.fit(train_data_matrix)
    param = {'kernel':('linear', 'rbf'), 'nu':np.linspace(start=1e-5, stop=0.1, num=n), 'gamma':np.linspace(start=1e-5, stop=0.1, num=n)}
    param = {'n_estimators':[10,50,100,150,200,250,300],'contamination':np.linspace(start=1e-5, stop=0.1, num=n),'max_features':[1,5,10,15,20]}
    classifier = IsolationForest(n_jobs=-1)
    clf = GridSearchCV(self.classifier, param)
    clf.fit(test_data_matrix,test_true_label)
    print(best_estimator_)



class LocalOutlierFactor_Classifier:
  """docstring for LocalOutlierFactor_Classifier"""
  def __init__(self, save_path):

    # 默认路径
    self.save_path = os.path.join(save_path,'LocalOutlierFactor')
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    self.n_neighbors=40
    # 数据集中的异常比例。当拟合时, 用于定义决策函数的阈值
    self.contamination = 0.1

    self.classifier = LocalOutlierFactor(n_neighbors=self.n_neighbors,contamination=self.contamination)

 
  def fit_model(self, train_data_matrix, test_data_matrix, test_true_label):
    """训练模型"""
    self.classifier.fit(train_data_matrix)
    y_pred_label = self.classifier.predict(test_data_matrix)
    n_errors_test = (y_pred_label!=test_true_label).sum()
    accuracy, classification_report, confusion_matrix = sklearn_evaluation(test_true_label, y_pred_label)
    print('Accuracy: {} \nClassification Report:\n{}\n'.format(accuracy, classification_report))
    sys.stdout.flush()

  def test_model(test_data,test_label):
    """测试模型
       such as test_label = [1,1,-1,....]
    """

    scores_pred = self.classifier.decision_function(train_data)
    y_pred_test = self.classifier.predict(test_data)

    n_errors = (y_pred_test!=test_label)

class EllipticEnvelope_Classifier:
  """docstring for EllipticEnvelope"""
  def __init__(self, save_path):

    # 默认路径
    # 保存路径
    self.save_path = os.path.join(save_path,'EllipticEnvelope')
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)
    self.contamination = 0.1

    self.classifier = EllipticEnvelope(contamination=self.contamination)
    

  def fit_model(self, train_data_matrix, test_data_matrix, test_true_label):
    """训练模型"""
    train_data_matrix = train_data_matrix.toarray()
    test_data_matrix = test_data_matrix.toarray()
    self.classifier.fit(train_data_matrix)
    y_pred_label = self.classifier.predict(test_data_matrix)
    n_errors_test = (y_pred_label!=test_true_label).sum()
    accuracy, classification_report, confusion_matrix = sklearn_evaluation(test_true_label, y_pred_label)
    print('Accuracy: {} \nClassification Report:\n{}\n'.format(accuracy, classification_report))
    sys.stdout.flush()


  def test_model(test_data,):
    """测试模型
       such as test_label = [1,1,-1,....]
    """
    scores_pred = self.classifier.decision_function(train_data)
    y_pred = self.classifier.predict(train_data)
    n_error_train = y_pred_test[y_pred_test == -1].size


def load_test(path):
  """"""
  text_label = []

  for file in os.listdir(path):
    if 'f_test' in file:
      with open(os.path.join(path,file)) as f_test:
        false_text_corpus = [line.strip() for line in f_test]
    elif 't_test' in file:
      with open(os.path.join(path,file)) as t_test:
        true_text_corpus = [line.strip() for line in t_test]
  text_corpus = false_text_corpus + true_text_corpus
  text_label.extend([-1]*len(false_text_corpus))
  text_label.extend([1]*len(true_text_corpus))
  print('false_text_corpus  count {}, true_text_corpus count {}'.format(len(false_text_corpus),len(true_text_corpus)))

  return text_corpus, text_label




def main(train_data, test_data, n=10):
  """"""
  save_path = '/'.join(train_data.split('/')[:-2])
  text_path = '/'.join(train_data.split('/')[:-1])
    
  # load train data and test data
  train_documents_orgin = load_train_data(train_data, segment=False)
  test_documents, test_label = load_test_data(test_data,segment=False)

  # reload data
  train_documents = train_documents_orgin[:-100]
  test_documents = test_documents + train_documents_orgin[-100:]
  test_label.extend(['1']*100)

  print('tfidf**************')
  # print("train data count {}, test data count {}".format(len(train_documents),len(test_documents)))

  # # merge corpus
  corpusa_doccuments = train_documents + test_documents
  test_label = np.array(test_label).astype(np.int64)

  # # transform corpue to tdidf vector
  corpus_matrix = Tfidf_transform(corpusa_doccuments)
  train_data_count = len(train_documents)

  train_data_matrix, test_data_matrix = corpus_matrix[:train_data_count], corpus_matrix[train_data_count:]

  # 测试
  texts_corpus ,texts_label =  load_test(text_path)
  texts_corpus_matrix = TfidfTransform(texts_corpus)
  texts_label = np.array(texts_label).astype(np.int64)
  print("texts data count {}, test labels shape {}".format(len(texts_corpus),texts_label.shape))
  # print('OneClassSVM model')
  # train and test model
  # OneClassSVM = OneClassSVMClassifier(save_path)
  # # # optimize parameters for test true data 
  # # # # OneClassSVM.test_model(train_data_matrix, test_data_matrix, test_label, train_documents)
  # # OneClassSVM.optimize_parameters(train_data_matrix, test_data_matrix, test_label, train_documents, cv)
  # # OneClassSVM.classifier.fit(train_data_matrix)
  OneClassSVM.fit_model(train_data_matrix, train_documents)

  # OneClassSVM.test_model(texts_corpus_matrix, texts_label)
  OneClassSVM.test_model(test_data_matrix, test_label)
  # OneClassSVM.optimize_parameters(train_data_matrix, texts_corpus_matrix, texts_label)

  # 多线程
  workers = [threading.Thread(target=OneClassSVM.optimize_parameters(train_data_matrix, texts_corpus_matrix, texts_label)) for _ in range(self.workers)]
  for thread in workers:
      thread.daemon = True  # make interrupting the process with ctrl+c easier
      thread.start()
  # print("IsolationForest model" )
  # IsolationForest_ = IsolationForest_Classifier(save_path)
  
  # IsolationForest_.test_model(train_data_matrix, test_data_matrix, test_label, train_documents)
  # IsolationForest_.optimize_GridSearchCV(train_data_matrix,test_data_matrix,test_label)
  # IsolationForest_.optimize_parameters(train_data_matrix, test_data_matrix, test_label, train_documents, cv, n)
  
  # LOF = LocalOutlierFator_Classifier(save_path)
  # LOF.fit_model(train_data_matrix, test_data_matrix, test_label)

  # EE = EllipticEnvelope_Classifier(save_path)
  # EE.fit_model(train_data_matrix,test_data_matrix,test_label)

  print('textsim**************')
  test_label = np.array([-1]*1000+[1]*99)
  # load train data and test data
  textsim_path = '/root/data/mixcontent/zhangqi/one_class/dataset/textsim_corpus.csv'

  textsim_path1 = '/root/data/mixcontent/zhangqi/one_class/dataset/textsim_test.csv'
  with open(textsim_path) as f:
    corpus_matrix = [vec for vec in csv.reader(f)]
  train_data_matrix, test_data_matrix = corpus_matrix[:train_data_count], corpus_matrix[train_data_count:]
  train_data_matrix, test_data_matrix = np.array(train_data_matrix), np.array(test_data_matrix)
  with open(textsim_path1) as f:
    texts_corpus_matrix = [vec for vec in csv.reader(f)]
  texts_corpus_matrix = np.array(texts_corpus_matrix)
  
  print(train_data_matrix.shape, test_data_matrix.shape)
  print('OneClassSVM model')
  # train and test model
  OneClassSVM = OneClassSVMClassifier(save_path)
  # optimize parameters for test true data 
  # OneClassSVM.test_model(train_data_matrix, test_data_matrix, test_label, train_documents)
  OneClassSVM.optimize_parameters(train_data_matrix, texts_corpus_matrix, texts_label)
  # 测试
  # OneClassSVM.fit_model(train_data_matrix, train_documents)
  # OneClassSVM.test_model(texts_corpus_matrix, texts_label)

  # print("IsolationForest model" )
  # IsolationForest_ = IsolationForest_Classifier(save_path)
  # # # IsolationForest_.test_model(train_data_matrix, test_data_matrix, test_label, train_documents)
  # # # IsolationForest_.optimize_GridSearchCV(train_data_matrix,test_data_matrix,test_label)
  # IsolationForest_.optimize_parameters(train_data_matrix, test_data_matrix, test_label, train_documents, cv, n)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.train_data, args.test_data)
