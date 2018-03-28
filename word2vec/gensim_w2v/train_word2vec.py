# coding:utf-8
import os
import sys
import time
import argparse
import numpy as np
import threading
import multiprocessing
import gzip
import shutil
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors




parser = argparse.ArgumentParser(description="train word2vec for original corpus")

# step for prepare word2vec corpus parameters
parser.add_argument('--num_thread', default=10, type=int, help='线程数')
parser.add_argument('--segment_corpus', default=None, help='原始分词语料')
parser.add_argument('--min_count', default=5, type=int, help='替换为<UNK>的词频')
parser.add_argument('--limit_size', default=100, type=int, help='拆分文件的大小')
parser.add_argument('--mode', default='train', choices=['train','eval'], help='训练或测试模型')

# gensim word2vec parameters
parser.add_argument('--sentences', default=None, help='可迭代的句子')
parser.add_argument('--window', default=5, type=int, help='句子中当前词和预测词之间的最大距离')
parser.add_argument('--size', default=100, type=int, help='向量的维数')
parser.add_argument('--sample', default=1e-5, type=float, help='用于配置哪些更高频率的词随机下采样的阈值,有用的范围是(0, 1e-5)')
parser.add_argument('--negative', default=25, type=int, help='负面抽样')
parser.add_argument('--alpha', default=0.025, type=float, help='最初的学习率。')
parser.add_argument('--hs', default=0, type=int, help='softmax用于模型训练')
parser.add_argument('--sg', default=0, choices=[0,1], type=int, help='选择训练算法.  1用skip-gram ,否则CBOW')
parser.add_argument('--iter', default=15, type=int, help='整个语料库的迭代次数epochs')
parser.add_argument('--seed', default=1, type=int, help='随机数发生器的种子')
parser.add_argument('--trim_rule', default=None, help='词汇修剪规则，指定某些单词是否应该保留在词汇表中，并予以修剪')


def get_FileSize(filePath):
  '''获取文件的大小,结果保留两位小数，单位为MB'''
  fsize = os.path.getsize(filePath)
  fsize = fsize/float(1024*1024)
  return round(fsize,2)

def spilt_fileBySize(large_file, file_name, save_path, limit_size=100):
  """将大文件拆分成 limit_size 大小的多个文件"""
  file_count = 0
  # path_list = large_file.split('/')
  # file_name = path_list[-1].split('.')[0]
  parts_path = os.path.join(save_path, file_name+'_split_parts')
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  try:
    with open(large_file, encoding='utf-8') as lar_f:
      file = os.path.join(save_path,"part_%d"%file_count)
      fout = open(file,"w")
      for line in lar_f:
        fout.write(line)
        if get_FileSize(file) >= limit_size:
          fout.close()
          file_count+=1
          file = os.path.join(save_path,"part_%d"%file_count)
          fout = open(file,"w")
      fout.close()
  except Exception as e:
    raise e
  return parts_path


def get_vocab_freq(large_file, save_path, file_name, args):
  """获取各词及其词频""" 
  vocabs_dict = {}
  replace_words = set()
  split_words = set()
  vocabs_freq_path = os.path.join(save_path,'vocabulary_frequency.txt')
  
  # split files path
  parts_path = os.path.join(save_path, file_name+'_split_parts')
  if not os.path.exists(parts_path):
    os.makedirs(parts_path)

  if not os.path.exists(vocabs_freq_path): 
    file_count = 0
    with open(large_file) as f:
      file_part = os.path.join(parts_path,"part_%d"%file_count)
      fout = open(file_part,"w")
      for i, line in enumerate(f):
        try:
          words = line.strip().split(' ')
          for word in words:
            if word in vocabs_dict:
              vocabs_dict[word] += 1
            elif word:
               vocabs_dict[word] = 1
        except Exception as e:
          print("error line {},{}".format(line,e))
          raise e
        # split file 
        fout.write(line)
        if get_FileSize(file_part) >= args.limit_size:
          fout.close()
          file_count+=1
          file_part = os.path.join(parts_path,"part_%d"%file_count)
          fout = open(file_part,"w")

        # see Progress
        if i%100000==0:
          print("vocabs frequency process %d"% i)
          sys.stdout.flush()
      fout.close()

    # write vocabs
    with open(vocabs_freq_path, 'w') as vocabs_f:
      for word, word_freq in sorted(vocabs_dict.items(),key=lambda x:x[1],reverse=True):
        vocabs_f.write("{} {}\n".format(word, word_freq))
  else:
    # read vocabs
    with open(vocabs_freq_path) as vocabs_f:
      for line in vocabs_f:
        word, num = line.split(' ')
        vocabs_dict[word] = int(num)

  # find split words and replace "<UNK>" words  
  for word in vocabs_dict:
    if vocabs_dict[word] < args.min_count:
      if len(word)>1:
        split_words.add(word)
        for w in word:
          if w in vocabs_dict and vocabs_dict[w] > args.min_count:
            vocabs_dict[w]+=1
          else:
            replace_words.add(w)
      else:
        replace_words.add(word)
        
  # write result 
  with open(os.path.join(save_path,'split_replaced_words.txt'), 'w') as unk_f:
    unk_f.write("********split words********\n")
    for split_word in split_words:
      unk_f.write(split_word+"\n")
    unk_f.write("********replaced UNK words********\n")
    for replace_word in replace_words:
      unk_f.write(replace_word+"\n")
    return split_words, replace_words, parts_path


def replace_UNK_by_frequency(file_list, save_f, split_words, replace_words):
  """先拆分词，再替换<UNK>"""
  for file in file_list:
    print('{} process {}'.format(threading.current_thread().name,file))
    with open(file) as f:
      try:
        for i, line in enumerate(f):
          words = line.strip().split(' ')
          new_words = []
          for word in words:
            if word == "" or word == " ": continue
            if word in split_words:
              for w in word:
                if w in replace_words:
                  new_words.append("<UNK>")
                else:
                  new_words.append(word)
            elif word in replace_words:
                new_words.append("<UNK>")
            else:
              new_words.append(word)
          # new_line = ' '.join(new_words)
          # new_line = new_line.encode('utf-8')
          # print(type(new_line))
          save_f.write("{}\n".format(' '.join(new_words)))
      except Exception as e:
        print('line : %s' % line)
        raise e
    print('{} finish {}'.format(threading.current_thread().name,file))
    sys.stdout.flush()


def prepare_corpus(read_file, num_thread, args):
  """多线程处理预料"""
  begin = time.time()
  # path 
  path_list = read_file.split('/')
  save_path = '/'.join(path_list[:-1])
  file_name = path_list[-1].split('.')[0]
  split_words, replace_words, parts_path = get_vocab_freq(read_file, save_path, file_name, args)
  word2vec_corpus_file = os.path.join(save_path, "%s_unk.txt" % file_name)
  # save_f = gzip.open(word2vec_corpus_file, 'wb')
  save_f = open(word2vec_corpus_file, 'w')

  # mutiple threads concurrent
  # tmp_path = spilt_fileBySize(read_file, file_name)
  all_files = [os.path.join(parts_path, file) for file in os.listdir(parts_path)]
  print('Will process files %d'%len(all_files))
  sys.stdout.flush()
  # Thread Assignment Task
  files_space = np.linspace(0,len(all_files),num_thread+1, dtype=np.int)
  thread_files = [all_files[files_space[i]:files_space[i+1]] for i in range(num_thread)]
  # thread begin
  workers = [threading.Thread(target=replace_UNK_by_frequency, args=(files_list, save_f, split_words, replace_words)) for files_list in thread_files]
  for thread in workers:
    # thread.daemon = True  #设置线程为后台线程
    thread.start()
  # 阻塞线程
  for thread in workers:
    thread.join()

  # 删除分割文件
  save_f.close()
  shutil.rmtree(parts_path)
  print("finished spend %d seconds" % int(time.time()-begin))
  return word2vec_corpus_file, file_name


class Word2vecModel():
  """ word2vec parameters"""
  def __init__(self, args):
    self.args = args
    self.save_path = '/'.join(self.args.segment_corpus.split('/')[:-1])
    self.model_path = os.path.join(self.save_path, "model_path")
    if not os.path.exists(self.model_path):
      os.makedirs(self.model_path)
    corpus_file, file_name = prepare_corpus(self.args.segment_corpus, self.args.num_thread, args)
    self.file_name = file_name
    self.corpus_file = corpus_file
    self.model_file  = os.path.join(self.model_path, "%s_general.model.bin"% file_name)

  def train_word2vec(self):
    """gensim train word2vec"""
    # model_file = os.path.join(model_path, "word2vec.model")
    # model = Word2vecModel(args, corpus_file)
    model = Word2Vec(sentences = LineSentence(self.corpus_file),
                    sg = self.args.sg,
                    size = self.args.size,
                    window = self.args.window,
                    alpha = self.args.alpha,
                    min_alpha = 0.0001,
                    seed = self.args.seed,
                    sample = self.args.sample,
                    negative = self.args.negative,
                    hs = self.args.hs,
                    iter = self.args.iter,
                    workers = multiprocessing.cpu_count(),
                    )
    #save model
    model.wv.save_word2vec_format(self.model_file, binary=True)

      # save vocabulary
      # vocabs_vectors_file = os.path.join(self.model_path, 'vocabs_vectors.txt')
      # if not os.path.exists(vocabs_vectors_file):
      #   with open(vocabs_vectors_file, 'w') as voacb_f:
      #     for word in model.wv.vocab:
      #       vocab_count = model.wv.vocab[word].count
      #       word_vector = model[word]
      #       # word = word.encode('utf-8')
      #       voacb_f.write("{},{},{}\n".format(word, vocab_count, word_vector))
      # model.save(model_file)
  def model_jugement(self, word):
    word_vectors = KeyedVectors.load_word2vec_format(self.model_file, binary=True)
    similar_words = word_vectors.similar_by_word(word)
    for word in similar_words:
      print(word)

if __name__ == '__main__':
  begin = time.time()
  args = parser.parse_args()
  word2vec = Word2vecModel(args)
  if args.mode == 'train':
    word2vec.train_word2vec(args)
    print('train finished spend %d' % int(time.time()-begin))
  elif args.mode=='eval':
    while True:
      print('inport word to see similar words')
      word = input("now input word:")
      word2vec.model_jugement(word)