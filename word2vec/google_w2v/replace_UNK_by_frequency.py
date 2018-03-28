# coding:utf-8
import os
import sys
import time
import argparse
import numpy as np
import threading
import multiprocessing
import shutil

parser = argparse.ArgumentParser(description="prepare train word2vec corpus")
parser.add_argument('--num_thread', default=10, type=int, help='Number of threads concurrent')
parser.add_argument('--segment_corpus', default=None, help='segment corpus file, must be file')
parser.add_argument('--word2vec_corpus', default=None, help='save results to train word2vec file, must be file')
parser.add_argument('--limit_size', default=100, type=int, help='spilt large file with limit size')
parser.add_argument('--min_count', default=5, type=int, help='替换为<UNK>的词频')

"""
  shell commend
  time python replace_UNK_by_frequency.py --data_={train corpus data} --save_={replace nuk file path}
  time time ./word2vec -train {replace nuk file path} -output ./word2vec.bin -cbow 1 -size 100 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 60 -binary 1 -iter 15
"""

def get_FileSize(filePath):
  '''获取文件的大小,结果保留两位小数，单位为MB'''
  fsize = os.path.getsize(filePath)
  fsize = fsize/float(1024*1024)
  return round(fsize,2)

def spilt_fileBySize(large_file, limit_size=100):
  """"""
  if get_FileSize(large_file) > limit_size:
    file_count = 0
    path_list = large_file.split('/')
    file_name = path_list[-1].split('.')[0]
    save_path = os.path.join('/'.join(path_list[:-1]),file_name+'_split_parts')
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
    return save_path


def get_vocab_freq(args, save_path):
  """获取各词及其词频""" 
  vocabs_dict = {}
  replace_words = set()
  split_words = set()
  vocabs_freq_path = os.path.join(save_path,'vocabulary_frequency.txt')
  
  # split files path
  path_list = args.segment_corpus.split('/')
  file_name = path_list[-1].split('.')[0]
  parts_path = os.path.join('/'.join(path_list[:-1]),file_name+'_split_parts')
  if not os.path.exists(parts_path):
    os.makedirs(parts_path)
  if not os.path.exists(vocabs_freq_path): 
    #
    file_count = 0
    with open(args.segment_corpus) as f:
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
    with open(vocabs_freq_path, 'w') as vocabs_f:
      for word, word_freq in sorted(vocabs_dict.items(),key=lambda x:x[1],reverse=True):
        vocabs_f.write("{} {}\n".format(word, word_freq))
  else:
    with open(vocabs_freq_path) as vocabs_f:
      for line in vocabs_f:
        word, num = line.split(' ')
        vocabs_dict[word] = int(num)
    
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
  with open(os.path.join(save_path,'splited_words.txt'), 'w') as split_f:
      for split_word in split_words:
        split_f.write(split_word+"\n")
  with open(os.path.join(save_path,'replaced_words.txt'), 'w') as unk_f:
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


def main(args, num_thread):
  """"""
  begin = time.time()
  save_path = '/'.join(args.segment_corpus.split('/')[:-1])
  split_words, replace_words, parts_path = get_vocab_freq(args, save_path)
  print("split_words length {}, replace_words length {}".format(len(split_words),len(replace_words)))
  save_f = open(args.word2vec_corpus, 'w')
  # mutiple threads concurrent
  # parts_path = spilt_fileBySize(read_file)
  all_files = [os.path.join(parts_path, file) for file in os.listdir(parts_path)]
  print('Will process files %d'%len(all_files))
  sys.stdout.flush()
  # Thread Assignment Task
  files_space = np.linspace(0,len(all_files),num_thread+1,dtype=np.int)
  thread_files = [all_files[files_space[i]:files_space[i+1]] for i in range(num_thread)]
  # thread begin
  workers = [threading.Thread(target=replace_UNK_by_frequency, args=(files_list, save_f, split_words, replace_words)) for files_list in thread_files]
  for thread in workers:
    # thread.daemon = True  # make interrupting the process with ctrl+c easier #设置线程为后台线程
    thread.start()
  # 阻塞线程
  for thread in workers:
    thread.join()

  # 删除分割文件
  save_f.close()
  shutil.rmtree(parts_path)
  print("finished spend %d seconds" % int(time.time()-begin))


if __name__ == '__main__':
  args = parser.parse_args()
  main(args, args.num_thread)
