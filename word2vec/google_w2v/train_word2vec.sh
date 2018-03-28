# shell commend word2vec train
# 指定分词后的文件, 处理后保存的文件以及线程数
# time python replace_UNK_by_frequency.py --data_={train corpus data} --save_={replace nuk file path} --num_thread=20
# 处理后的语料训练word2vec
# time ./word2vec -train {replace nuk file path} -output ./word2vec.bin -cbow 1 -size 100 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 60 -binary 1 -iter 15

# word2vec train
time python3 replace_UNK_by_frequency.py --segment_corpus=/root/data/mixcontent/zhangqi/sentiment/v0.0.3/word2vec/general_w2vcorpus.txt 
--word2vec_corpus=/root/data/mixcontent/zhangqi/sentiment/v0.0.3/word2vec/general_w2vcorpus_unk.txt --num_thread=20 --limit_size=100 --min_count=5
time ./word2vec -train /root/data/mixcontent/zhangqi/sentiment/v0.0.3/word2vec/general_w2vcorpus_unk.txt 
-output /root/data/mixcontent/zhangqi/sentiment/v0.0.3/word2vec/general_word2vec.bin -cbow 1 -size 100 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 60 -binary 1 -iter 15
