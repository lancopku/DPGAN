import os
import json
import time
import codecs
import tensorflow as tf
import glob
from nltk import sent_tokenize
FLAGS = tf.app.flags.FLAGS

class Evaluate(object):

    def __init__(self ):
        self.all_bigram=dict()
        self.all_trigram=dict()
        self.all_unigram=dict()
        self.all_sentence=dict()
        self.bigram_num=0
        self.trigram_num=0
        self.unigram_num=0
        self.sen_num=0



    def diversity_evaluate(self, data_path):


        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path)  # check filelist isn't empty
        filelist = sorted(filelist)
        for f in filelist:
            reader = codecs.open(f, 'r', 'utf-8')
            while True:
                string_ = reader.readline()
                if not string_: break
                dict_example = json.loads(string_)
                review = dict_example["example"]
                review_sen= sent_tokenize(review)
                for sen in review_sen:
                    self.all_sentence[sen] =1
                    self.sen_num+=1
                    sen_words=sen.strip().split()
                    unigram = [sen_words[i] for i in range(len(sen_words))]
                    if len(sen_words)>=2:
                        bigram = [sen_words[i]+sen_words[i+1] for i in range(len(sen_words)-2)]
                    else:
                        bigram = []
                    if len(sen_words)>=3:
                        trigram = [sen_words[i] + sen_words[i + 1] + sen_words[i + 2] for i in range(len(sen_words) - 3)]
                    else:
                        trigram = []
                    for word in bigram:
                        self.all_bigram[word]=1
                        self.bigram_num+=1
                    for word in trigram:
                        self.all_trigram[word]=1
                        self.trigram_num+=1
                    for word in unigram:
                        self.all_unigram[word]=1
                        self.unigram_num+=1


        tf.logging.info("sentence number: "+str(self.sen_num)+" unique sentence number: "+str(len(self.all_sentence))+" unique sentence rate: "+str(len(self.all_sentence)/(1.0*self.sen_num)))
        tf.logging.info("unigram number: "+str(self.unigram_num)+" unique unigram number: "+str(len(self.all_unigram))+" unique unigram rate: "+str(len(self.all_unigram)/(1.0*self.unigram_num)))
        tf.logging.info("bigram number: " + str(self.bigram_num) + " unique bigram number: " + str(
            len(self.all_bigram)) + " unique bigram rate: " + str(len(self.all_bigram) / (1.0 * self.bigram_num)))
        tf.logging.info("trigram number: " + str(self.trigram_num) + " unique trigram number: " + str(
            len(self.all_trigram)) + " unique trigram rate: " + str(len(self.all_trigram) / (1.0 * self.trigram_num)))

