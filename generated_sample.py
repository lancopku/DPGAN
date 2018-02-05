import os
import json
import time
import codecs
import tensorflow as tf
import data
import shutil
import util
import re
from  result_evaluate import Evaluate
import nltk
from nltk.translate.bleu_score import corpus_bleu
FLAGS = tf.app.flags.FLAGS

class Generated_sample(object):
    def __init__(self, model, vocab, batcher, sess):
        self._model = model
        self._vocab = vocab
        self._sess = sess
        self.batches = batcher.get_batches(mode='train')
        self.test_batches = batcher.get_batches(mode='test')
        self.current_batch = 0
        self.train_sample_whole_positive_dir = os.path.join("train","positive")
        self.train_sample_whole_negative_dir = os.path.join("train","negative")
        self.test_sample_whole_positive_dir = os.path.join("test", "positive")
        self.test_sample_whole_negative_dir = os.path.join("test", "negative")
        if not os.path.exists(self.train_sample_whole_positive_dir): os.mkdir(self.train_sample_whole_positive_dir)
        if not os.path.exists(self.train_sample_whole_negative_dir): os.mkdir(self.train_sample_whole_negative_dir)
        if not os.path.exists(self.test_sample_whole_positive_dir): os.mkdir(self.test_sample_whole_positive_dir)
        if not os.path.exists(self.test_sample_whole_negative_dir): os.mkdir(self.test_sample_whole_negative_dir)
        self.temp_positive_dir = os.path.join("temp_positive")
        self.temp_negative_dir = os.path.join("temp_negative")
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)


    def generator_sample_example(self, positive_dir, negative_dir, num_batch):

        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        counter = 0


        for i in range(num_batch):
            decode_result = self._model.run_eval_given_step(self._sess, self.batches[self.current_batch])


            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = self.batches[self.current_batch].original_review_output[i]

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)

                counter += 1  # this is how many examples we've decoded
            self.current_batch +=1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0
        
        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")


    def generator_test_sample_example(self, positive_dir, negative_dir, num_batch):

        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        counter = 0
        batches = self.test_batches
        step = 0
        list_hop = []
        list_ref = []
        

        while step < num_batch:
            
            batch = batches[step]
            step += 1

            decode_result = self._model.run_eval_given_step(self._sess, batch)
            #decode_result = self._model.run_eval_given_step(self._sess, self.batches[self.current_batch])


            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = batch.original_review_output[i]

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)
                list_ref.append([nltk.word_tokenize(original_review)])
                list_hop.append(nltk.word_tokenize(decoded_words_all))

                counter += 1  # this is how many examples we've decoded
            '''self.current_batch +=1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0'''
        
        bleu_score = corpus_bleu(list_ref, list_hop)
        tf.logging.info('bleu: '  + str(bleu_score))
        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")


    def generator_test_max_example(self, positive_dir, negative_dir, num_batch):

        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        counter = 0
        batches = self.test_batches
        step = 0
        list_hop = []
        list_ref = []

        while step < num_batch:
            
            batch = batches[step]
            step += 1

            decode_result = self._model.max_generator(self._sess, batch)
            #decode_result = self._model.run_eval_given_step(self._sess, self.batches[self.current_batch])


            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = batch.original_review_output[i]

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)
                list_ref.append([nltk.word_tokenize(original_review)])
                list_hop.append(nltk.word_tokenize(decoded_words_all))

                counter += 1  # this is how many examples we've decoded
            '''self.current_batch +=1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0'''
        
        
        bleu_score = corpus_bleu(list_ref, list_hop)
        tf.logging.info('bleu: '  + str(bleu_score))
        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")

    def generator_max_example(self, positive_dir, negative_dir, num_batch):

        self.temp_positive_dir = positive_dir
        self.temp_negative_dir = negative_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        shutil.rmtree(self.temp_negative_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negative_dir): os.mkdir(self.temp_negative_dir)
        counter = 0


        for i in range(num_batch):
            decode_result = self._model.max_generator(self._sess, self.batches[self.current_batch])


            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = self.batches[self.current_batch].original_review_output[i]

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words
                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(original_review, decoded_words_all, counter)

                counter += 1  # this is how many examples we've decoded
            self.current_batch +=1
            if self.current_batch >= len(self.batches):
                self.current_batch = 0

        eva = Evaluate()
        eva.diversity_evaluate(negative_dir + "/*")

    def write_negtive_temp_to_json(self, positive, negative, counter):
        positive_file = os.path.join(self.temp_positive_dir, "%06d.txt" % ((counter // 1000)))
        negative_file = os.path.join(self.temp_negative_dir, "%06d.txt" % ((counter // 1000)))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        write_negative_file = codecs.open(negative_file, "a", "utf-8")
        dict = {"example": str(positive),
                "label": str(1)
                }
        string_ = json.dumps(dict)
        write_positive_file.write(string_ + "\n")

        dict = {"example": str(negative),
                "label": str(0)
                }
        string_ = json.dumps(dict)
        write_negative_file.write(string_ + "\n")
        write_negative_file.close()
        write_positive_file.close()


    def write_negtive_to_json(self, positive, negative, counter, positive_dir, negtive_dir):
        positive_file = os.path.join(positive_dir, "%06d.txt" % (counter // 1000))
        negative_file = os.path.join(negtive_dir, "%06d.txt" % (counter // 1000))
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        write_negative_file = codecs.open(negative_file, "a", "utf-8")
        dict = {"example": str(positive),
                "label": str(1)
                }
        string_ = json.dumps(dict)
        write_positive_file.write(string_ + "\n")

        dict = {"example": str(negative),
                "label": str(0)
                }
        string_ = json.dumps(dict)
        write_negative_file.write(string_ + "\n")
        write_negative_file.close()
        write_positive_file.close()

    def generator_whole_negative_example(self):

        counter = 0
        step = 0

        t0 = time.time()
        batches = self.batches

        while step < 1000:
            
            batch = batches[step]
            step += 1

            decode_result = self._model.run_eval_given_step(self._sess, batch)

            for i in range(FLAGS.batch_size):
                decoded_words_all = []
                original_review = batch.original_review_output[i]  # string

                for j in range(FLAGS.max_dec_sen_num):

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
                        decoded_words.append('.')
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)

                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)

                self.write_negtive_to_json(original_review, decoded_words_all, counter, self.train_sample_whole_positive_dir, self.train_sample_whole_negative_dir)

                counter += 1  # this is how many examples we've decoded


    def generator_test_negative_example(self):

        counter = 0
        step = 0

        t0 = time.time()
        batches = self.test_batches

        while step < 100:
            step += 1
            batch = batches[step]

            decode_result =self._model.run_eval_given_step(self._sess, batch)

            for i in range(FLAGS.batch_size):
                decoded_words_all = []
                original_review = batch.original_review_output[i]  # string

                for j in range(FLAGS.max_dec_sen_num):


                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
                        decoded_words.append('.')
                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)

                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)
                self.write_negtive_to_json(original_review, decoded_words_all, counter, self.test_sample_whole_positive_dir,self.test_sample_whole_negative_dir)

                counter += 1  # this is how many examples we've decoded

    def compute_BLEU(self, train_step):

        counter = 0
        step = 0


        t0 = time.time()
        batches = self.test_batches
        list_hop = []
        list_ref = []

        #tf.logging.info(len(batches))

        while step <  100:
            #tf.logging.info(step)


            batch = batches[step]
            step += 1

            decode_result = self._model.run_eval_given_step(self._sess, batch)

            #tf.logging.info(step)

            for i in range(FLAGS.batch_size):

                #tf.logging.info("i: " + str(i))

                decoded_words_all = []
                original_review = batch.original_review_output[i]  # string

                for j in range(FLAGS.max_dec_sen_num):

                    #tf.logging.info("j: " + str(j))

                    output_ids = [int(t) for t in decode_result['generated'][i][j]][1:]
                    decoded_words = data.outputids2words(output_ids, self._vocab, None)
                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    if len(decoded_words)<2:
                        continue

                    '''if j>0:
                        new_set1 =set(decoded_words_all[j-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set1):
                            continue'''
                    if len(decoded_words_all)>0:
                        new_set1 =set(decoded_words_all[len(decoded_words_all)-1].split())
                        new_set2= set(decoded_words)
                        if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                            continue
                    if decoded_words[-1] !='.' and decoded_words[-1] !='!' and decoded_words[-1] !='?':
                        decoded_words.append('.')

                    decoded_output = ' '.join(decoded_words).strip()  # single string
                    decoded_words_all.append(decoded_output)
                decoded_words_all = ' '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
                decoded_words_all,_ = re.subn(r"(\. ){2,}", "", decoded_words_all)


                list_hop.append(decoded_words_all)
                list_ref.append(original_review)
                #self.write_negtive_to_json(original_review, decoded_output, counter)

                #counter += 1  # this is how many examples we've decoded
        file_temp = open(train_step+"_temp_result.txt",'w')
        for hop in list_hop:
            file_temp.write(hop+"\n")
        file_temp.close()
        '''new_ref_list =[]
        for ref in list_ref:
            sens = nltk.sent_tokenize(ref)
            for sen in sens:
                new_ref_list.append(nltk.word_tokenize(sen))
        t0 = time.time()
        new_sen_list =[]
        new_ref_ref =[]
        for hop in list_hop:
            sens = nltk.sent_tokenize(hop)
            for sen in sens:
                new_sen_list.append(nltk.word_tokenize(sen))

                new_ref_ref.append(new_ref_list)'''

        #print (new_sen_list)



        #bleu_score = corpus_bleu(new_ref_ref, new_sen_list)
        t1 = time.time()
        tf.logging.info('seconds for test generator: %.3f ', (t1 - t0))
        return 0

