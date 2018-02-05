# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to process data into batches"""

import queue
from random import shuffle
import codecs
import json
import glob
import numpy as np
import tensorflow as tf
import data
from nltk.tokenize import sent_tokenize

FLAGS = tf.app.flags.FLAGS
class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, review, label, vocab, hps):

    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)
    review_sentenc_orig = []


    self.hps = hps
    self.label = label

    #abstract_sentences = [x.strip() for x in abstract_sentences]
    article_sens = sent_tokenize(review)

    article_words = []
    for i in range(len(article_sens)):
        if i >= hps.max_enc_sen_num:
            article_words = article_words[:hps.max_enc_sen_num]
            review_sentenc_orig = review_sentenc_orig[:hps.max_enc_sen_num]
            break
        article_sen = article_sens[i]
        article_sen_words = article_sen.split()
        if len(article_sen_words) > hps.max_enc_seq_len:
            article_sen_words = article_sen_words[:hps.max_enc_seq_len]
        article_words.append(article_sen_words)
        review_sentenc_orig.append(article_sens[i])


    # Process the abstract
    #abstract = ' '.join(abstract_sentences)  # string
    # abstract_words = abstract.split() # list of strings
    abs_ids = [[vocab.word2id(w) for w in sen] for sen in
               article_words]  # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_enc_sen_num, hps.max_enc_seq_len,
                                                              start_decoding,
                                                             stop_decoding)  # max_sen_num,max_len, start_doc_id, end_doc_id,start_id, stop_id
    self.dec_len = len(self.dec_input)
    self.dec_sen_len = [len(sentence) for sentence in self.target]

    self.original_reivew = review_sentenc_orig


  def get_dec_inp_targ_seqs(self, sequence, max_sen_num, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """

        inps = sequence[:]
        targets = sequence[:]

        if len(inps) > max_sen_num:
            inps = inps[:max_sen_num]
            targets = targets[:max_sen_num]

        for i in range(len(inps)):
            inps[i] = [start_id] + inps[i][:]
            if len(inps[i]) > max_len:
                inps[i] = inps[i][:max_len]

        for i in range(len(targets)):
            if len(targets[i]) >= max_len:
                targets[i] = targets[i][:max_len - 1]  # no end_token
                targets[i].append(stop_id)  # end token
            else:
                targets[i] = targets[i] + [stop_id]

        return inps, targets

  def pad_decoder_inp_targ(self, max_sen_len, max_sen_num, pad_doc_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""

        while len(self.dec_sen_len) < max_sen_num:
            self.dec_sen_len.append(1)
            self.original_reivew.append("paded")

        for i in range(len(self.dec_input)):
            while len(self.dec_input[i]) < max_sen_len:
                self.dec_input[i].append(pad_doc_id)

        while len(self.dec_input) < max_sen_num:
            self.dec_input.append([pad_doc_id for i in range(max_sen_len)])

        for i in range(len(self.target)):
            while len(self.target[i]) < max_sen_len:
                self.target[i].append(pad_doc_id)

        while len(self.target) < max_sen_num:
            self.target.append([pad_doc_id for i in range(max_sen_len)])
            # print (self.target)




class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_decoder_seq(example_list, hps)  # initialize the input to the encoder

  def init_decoder_seq(self, example_list, hps):

      for ex in example_list:
          ex.pad_decoder_inp_targ(hps.max_enc_seq_len, hps.max_enc_sen_num, self.pad_id)

      # Initialize the numpy arrays.
      # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
      self.dec_batch = np.zeros((hps.batch_size, hps.max_enc_sen_num, hps.max_enc_seq_len), dtype=np.int32)
      self.target_batch = np.zeros((hps.batch_size, hps.max_enc_sen_num, hps.max_enc_seq_len), dtype=np.int32)
      self.dec_padding_mask = np.zeros((hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len),
                                       dtype=np.float32)
      self.labels = np.zeros((hps.batch_size, hps.max_enc_sen_num, hps.max_enc_seq_len), dtype=np.int32)
      self.dec_sen_lens = np.zeros((hps.batch_size, hps.max_enc_sen_num), dtype=np.int32)
      self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.review_sentenc_orig = []

      for i, ex in enumerate(example_list):
          #self.new_review_text = []
          self.labels[i]=np.array([[ex.label for k in range(hps.max_enc_seq_len) ] for j in range(hps.max_enc_sen_num)])
          self.review_sentenc_orig.append([sen for sen in ex.original_reivew])

          self.dec_lens[i] = ex.dec_len
          self.dec_batch[i, :, :] = np.array(ex.dec_input)
          self.target_batch[i] = np.array(ex.target)
          for j in range(len(ex.dec_sen_len)):
              self.dec_sen_lens[i][j] = ex.dec_sen_len[j]

      self.target_batch = np.reshape(self.target_batch,
                                     [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len])

      for j in range(len(self.target_batch)):
          for k in range(len(self.target_batch[j])):
              if int(self.target_batch[j][k]) != self.pad_id:
                  self.dec_padding_mask[j][k] = 1

      self.dec_batch = np.reshape(self.dec_batch, [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len])
      self.dec_sen_lens = np.reshape(self.dec_sen_lens, [hps.batch_size * hps.max_enc_sen_num])
      self.labels = np.reshape(self.labels, [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len])







class DisBatcher(object):
    def __init__(self, hps, vocab, train_path_positvie, train_path_negetive, test_path_positive, test_path_negetive):
        self._vocab = vocab
        self._hps = hps
        self._train_path_positive =train_path_positvie
        self._train_path_negetive = train_path_negetive
        self._test_path_positive = test_path_positive
        self._test_path_negetive = test_path_negetive

        self.train_queue = self.fill_example_queue(self._train_path_positive)
        self.train_queue += self.fill_example_queue(self._train_path_negetive)

        self.test_queue = self.fill_example_queue(self._test_path_positive)
        self.test_queue += self.fill_example_queue(self._test_path_negetive)
        #shuffle(self.all_data)

        #self.train_queue = self.all_data[:int(0.9*len(self.all_data))]
        #self.test_queue  = self.all_data[int(0.9*len(self.all_data)):]

        self.train_batch = self.create_batches(mode="train", shuffleis=True)
        self.test_batch = self.create_batches(mode="test", shuffleis=False)

    def create_batches(self, mode="train", shuffleis=True):

        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)
            if shuffleis:
                shuffle(self.train_queue)
        elif mode == 'test':
            num_batches = int(len(self.test_queue) / self._hps.batch_size)

        for i in range(0, num_batches):
            batch = []
            if mode == 'train':
                batch += (self.train_queue[i*self._hps.batch_size:i*self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue[i*self._hps.batch_size:i*self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Batch(batch, self._hps, self._vocab))
        return all_batch

    def get_batches(self, mode="train"):

        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch

        elif mode == 'test':
            return self.test_batch




    def fill_example_queue(self, data_path):

        new_queue =[]

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
                if review.strip() =="":
                    continue
                label = dict_example["label"]
                if int(label) == 1:
                    label = -0.0001
                elif int(label) == 0:
                    label = 1
                example = Example(review, label, self._vocab, self._hps)
                new_queue.append(example)
        return new_queue



