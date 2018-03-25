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

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
from random import shuffle
import time
import codecs
import data
import os
import math
import tensorflow as tf
import numpy as np
from collections import namedtuple
import batcher_discriminator as bd
from data import Vocab
from batcher import Example
from batcher import Batch
from batcher import GenBatcher
from batcher_discriminator import DisBatcher
from model import Generator
from discriminator import Discriminator
import json
from generated_sample import  Generated_sample
from result_evaluate import Evaluate
import util
import re

import nltk
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', 'review_generation_dataset/train/* ', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', 'review_generation_dataset/vocab.txt', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of adversarial_train/train_generator/train_discriminator')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'myexperiment', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')


tf.app.flags.DEFINE_integer('gpuid', 0, 'for gradient clipping')
tf.app.flags.DEFINE_string('dataset', 'yelp', "dataset which you use")
tf.app.flags.DEFINE_string('run_method', 'auto-encoder', 'must be one of auto-encoder/language_model')

tf.app.flags.DEFINE_integer('max_enc_sen_num', 6, 'max timesteps of encoder (max source text tokens)')   # for discriminator
tf.app.flags.DEFINE_integer('max_enc_seq_len', 40, 'max timesteps of encoder (max source text tokens)')   # for discriminator

tf.app.flags.DEFINE_integer('max_dec_sen_num',6, 'max timesteps of decoder (max source text tokens)')   # for generator
tf.app.flags.DEFINE_integer('max_dec_steps', 40, 'max timesteps of decoder (max source text tokens)')   # for generator


# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states') # for discriminator and generator
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings') # for discriminator and generator
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size') # for discriminator and generator
tf.app.flags.DEFINE_integer('max_enc_steps', 50, 'max timesteps of encoder (max source text tokens)') # for generator
#tf.app.flags.DEFINE_integer('max_dec_steps', 200, 'max timesteps of decoder (max summary tokens)') # for generator
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode') # for generator
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.6, 'learning rate') # for discriminator and generator
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad') # for discriminator and generator
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization') # for discriminator and generator
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else') # for discriminator and generator
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping') # for discriminator and generator


'''the generator model is saved at FLAGS.log_root + "train-generator"
   give up sv, use sess
'''
def setup_training_generator(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train-generator")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph

  saver = tf.train.Saver(max_to_keep=10)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()
  sess.run(init)

  # Load an initial checkpoint to use for decoding
  #util.load_ckpt(saver, sess, ckpt_dir="train-generator")


  return sess, saver,train_dir


def setup_training_discriminator(model):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(FLAGS.log_root, "train-discriminator")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph()  # build the graph

    saver = tf.train.Saver(max_to_keep=10)  # we use this to load checkpoints for decoding
    sess = tf.Session(config=util.get_config())
    init = tf.global_variables_initializer()
    sess.run(init)
    #util.load_ckpt(saver, sess, ckpt_dir="train-discriminator")



    return sess, saver,train_dir


def print_batch(batch):
    '''tf.logging.info("enc_batch")
    tf.logging.info(list(batch.enc_batch))
    tf.logging.info("enc_lens")
    tf.logging.info(list(batch.enc_lens))


    tf.logging.info('dec_batch')
    tf.logging.info(list(batch.dec_batch))

    tf.logging.info('target_batch')
    tf.logging.info(list(batch.target_batch))

    tf.logging.info('dec_padding_mask')
    tf.logging.info(list(batch.dec_padding_mask))'''
    tf.logging.info(batch.original_reviews)




def run_pre_train_generator(model, batcher, max_run_epoch, sess, saver, train_dir, generated):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            #print_batch(current_batch)
            step += 1
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 100 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                #bleu_score = generated.compute_BLEU(str(train_step))
                #tf.logging.info('bleu: %f', bleu_score)  # print the loss to screen

        epoch += 1
        tf.logging.info("finished %d epoches", epoch)


def batch_to_batch(batch, batcher, dis_batcher):

    db_example_list = []

    for i in range(FLAGS.batch_size):

        new_dis_example = bd.Example(batch.original_review_output[i], -0.01, dis_batcher._vocab, dis_batcher._hps)
        db_example_list.append(new_dis_example)

    return bd.Batch(db_example_list, dis_batcher._hps, dis_batcher._vocab)

def output_to_batch(current_batch, result, batcher, dis_batcher):
    example_list= []
    db_example_list = []

    for i in range(FLAGS.batch_size):
        decoded_words_all = []
        encode_words = current_batch.original_review_inputs[i]

        for j in range(FLAGS.max_dec_sen_num):

            output_ids = [int(t) for t in result['generated'][i][j]][1:]
            decoded_words = data.outputids2words(output_ids, batcher._vocab, None)
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words
            if len(decoded_words) < 2:
                continue
            if len(decoded_words_all) > 0:
                new_set1 = set(decoded_words_all[len(decoded_words_all) - 1].split())
                new_set2 = set(decoded_words)
                if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                    continue
            if decoded_words[-1] != '.' and decoded_words[-1] != '!' and decoded_words[-1] != '?':
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

        if decoded_words_all.strip() == "":
            '''tf.logging.info("decode")
            tf.logging.info(current_batch.original_reviews[i])
            tf.logging.info("encode")
            tf.logging.info(encode_words)'''
            new_dis_example = bd.Example(current_batch.original_review_output[i], -0.0001, dis_batcher._vocab, dis_batcher._hps)
            new_example = Example(current_batch.original_review_output[i],  batcher._vocab, batcher._hps,encode_words)

        else:
            '''tf.logging.info("decode")
            tf.logging.info(decoded_words_all)
            tf.logging.info("encode")
            tf.logging.info(encode_words)'''
            new_dis_example = bd.Example(decoded_words_all, 1, dis_batcher._vocab, dis_batcher._hps)
            new_example = Example(decoded_words_all, batcher._vocab, batcher._hps,encode_words)
        example_list.append(new_example)
        db_example_list.append(new_dis_example)

    return Batch(example_list, batcher._hps, batcher._vocab), bd.Batch(db_example_list, dis_batcher._hps, dis_batcher._vocab)
def run_train_generator(model, discirminator_model, discriminator_sess, batcher, dis_batcher, batches, sess, saver, train_dir, generated):
    tf.logging.info("starting training generator")

    step = 0
    t0 = time.time()
    loss_window = 0.0
    new_loss_window = 0.0
    while step < len(batches):
        current_batch = batches[step]
        step += 1

        for i in range(1):
            results = model.run_eval_given_step(sess, current_batch)

            new_batch, new_dis_batch = output_to_batch(current_batch, results, batcher, dis_batcher)


            reward = discirminator_model.run_ypred_auc(discriminator_sess,new_dis_batch)
            reward_sentence_level = reward['y_pred_auc_sentence']
            for i in range(len(reward['y_pred_auc'])):
                for j in range(len(reward['y_pred_auc'][i])):
                  for k in range(len(reward['y_pred_auc'][i][j])):

                       if reward['y_pred_auc'][i][j][k] > 12:
                          reward['y_pred_auc'][i][j][k] = 12/ 10000.0 
                       else:
                          reward['y_pred_auc'][i][j][k] = reward['y_pred_auc'][i][j][k] / 10000.0


                    
            reward['y_pred_auc'] = np.reshape(np.array(reward['y_pred_auc']), [batcher._hps.batch_size*batcher._hps.max_dec_sen_num,batcher._hps.max_dec_steps])
            #reward = [math.fabs(re-0.3) for re in reward['y_pred_auc'][:,1]]
            #for i in range(batcher._hps.max_dec_steps):
            #    reward[i] = 1

            results = model.run_train_step(sess, new_batch,reward['y_pred_auc'])

            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

        new_dis_batch = batch_to_batch(current_batch, batcher, dis_batcher)
        # print_batch(new_batch)


        reward = discirminator_model.run_ypred_auc(discriminator_sess, new_dis_batch)
        reward_sentence_level = reward['y_pred_auc_sentence']

        for i in range(len(reward['y_pred_auc'])):
            for j in range(len(reward['y_pred_auc'][i])):
              for k in range(len(reward['y_pred_auc'][i][j])):

                if reward['y_pred_auc'][i][j][k] > 12:
                    reward['y_pred_auc'][i][j][k] = 1
                else:
                    reward['y_pred_auc'][i][j][k] = reward['y_pred_auc'][i][j][k] / 10.0



                

        reward['y_pred_auc'] = np.reshape(np.array(reward['y_pred_auc']),
                                          [FLAGS.batch_size * batcher._hps.max_dec_sen_num,batcher._hps.max_dec_steps])
        #results = model.run_train_step(sess, current_batch, reward['y_pred_auc'])
        new_results = model.run_train_step(sess, current_batch,
                                           reward['y_pred_auc'])
        new_loss = new_results['loss']
        new_loss_window += new_loss
        if not np.isfinite(new_loss):
            raise Exception("new Loss is not finite. Stopping.")
        train_step = new_results['global_step']  # we need this to update our running average loss







        '''if train_step % 10000 == 0:
            #saver.save(sess, train_dir + "/model", global_step=train_step)
            bleu_score = generated.compute_BLEU(str(train_step))
            tf.logging.info('bleu: %f', bleu_score)  # print the loss to screen'''

    t1 = time.time()
    tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / len(batches))
    tf.logging.info('loss: %f', loss_window / (len(batches)/ len(batches)))  # print the loss to screen

    tf.logging.info('teach forcing loss: %f', new_loss_window / len(batches))  # print the loss to screen


def print_discriminator_batch(batch):
    tf.logging.info("enc_batch")
    tf.logging.info(list(batch.enc_batch))
    tf.logging.info("enc_sen_lens")
    tf.logging.info(list(batch.enc_sen_lens))


    tf.logging.info('labels')
    tf.logging.info(list(batch.labels))

    tf.logging.info('target_mask')
    tf.logging.info(list(batch.target_mask))




def run_pre_train_discriminator(model, bachter, max_run_epoch, sess,saver, train_dir):
    tf.logging.info("starting run_pre_train_discriminator")

    epoch = 0
    while epoch < max_run_epoch:
        batches = bachter.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            step += 1
            #print_discriminator_batch(current_batch)
            results = model.run_pre_train_step(sess, current_batch)

            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training dirscriminator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0

            if train_step % 100 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                run_test_discriminator(model, bachter, sess, saver, str(train_step))
                #tf.logging.info('acc: %.6f', acc)  # print the loss to screen

        epoch +=1
        tf.logging.info("finished %d epoches", epoch)

def run_test_discriminator(model, batcher, sess,saver, train_step):
    tf.logging.info("starting run testing discriminator")

    discriminator_file = codecs.open("discriminator_result/"+train_step+ "discriminator.txt","w","utf-8")

    batches = batcher.get_batches("test")
    step = 0
    right =0.0
    all = 0.0
    while step < len(batches):
        current_batch = batches[step]
        step += 1
        result = model.run_ypred_auc(sess, current_batch)
        outloss=result['y_pred_auc']
        outloss_sentence = result['y_pred_auc_sentence']

        for i in range(FLAGS.batch_size):
            for j in range(batcher._hps.max_enc_sen_num):
                #print ([outloss[i][j][k] for k in range(len(outloss[i][j]))])
                a ={"example": current_batch.review_sentenc_orig[i][j], "score": [np.float64(outloss[i][j][k]) for k in range(len(outloss[i][j]))], "sentence_level_score" : np.float64(outloss_sentence[i][j])}
                string_a = json.dumps(a)
                discriminator_file.write(string_a+"\n")
    discriminator_file.close()
    return 0



def run_train_discriminator(model, max_epoch, batcher, batches, sess,saver, train_dir, whole_decay=False):
    tf.logging.info("starting trining discriminator")
    #batches = batcher.get_batches("train")

    step = 0
    t0 = time.time()
    loss_window = 0.0
    right = 0.0
    number = 0.0
    epoch =0
    while epoch < max_epoch:
        epoch+=1

        while step < len(batches):

            current_batch = batches[step]
            step += 1
            results = model.run_pre_train_step(sess, current_batch)

            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training dirscriminator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
               # tf.logging.info('acc: %f', right / number)  # print the loss to screen
                loss_window = 0.0


            if train_step % 10000 == 0:
                #saver.save(sess, train_dir + "/model", global_step=train_step)
                run_test_discriminator(model, batcher, sess, saver, str(train_step))
    return whole_decay


def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting running in %s mode...', (FLAGS.mode))

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if "train" in FLAGS.mode:
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary


  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_sen_num','max_dec_steps', 'max_enc_steps']
  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps_generator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  hparam_list = ['lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                 'hidden_dim', 'emb_dim', 'batch_size', 'max_enc_sen_num', 'max_enc_seq_len']
  hps_dict = {}
  for key, val in FLAGS.__flags.items():  # for each flag
      if key in hparam_list:  # if it's in the list
          hps_dict[key] = val  # add it to the dict
  hps_discriminator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data
  batcher = GenBatcher(vocab, hps_generator)




  tf.set_random_seed(111) # a seed value for randomness





  if hps_generator.mode == 'adversarial_train':
    print("Start pre-training......")
    model = Generator(hps_generator, vocab)

    sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)
    generated = Generated_sample(model, vocab, batcher, sess_ge)
    #print("Start pre-training generator......")
    #run_pre_train_generator(model, batcher, 1, sess_ge, saver_ge, train_dir_ge,generated) # this is an infinite loop until 

    #print("Generating negative examples......")
    #generated.generator_train_negative_example()
    #generated.generator_test_negative_example()

    model_dis = Discriminator(hps_discriminator, vocab)
    dis_batcher = DisBatcher(hps_discriminator, vocab, "discriminator_train/positive/*", "discriminator_train/negative/*", "discriminator_test/positive/*", "discriminator_test/negative/*")
    sess_dis, saver_dis, train_dir_dis = setup_training_discriminator(model_dis)
    #print("Start pre-training discriminator......")
    #run_test_discriminator(model_dis, dis_batcher, sess_dis, saver_dis, "test")
    if not os.path.exists("discriminator_result"): os.mkdir("discriminator_result")
    #run_pre_train_discriminator(model_dis, dis_batcher, 1, sess_dis, saver_dis, train_dir_dis)
    
    util.load_ckpt(saver_dis, sess_dis, ckpt_dir="train-discriminator")

    util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
    
    
    
    #print("generate training data for discriminator")
    #generated.generator_sample_example("discriminator_train_sample_positive", "discriminator_train_negative", 1000)
    
    if not os.path.exists("MLE"): os.mkdir("MLE")

    print("evaluate the diversity of MLE (decode based on sampling)")
    generated.generator_test_sample_example("MLE/"+"MLE_sample_positive",
                                       "MLE/"+"MLE_sample_negative",
                                       200)
                                       
    print("evaluate the diversity of MLE (decode based on max probability)")
    generated.generator_test_max_example("MLE/"+"MLE_max_temp_positive",
                                       "MLE/"+"MLE_max_temp_negative",
                                       200)
    #tf.logging.info("true data diversity: ")
    #eva = Evaluate()
    #print("evaluate the diversity of true data")
    #eva.diversity_evaluate("test_sample_temp_positive" + "/*")



    print("Start adversarial  training......")
    if not os.path.exists("train_sample_generated"): os.mkdir("train_sample_generated")
    if not os.path.exists("test_max_generated"): os.mkdir("test_max_generated")
    if not os.path.exists("test_sample_generated"): os.mkdir("test_sample_generated")
    
    
    
    whole_decay = False
    for epoch in range(1):
        batches = batcher.get_batches(mode='train')
        for step in range(int(len(batches)/1000)):

            run_train_generator(model,model_dis,sess_dis,batcher,dis_batcher,batches[step*1000:(step+1)*1000],sess_ge, saver_ge, train_dir_ge,generated) #(model, discirminator_model, discriminator_sess, batcher, dis_batcher, batches, sess, saver, train_dir, generated):
            generated.generator_sample_example("train_sample_generated/"+str(epoch)+"epoch_step"+str(step)+"_temp_positive", "train_sample_generated/"+str(epoch)+"epoch_step"+str(step)+"_temp_negative", 1000)
            #generated.generator_max_example("max_generated/"+str(epoch)+"epoch_step"+str(step)+"_temp_positive", "max_generated/"+str(epoch)+"epoch_step"+str(step)+"_temp_negetive", 200)

            tf.logging.info("test performance: ")
            tf.logging.info("epoch: "+str(epoch)+" step: "+str(step))
            print("evaluate the diversity of DP-GAN (decode based on  max probability)")
            generated.generator_test_sample_example(
                "test_sample_generated/" + str(epoch) + "epoch_step" + str(step) + "_temp_positive",
                "test_sample_generated/" + str(epoch) + "epoch_step" + str(step) + "_temp_negative", 200)
            print("evaluate the diversity of DP-GAN (decode based on sampling)")
            generated.generator_test_max_example("test_max_generated/" + str(epoch) + "epoch_step" + str(step) + "_temp_positive",
                                            "test_max_generated/" + str(epoch) + "epoch_step" + str(step) + "_temp_negative",
                                            200)

            dis_batcher.train_queue = []
            dis_batcher.train_queue = []
            for i in range(epoch+1):
              for j in range(step+1):
                dis_batcher.train_queue += dis_batcher.fill_example_queue("train_sample_generated/"+str(i)+"epoch_step"+str(j)+"_temp_positive/*")
                dis_batcher.train_queue += dis_batcher.fill_example_queue("train_sample_generated/"+str(i)+"epoch_step"+str(j)+"_temp_negative/*")
            dis_batcher.train_batch = dis_batcher.create_batches(mode="train", shuffleis=True)

            #dis_batcher.valid_batch = dis_batcher.train_batch
            whole_decay = run_train_discriminator(model_dis, 5, dis_batcher, dis_batcher.get_batches(mode="train"),
                                                  sess_dis, saver_dis, train_dir_dis, whole_decay)

  elif hps_generator.mode == 'train_generator':
    print("Start pre-training......")
    model = Generator(hps_generator, vocab)

    sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)
    generated = Generated_sample(model, vocab, batcher, sess_ge)
    print("Start pre-training generator......")
    run_pre_train_generator(model, batcher, 4, sess_ge, saver_ge, train_dir_ge,generated) # this is an infinite loop until 

    print("Generating negative examples......")
    generated.generator_train_negative_example()
    generated.generator_test_negative_example()
  elif hps_generator.mode == 'train_discriminator':
    print("Start pre-training......")
    model = Generator(hps_generator, vocab)

    sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)
    #generated = Generated_sample(model, vocab, batcher, sess_ge)
    #print("Start pre-training generator......")
    #run_pre_train_generator(model, batcher, 1, sess_ge, saver_ge, train_dir_ge,generated) # this is an infinite loop until 

    #print("Generating negative examples......")
    #generated.generator_train_negative_example()
    #generated.generator_test_negative_example()
    #util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")

    model_dis = Discriminator(hps_discriminator, vocab)
    dis_batcher = DisBatcher(hps_discriminator, vocab, "discriminator_train/positive/*", "discriminator_train/negative/*", "discriminator_test/positive/*", "discriminator_test/negative/*")
    sess_dis, saver_dis, train_dir_dis = setup_training_discriminator(model_dis)
    print("Start pre-training discriminator......")
    #run_test_discriminator(model_dis, dis_batcher, sess_dis, saver_dis, "test")
    if not os.path.exists("discriminator_result"): os.mkdir("discriminator_result")
    run_pre_train_discriminator(model_dis, dis_batcher, 1, sess_dis, saver_dis, train_dir_dis)

    #util.load_ckpt(saver_ge, sess_ge, ckpt_dir="train-generator")
    


if __name__ == '__main__':
  tf.app.run()
