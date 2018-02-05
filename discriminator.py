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

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import data

FLAGS = tf.app.flags.FLAGS



class Discriminator(object):
    """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self._hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps

        # encoder part
        self._target_batch = tf.placeholder(tf.int32, [hps.batch_size* hps.max_enc_sen_num, hps.max_enc_seq_len], name='enc_batch')
        #self._target_lens = tf.placeholder(tf.int32, [hps.batch_size* hps.max_enc_sen_num], name='enc_lens')

        self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len], name='enc_batch')
        self._dec_lens = tf.placeholder(tf.int32, [hps.batch_size * hps.max_enc_sen_num], name='enc_lens')
        #self._enc_sen_lens = tf.placeholder(tf.int32, [hps.batch_size * hps.], name='enc_sen_lens')
        self._target_mask = tf.placeholder(tf.float32,
                                            [hps.batch_size* hps.max_enc_sen_num, hps.max_enc_seq_len],
                                            name='target_mask')
        #self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
        self._decay = tf.placeholder(tf.float32, name="decay_learning_rate")
        self.label = tf.placeholder(tf.float32, [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len], name="positive_negtive")

        #self._target_batch = tf.placeholder(tf.int32,
        #                                    [hps.batch_size* hps.max_enc_sen_num],
        #                                    name='target_batch')


    def _make_feed_dict(self, batch):
        feed_dict = {}
        feed_dict[self._target_batch] = batch.target_batch
        feed_dict[self._dec_batch] = batch.dec_batch
        feed_dict[self._dec_lens] = batch.dec_sen_lens
        feed_dict[self.label] = batch.labels
        #feed_dict[self._enc_sen_lens] = batch.enc_sen_lens
        #feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
        feed_dict[self._target_mask] = batch.dec_padding_mask
        #feed_dict[self.label] = batch.labels
        return feed_dict






    def _build_model(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()  # size of the vocabulary

        with tf.variable_scope('discriminator'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag,
                                                                seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            # Add embedding matrix (shared by the encoder and decoder inputs)
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
                                            initializer=self.trunc_norm_init)


                emb_dec_inputs = tf.nn.embedding_lookup(embedding,
                                                        self._dec_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
                self.emb_enc_inputs = emb_dec_inputs

            ## Add the encoder.
            #encoder_vector = self._add_encoder(emb_enc_inputs, self._enc_lens, hps)


            with tf.variable_scope('output_projection'):
                w = tf.get_variable('w_output', [hps.hidden_dim, vsize], dtype=tf.float32,
                                    initializer=self.trunc_norm_init)
                v = tf.get_variable('v_output', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)

            with tf.variable_scope('decoder'):
                # When decoding, use model output from the previous step
                # for the next step.
                loop_function = None

                cell = tf.contrib.rnn.LSTMCell(
                    hps.hidden_dim,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=False)

                #tf.logging.info(emb_dec_inputs)
                emb_dec_inputs = tf.unstack(emb_dec_inputs, axis=1)
                self._dec_in_state = cell.zero_state(FLAGS.batch_size* hps.max_enc_sen_num, tf.float32)
                # tf.logging.info(self._dec_in_state)
                # tf.logging.info(emb_dec_inputs)
                decoder_outputs, self._dec_out_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                    emb_dec_inputs,self._dec_in_state,
                    cell, loop_function=None
                )
                decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])

            decoder_outputs = tf.reshape(decoder_outputs,
                                         [-1,
                                          hps.hidden_dim])
            decoder_outputs = tf.nn.xw_plus_b(decoder_outputs, w, v)


            decoder_outputs = tf.reshape(decoder_outputs,
                                             [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len,
                                              FLAGS.vocab_size])

            '''crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self._target_batch, logits=decoder_outputs)
            self.out_put = tf.argmax(crossent, 1)
            self.out_put = tf.reshape(self.out_put, [hps.batch_size, hps.max_enc_sen_num])'''
            '''weights = self._target_mask * self.label
            self.train_loss = tf.contrib.seq2seq.sequence_loss(
                decoder_outputs,
                self._target_batch,
                weights,
                average_across_timesteps=True,
                average_across_batch=True)'''
                
                
            weights = self._target_mask * self.label
            self.train_loss = tf.contrib.seq2seq.sequence_loss(
                decoder_outputs,
                self._target_batch,
                weights,
                average_across_timesteps=True,
                average_across_batch=True)
                
                
            self.out_loss = tf.contrib.seq2seq.sequence_loss(
                decoder_outputs,
                self._target_batch,
                self._target_mask,
                average_across_timesteps=False,
                average_across_batch=False)
            self.out_loss=tf.reshape(self.out_loss, [-1])
            #label=tf.reshape(self.label, [-1])
            #self.train_loss = tf.reduce_mean(self.out_loss)/(hps.batch_size*hps.max_enc_sen_num*hps.max_enc_seq_len)
            self.out_loss = tf.reshape(self.out_loss, [hps.batch_size, hps.max_enc_sen_num, hps.max_enc_seq_len])
            self.out_loss_sentence = tf.reduce_mean(self.out_loss,axis = -1)






    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        # Take gradients of the trainable variables w.r.t. the loss function to minimize
        loss_to_minimize =  self.train_loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)

        # Add a summary
        tf.summary.scalar('global_norm', global_norm)

        # Apply adagrad optimizer
        optimizer = tf.train.AdagradOptimizer(self._hps.lr, initial_accumulator_value=self._hps.adagrad_init_acc)

        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def build_graph(self):

        """Add the placeholders, model, global step, train_op and summaries to the graph"""
        with tf.device("/gpu:" + str(FLAGS.gpuid)):
            tf.logging.info('Building graph...')
            t0 = time.time()
            self._add_placeholders()
            self._build_model()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._add_train_op()
            t1 = time.time()
            tf.logging.info('Time to build graph: %i seconds', t1 - t0)

    def run_train_step(self, sess, batch, decay=False):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""

        feed_dict = self._make_feed_dict(batch)
        feed_dict[self._decay] = 1.0
        if decay:
            feed_dict[self._decay] = 0.001

        to_return = {
            'train_op': self._train_op,
            'loss': self.train_loss,
            'out_loss': self.out_loss,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)
    def run_pre_train_step(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        feed_dict[self._decay] = 1.0
        to_return = {
            'train_op': self._train_op,
            'loss': self.train_loss,
            'out_loss': self.out_loss,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)

    def run_ypred_auc(self, sess, batch):
        """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'y_pred_auc': self.out_loss,
            'y_pred_auc_sentence': self.out_loss_sentence
        }

        return sess.run(to_return, feed_dict)

    '''def run_eval_step(self, sess, batch):
        """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
        feed_dict = self._make_feed_dict(batch)
        error_list =[]
        error_label = []
        to_return = {
            'predictions': self.out_put,
        }
        results = sess.run(to_return, feed_dict)
        right =0
        number =0
        output = results['predictions']

        for i in range(len(batch.labels)):
            if batch.target_mask[i] == 1:
                if results['predictions'][i] == batch.labels[i]:
                    right +=1
                else:
                    error_label.append(results['predictions'][i])
                    error_list.append(batch.original_reviews[i])
                number+=1
        print (batch.labels)
        print (batch.target_mask)
        print (results['predictions'])
        print (right)
        print (number)
        print (error_label)
        print (error_list)
        return right, number,error_list,error_label'''
