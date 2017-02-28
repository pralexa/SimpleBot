'''A simple wrapper for the Seq2Seq Tensorflow example model.
main() will train this model on the cornell movie dataset.

Hacked together from the Tensorflow french-english translation example.

Train a model with
    python seq2seq_bot.py

Converse with previously trained bot with
    python seq2seq_bot.py --converse=True
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import bot
import cornell_movie_dialog

from tensorflow.models.rnn.translate import data_utils
from tensorflow.models.rnn.translate import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("temperature", 1.0,
                          "Temperature for sampling outputs for get_response.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "tmp/bot_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "tmp/bot_train", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_string("converse", False,
                            "Reload a previously trained model and converse immediately.")
tf.app.flags.DEFINE_string("dataset", 'cornell',
                           "String name of dataset to use")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


class Seq2Seq(bot.Bot):

    def init_for_conversation(self):
        self.sess = tf.Session()

        # Create model and load parameters.
        self.model = self.create_model(self.sess, True)
        self.model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        self.source_vocab_path = os.path.join(FLAGS.data_dir,
                                         "vocab%d.source" % FLAGS.vocab_size)
        self.target_vocab_path = os.path.join(FLAGS.data_dir,
                                         "vocab%d.target" % FLAGS.vocab_size)
        self.source_vocab, _ = data_utils.initialize_vocabulary(self.source_vocab_path)
        _, self.rev_target_vocab = data_utils.initialize_vocabulary(self.target_vocab_path)


    def init_and_train(self):
        self.train()
        print("Training completed!")
        print("Reloading parameters and starting conversation mode.")
        self.init_for_conversation()

    def get_response(self, sentence):
        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), self.source_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        else:
            logging.warning("Sentence truncated: %s", sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
            {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        output_with_temp = [logit * FLAGS.temperature for logit in output_logits]
        output_softmax = [np.exp(logit) / np.sum(np.exp(logit), axis=1) for logit in output_with_temp]
        top_5 = [np.argsort(output_softmax)[-5:] for logit in output_softmax]
        print(top_5)
        outputs = [np.random.choice(FLAGS.vocab_size, 1, p=logits)[0] for logits in output_with_temp]

        # outputs = [int(np.argmax(logit, axis=1)) for logit in output_softmax]
        # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Print out sentence corresponding to outputs.
        return " ".join([tf.compat.as_str(self.rev_target_vocab[output]) for output in outputs])


    def read_data(self, source_path, target_path, max_size=None):
        """Read data from source and target files and put into buckets.
        Args:
          source_path: path to the files with token-ids for the source language.
          target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
          max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).
        Returns:
          data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
        """
        data_set = [[] for _ in _buckets]
        with tf.gfile.GFile(source_path, mode="r") as source_file:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                counter = 0
                while source and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(_buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids])
                            break
                    source, target = source_file.readline(), target_file.readline()
        return data_set

    def create_model(self, session, forward_only):
        """Create translation model and initialize or load parameters in session."""
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        model = seq2seq_model.Seq2SeqModel(
            FLAGS.vocab_size,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            forward_only=forward_only,
            dtype=dtype)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            init_op = tf.initialize_all_variables()
            session.run(init_op)
        return model

    def train(self):
        """Train a model using movie dialog data."""
        # Prepare data.
        print("Preparing data in %s" % FLAGS.data_dir)
        if FLAGS.dataset == 'cornell':
            self.dataset = cornell_movie_dialog.CornellMovieData
        else:
            raise NotImplementedError("Dataset " + FLAGS.dataset + " has not been implemented.")

        source_train, target_train, _, _ = self.dataset.prepare_data(
            FLAGS.vocab_size, FLAGS.data_dir)
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)

        with tf.Session() as sess:
            # Create model.
            print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = self.create_model(sess, False)

            # Read data into buckets and compute their sizes.
            print("Reading training data (limit: %d)."
                  % FLAGS.max_train_data_size)
            train_set = self.read_data(source_train, target_train, FLAGS.max_train_data_size)
            train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
            train_total_size = float(sum(train_bucket_sizes))

            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in xrange(len(train_bucket_sizes))]

            # This is the training loop.
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            while True:
                # Choose a bucket according to data distribution. We pick a random number
                # in [0, 1] and use the corresponding interval in train_buckets_scale.
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in xrange(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])

                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    train_set, bucket_id)
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()


def main(_):
    if FLAGS.converse:
        bot = Seq2Seq(mode='converse')
    else:
        bot = Seq2Seq(mode='train')
    bot.converse()


if __name__ == "__main__":
    tf.app.run()
