'''Reads the Cornell movie dialog dataset and creates training and test datasets
'''

import numpy as np
import os
import sys
from string import punctuation
from six.moves import urllib
import tarfile
import shutil
import zipfile
from tensorflow.models.rnn.translate import data_utils
import dataset


DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "cornell")
DATA_URL = "http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

class CornellMovieData(dataset.Dataset):

    @staticmethod
    def get_line_pairs(data_dir):
        lines_by_number = {}
        with open(os.path.join(data_dir, 'movie_lines.txt')) as lines:
            for line in lines:
                lines_by_number[line.split()[0]] = line.split('+++$+++ ')[-1]


        with open(os.path.join(data_dir, 'movie_conversations.txt')) as conversations:

            source = open(os.path.join(data_dir, 'source.txt'), 'w')
            target = open(os.path.join(data_dir, 'target.txt'), 'w')

            line_pairs = []

            def strip_punctuation(s):
                return ''.join(c for c in s if c not in punctuation)

            for conversation in conversations:

                # Get the line nums (between [ and ]) and split by commma
                conv_lines = conversation.split('[')[1].split(']')[0].split(',')

                # Strip quote marks
                conv_lines = [strip_punctuation(lines) for lines in conv_lines]
                conv_lines = [lines.strip() for lines in conv_lines]

                for i in range(0, len(conv_lines) - 1):
                    if conv_lines[i] in lines_by_number and conv_lines[i + 1] in lines_by_number:
                        source.write(lines_by_number[conv_lines[i]])
                        target.write(lines_by_number[conv_lines[i + 1]])
                    if conv_lines[i] not in lines_by_number:
                        print("Could not find " + conv_lines[i] + "in movie lines")
                    if conv_lines[i + 1] not in lines_by_number:
                        print("Could not find " + conv_lines[i + 1] + "in movie lines")

            source.close()
            target.close()

    @staticmethod
    def maybe_download_and_extract(dest_directory):
      """Download and extract the tarball from the Cornell website"""

      if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
      filename = DATA_URL.split('/')[-1]
      filepath = os.path.join(dest_directory, filename)
      if not os.path.exists(filepath):
        sys.stdout.write('\r>> Downloading to ' + dest_directory)
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        zipfile.ZipFile(filepath, 'r').extractall(dest_directory)
        # Now move them out of that folder for ease of access
        files = os.listdir(os.path.join(dest_directory, 'cornell movie-dialogs corpus'))
        for f in files:
            if f.endswith('.txt'):
                full_file = os.path.join(dest_directory, 'cornell movie-dialogs corpus', f)
                print("Moving " + f + " to " + dest_directory)
                shutil.move(full_file, dest_directory)

    @staticmethod
    def prepare_data(vocabulary_size, data_dir=DEFAULT_DATA_DIR, tokenizer=None):
      """Get cornell data into data_dir, create vocabularies and tokenize data.
      Args:
        vocabulary_size: size of the English vocabulary to create and use.
        data_dir: where to store or look for the data
        tokenizer: a function to use to tokenize each data sentence;
          if None, basic_tokenizer will be used.
      Returns:
        A tuple of 6 elements:
          (1) path to the token-ids for source training data-set,
          (2) path to the token-ids for target training data-set,
          (3) path to the source vocabulary file,
          (4) path to the target vocabulary file
      """
      # Get data to the specified directory.
      CornellMovieData.maybe_download_and_extract(data_dir)

      # Parse into sentence pairs
      CornellMovieData.get_line_pairs(data_dir)

      # Create vocabularies of the appropriate sizes.
      source_vocab_path = os.path.join(data_dir, "vocab%d.source" % vocabulary_size)
      target_vocab_path = os.path.join(data_dir, "vocab%d.target" % vocabulary_size)
      data_utils.create_vocabulary(source_vocab_path, os.path.join(data_dir, 'source.txt'), vocabulary_size, tokenizer)
      data_utils.create_vocabulary(target_vocab_path, os.path.join(data_dir, 'target.txt'), vocabulary_size, tokenizer)


      # Create token ids for the training data.
      source_train_ids_path = os.path.join(data_dir, ("ids%d.source" % vocabulary_size))
      target_train_ids_path = os.path.join(data_dir, ("ids%d.target" % vocabulary_size))
      data_utils.data_to_token_ids(os.path.join(data_dir, 'source.txt'), source_train_ids_path, source_vocab_path, tokenizer)
      data_utils.data_to_token_ids(os.path.join(data_dir, 'target.txt'), target_train_ids_path, target_vocab_path, tokenizer)

      return (source_train_ids_path, target_train_ids_path,
              source_vocab_path, target_vocab_path)