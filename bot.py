from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import os
import random
import sys
import time
import logging

class Bot(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, mode='train'):
        if mode == 'converse':
            self.init_for_conversation()
        elif mode == 'train':
            self.init_and_train()
        else:
            raise Exception('Incorrect mode string')

    @abc.abstractmethod
    def init_for_conversation(self, model_path, model_args):
        return

    @abc.abstractmethod
    def init_and_train(self, train_data, model_args):
        return

    @abc.abstractmethod
    def get_response(self, query):
        return

    def converse(self):
        sys.stdout.write("---")
        sys.stdout.flush()
        sys.stdout.write("---")
        sys.stdout.flush()
        sys.stdout.write("Hello, I'm Lexi. Let's talk.")
        sys.stdout.flush()
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            print(self.get_response(sentence))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

