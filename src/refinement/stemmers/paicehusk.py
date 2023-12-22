#!/bin/python
from src.refinement.stemmers.abstractstemmer import AbstractStemmer
import src.refinement.stemmers.paicehuskstemmer as paicehuskstemmer
import sys


class PaiceHuskStemmer(AbstractStemmer):

    def __init__(self, ):
        super(PaiceHuskStemmer, self).__init__()
        self.basename = 'paicehusk'

    def process(self, words):
        return [paicehuskstemmer.stem(word) for word in words]


if __name__ == '__main__':
    stemmer = PaiceHuskStemmer()
    stemmer.stem(sys.argv[1:])
