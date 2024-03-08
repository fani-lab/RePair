#!/bin/python
from src.refinement.stemmers.abstractstemmer import AbstractStemmer
from src.refinement.stemmers import porterstemmer
import sys


class PorterStemmer(AbstractStemmer):

    def __init__(self, ):
        super(PorterStemmer, self).__init__()
        self.basename = 'porter'

    def process(self, words):
        return [porterstemmer.stem(word) for word in words]


if __name__ == '__main__':
    stemmer = PorterStemmer()
    # stemmer.stem(sys.argv[1:])
