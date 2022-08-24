import json
import csv
import os
from os.path import exists

from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2-passage')
