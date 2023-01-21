import sys
import subprocess
import os

def create_index(index_ds, index_item='title_and_text'):
    """
    common code to create index using the subprocess module
    :return: a lucene index of dataset
    """
    if len(os.listdir(f'./../data/raw/{index_ds}/indexes/lucene-index-{index_ds}-{index_item}')) == 0:
        print(f'Creating index:{index_ds}\n')
        subprocess.run(['python', '-m',
                        'pyserini.index.lucene', '--collection', 'JsonCollection',
                        '--input', f'./../data/raw/{index_ds}/{index_item}',
                        '--index', f'./../data/raw/{index_ds}/indexes/lucene-index-{index_ds}-{index_item}',
                        '--generator', 'DefaultLuceneDocumentGenerator',
                        '--threads', '8', '--storePositions', '--storeDocvectors', '--storeRaw', '--optimize'])
        print(f'finished creating index')
