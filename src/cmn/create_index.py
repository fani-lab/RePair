import subprocess
def create_index(index_ds, index_item=None):
    """
    common code to create index using the subprocess module
    :return: a lucene index of aol
    """
    print(f'Creating index:{index_ds}\n')
    subprocess.run(['python', '-m',
                    'pyserini.index.lucene', '--collection', 'JsonCollection',
                    '--input', f'./../data/raw/{index_ds}/{index_item}',
                    '--index', f'./../data/raw/{index_ds}/indexes/lucene-index-{index_ds}{"-" + index_item if index_item is not None else ""}',
                    '--generator', 'DefaultLuceneDocumentGenerator',
                    '--threads', '8', '--storePositions', '--storeDocvectors', '--storeRaw'])
    print(f'finished creating index')
