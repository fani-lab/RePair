import sys, subprocess, os

def lucenex(corpus, output, ncore):
    """
    common code to create index using the subprocess module
    :return: a lucene index of dataset
    """
    print(f'Lucenex (indexing) for {corpus} at {output} ...')
    if not os.path.isdir(output): os.makedirs(output)
    subprocess.run(['python', '-m',
                    'pyserini.index.lucene', '--collection', 'JsonCollection',
                    '--input', corpus,
                    '--index', output,
                    '--generator', 'DefaultLuceneDocumentGenerator',
                    '--threads', str(ncore), '--storePositions', '--storeDocvectors', '--storeRaw', '--optimize'])
    print(f'Finished creating index.')
