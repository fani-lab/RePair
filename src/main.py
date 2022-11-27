import argparse
import os
import pandas as pd
import param
from dal.msmarco import msmarco
from eval.msmarco import getHits

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    if ('msmarco' in domain_list):
        datapath = data_list[domain_list.index('msmarco')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        try:
            print('Loading (query,passage) file ...')
            query_doc_pair = pd.read_csv(f'{prep_output}/query-doc.train.tsv', sep='\t', on_bad_lines='warn')
        except (FileNotFoundError, EOFError) as e:
            print('Loading (query,passage) file failed! Pairing queries and relevant passages ...')
            msmarco(datapath, prep_output)
            query_doc_pair = pd.read_csv(f'{prep_output}/query-doc.train.tsv', sep='\t', on_bad_lines='warn')
            print(f"query_doc_pairs first 100\n{query_doc_pair.head()}")
            '''
            This needs to be updated for the new training using T5 tensorflow. 
            '''
        # if 'train' in param.settings['cmd']:
        #     print('Training t5-small on (query, passage) pairs ...')
        #     train(qrels, './../output')
        getHits(f'{output}predictions/msmarco', output)
    if ('aol' in data_list): print('processing aol...')
    if ('yandex' in data_list): print('processing yandex...')


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=[], required=True,
                         help='a list of dataset paths; required; (eg. -data ./../data/raw/msmarco)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str, default=[], required=True,
                         help='a list of dataset paths; required; (eg. -domain msmarco)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/',
                        help='The output path (default: -output ./../output/)')


# python -u main.py -data ../data/raw/toy.msmarco -domain msmarco

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Personalized Query Refinement')
    addargs(parser)
    args = parser.parse_args()

    run(data_list=args.data_list,
        domain_list=args.domain_list,
        output=args.output,
        settings=param.settings)
