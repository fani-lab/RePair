import argparse, os, pandas as pd
from multiprocessing import freeze_support

import param
from dal.msmarco import *
#from eval.msmarco import getHits

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    if ('msmarco' in domain_list):
        datapath = data_list[domain_list.index('msmarco')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        try:
            print('Loading (query,passage) file ...')
            query_qrel_doc = pd.read_csv(f'{prep_output}/queries.qrels.doc.ctx.train.tsv', sep='\t')
        except (FileNotFoundError, EOFError) as e:
            print('Loading (query,passage) file failed! Pairing queries and relevant passages ...')
            query_qrel_doc = to_pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.train.tsv')
            if settings['psgtxt'] == 'cancat':
                prep_output += '/' + settings['psgtxt']
                pass #concatenate rows with same qid
            for i in settings['msmarco']['pairing']:
                #https://github.com/google-research/text-to-text-transfer-transformer#textlinetask
                query_qrel_doc[i[1]] = query_qrel_doc[i[0]] + ': ' + query_qrel_doc[i[1]]
                query_qrel_doc.to_csv(f'{prep_output}/{".".join(i)}.train.tsv', sep='\t', encoding='utf-8', index=False, columns=i[1:])

            '''
            This needs to be updated for the new training using T5 tensorflow. 
            '''
        #getHits(f'{output}predictions/{os.path.split(datapath)[-1]}', output, os.path.split(datapath)[-1])
    if ('aol' in data_list): print('processing aol...')
    if ('yandex' in data_list): print('processing yandex...')


def addargs(parser):
    dataset = parser.add_argument_group('dataset')
    dataset.add_argument('-data', '--data-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -data ./../data/raw/msmarco)')
    dataset.add_argument('-domain', '--domain-list', nargs='+', type=str, default=[], required=True, help='a list of dataset paths; required; (eg. -domain msmarco)')

    output = parser.add_argument_group('output')
    output.add_argument('-output', type=str, default='./../output/', help='The output path (default: -output ./../output/)')


# python -u main.py -data ../data/raw/toy.msmarco -domain msmarco

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='Personalized Query Refinement')
    addargs(parser)
    args = parser.parse_args()

    run(data_list=args.data_list,
        domain_list=args.domain_list,
        output=args.output,
        settings=param.settings)
