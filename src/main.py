import argparse

import param
from dal.msmarco import msmarco
from mdl.t5 import *

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    if('msmarco' in domain_list):
        datapath = data_list[domain_list.index('msmarco')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        try:
            qrels = pd.read_csv(f'{prep_output}/qrels.target.tsv', sep='\t')
            queries = pd.read_csv(f'{prep_output}/queries.target.tsv', sep='\t')
        except (FileNotFoundError, EOFError) as e:
            msmarco(datapath, prep_output)
            qrels = pd.read_csv(f'{prep_output}/qrels.target.tsv', sep='\t')
            queries = pd.read_csv(f'{prep_output}/queries.target.tsv', sep='\t')
        qrels["query"] = queries["query"]
        train(qrels)

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
    parser = argparse.ArgumentParser(description='Personalized Query Refinement')
    addargs(parser)
    args = parser.parse_args()

    run(data_list = args.data_list,
        domain_list = args.domain_list,
        output = args.output,
        settings = param.settings)


