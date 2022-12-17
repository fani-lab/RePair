import argparse, os, pandas as pd
from multiprocessing import freeze_support

import param
from dal.msmarco import *
#from eval.msmarco import getHits

from mdl import mt5w

def run(data_list, domain_list, output, settings):
    # 'qrels.train.tsv' => ,["qid","did","pid","relevancy"]
    # 'queries.train.tsv' => ["qid","query"]

    if ('msmarco-passage' in domain_list):
        datapath = data_list[domain_list.index('msmarco-passage')]
        prep_output = f'./../data/preprocessed/{os.path.split(datapath)[-1]}'
        if not os.path.isdir(prep_output): os.makedirs(prep_output)
        in_type = settings['msmarco-passage']['pairing'][1]
        out_type = settings['msmarco-passage']['pairing'][2]
        tsv_path = {'train': f'{prep_output}/{in_type}.{out_type}.train.tsv', 'test': f'{prep_output}/{in_type}.{out_type}.test.tsv'}

        query_qrel_doc = None
        if any(not os.path.exists(v) for k, v in tsv_path.items()):
            print('Pairing queries and relevant passages ...')
            query_qrel_doc = to_pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.train.tsv')
            #TODO: query_qrel_doc = to_pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.test.tsv')
            query_qrel_doc = to_pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.test.tsv')
            if settings['concat']:
                prep_output += '/concat'
                pass #concatenate rows with same qid
            query_qrel_doc.to_csv(tsv_path['train'], sep='\t', encoding='utf-8', index=False, columns=[in_type, out_type], header=False)
            query_qrel_doc.to_csv(tsv_path['test'], sep='\t', encoding='utf-8', index=False, columns=[in_type, out_type], header=False)

        t5_model = 'small'  # "gs://t5-data/pretrained_models/{"small", "base", "large", "3B", "11B"}
        output = f'../output/t5.{t5_model}.local.{in_type}.{out_type}'
        if 'finetune' in settings['cmd']:
            mt5w.finetune(
                tsv_path=tsv_path,
                pretrained_dir=f'./../output/t5-data/pretrained_models/{t5_model}',
                steps=5,
                output=output, task_name='msmarco_passage_cf',
                lseq={"inputs": 32, "targets": 256},  #query length and doc length
                nexamples=query_qrel_doc.shape[0] if query_qrel_doc is not None else None, in_type=in_type, out_type=out_type, gcloud=False)

        if 'predict' in settings['cmd']:
            mt5w.predict(
                iter=5,
                split='test',
                tsv_path=tsv_path,
                output=output,
                lseq={"inputs": 32, "targets": 256}, gcloud=False)

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
