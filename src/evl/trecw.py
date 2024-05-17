import os
import evaluate as eval
from scipy.spatial.distance import cosine
import pandas as pd


'''
Calculates the difference between the original and back-translated query
'''
def semsim_compute(predictions, references, transformer_model):
    me, you = transformer_model.encode([predictions, references])
    return {'semsim': 1 - cosine(me, you)}


def compare_query_similarity(refined_q_file, output, transformer_model):
    refined_list = (pd.read_csv(refined_q_file, sep='\t', header=None, names=['id', 'original', 'refined']).dropna()).to_records(index=False)
    similarity_results = pd.DataFrame()
    bleu = eval.load("bleu")
    rouge = eval.load('rouge')
    for (id, original, refined) in refined_list:
        rouge_results = rouge.compute(predictions=[refined], references=[original])
        bleu_results = bleu.compute(predictions=[refined], references=[original])
        semsim = semsim_compute(predictions=refined, references=original, transformer_model=transformer_model)
        new_row_df = pd.DataFrame([{'id':id, 'original':original, 'refined': refined, **rouge_results, **bleu_results, **semsim}])
        similarity_results = pd.concat([similarity_results, new_row_df], ignore_index=True)

    similarity_results.to_csv(output, index=False)


def evaluate(in_docids, out_metrics, qrels, metric, lib, mean=False, topk=10):
    # qrels can have queries that are not in in_docids (superset)
    # also prediction may have queries that are not known to qrels
    # with open('pred', 'w') as f:
    #   f.write(f'1\tQ0\t2\t1\t20.30781\tPyserini Batch\n')
    #   f.write(f'1\tQ0\t3\t1\t5.30781\tPyserini Batch\n')
    #   f.write(f'5\tQ0\t3\t1\t5.30781\tPyserini Batch\n')#does not exist in qrel
    # with open('qrel', 'w') as f:
    #   f.write(f'1\t0\t2\t1\n')
    #   f.write(f'3\t0\t3\t1\n')#does not exist in prediction
    # "./../trec_eval.9.0.4/trec_eval" -q -m ndcg qrel pred
    # ndcg                    1       1.0000
    # ndcg                    all     1.0000
    #
    # However, no duplicate [qid, docid] can be in qrels!!

    print(f'Evaluating retrieved docs for {in_docids} ...')
    if 'trec_eval' in lib:
        cli_cmd = f'{lib} {"-n" if not mean else ""} -q -m {metric} {qrels} {in_docids} > {out_metrics}'
        print(cli_cmd)
        stream = os.popen(cli_cmd)
        print(stream.read())
    else: raise NotImplementedError
