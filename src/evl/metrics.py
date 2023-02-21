import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
datasets = ['diamond', 'platinum', 'gold']
t5_refinement = '../../output/msmarco.passage/t5.base.gc.docs.query/transfer.refinement'

for ds in datasets:
    queries_dev = pd.read_csv(f'../../output/msmarco.passage/t5.base.gc.docs.query/bm25.map.datasets/{ds}_test.tsv', sep='\t', names=['query', 'query_'], usecols=['query_'])
    queries_pred = pd.read_csv(f'{t5_refinement}/{ds}.test.pred-1004100', skip_blank_lines=False, sep='\r\r', names=['query'])
    queries_dev['pred_query'] = queries_pred['query']
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_result = list()
    bleu_scores = list()
    for i, row in tqdm(queries_dev.iterrows(), total=queries_dev.shape[0]):
        scores = scorer.score(str(row.query_), str(row.pred_query))
        rouge_result.append(scores["rougeL"].fmeasure)
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(references=[(str(row.query_)).split(' ')], hypothesis=(str(row.pred_query)).split(' '), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7)
        bleu_scores.append(BLEUscore)
    rouge_mean = np.mean(rouge_result)
    bleu_mean = np.mean(bleu_scores)
    print(f'rouge score for {ds}: {rouge_mean}')
    print(f'BLEU score {ds}: {bleu_mean}')
    print(f'f1 measure {ds}: {2 * ((bleu_mean * rouge_mean) / (bleu_mean + rouge_mean))}')
# DS TEST SIZE    ROUGE   BLEU    F1 ###TRANSFER REFINEMENT with target_query
# DIAMOND  35384  41.86   20.61   27.69
# PLATINUM  82867   41.80 20.54 27.55
# GOLD  94388  42.56   21.15   28.26

# DS TEST SIZE    ROUGE   BLEU    F1 ###BASE REFINEMENT with target_query
# DIAMOND  35384  42.56   21.02   28.15
# PLATINUM  82867   42.96 21.15 28.35
# GOLD  94388   43.56   21.67   28.95







#bleu scores
# msmarco.pred 32.08
# msmarco.paraphrase.pred 39.31
# aol.title.pred.msmarco 9.59
# aol.title.url.pred.msmarco 15.30
# aol.title.pred 18.31
# aol.title.url.pred 28.44
# msmarco.pred.aol.title 19.03
# msmarco.pred.aol.title.url 19.67


# rouge scores
# msmarco.pred 54.75
# msmarco.paraphrase.pred 60.35
# aol.title.pred.msmarco 23.87
# aol.title.url.pred.msmarco 32.49
# aol.title.pred 34.24
# aol.title.url.pred 47.47
# msmarco.pred.aol.title 42.54
# msmarco.pred.aol.title.url 43.06

# f1 measures
# msmarco.pred 40.45
# msmarco.paraphrase.pred 47.61
# aol.title.pred.msmarco 13.69
# aol.title.url.pred.msmarco 20.80
# aol.title.pred 23.86
# aol.title.url.pred 35.57
# msmarco.pred.aol.title 26.30
# msmarco.pred.aol.title.url 27.00

