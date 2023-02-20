import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
t5_refinement = '../../output/t5-refinement'
queries_dev = pd.read_csv('../../data/preprocessed/msmarco.passage/queries.dev.small.tsv', sep='\t', names=['qid', 'query'])
queries_pred = pd.read_csv(f'{t5_refinement}/aol.title.url.pred.msmarco-1004000', skip_blank_lines=False, sep='\r\r', names=['query'])
queries_dev['pred_query'] = queries_pred['query']
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_result = list()
bleu_scores = list()
for i, row in tqdm(queries_dev.iterrows(), total=queries_dev.shape[0]):
    scores = scorer.score(str(row.query), str(row.pred_query))
    rouge_result.append(scores["rougeL"].fmeasure)
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([(str(row.query)).split(' ')], (str(row.pred_query)).split(' '), weights=(0.5, 0.5, 0, 0))
    bleu_scores.append(BLEUscore)
rouge_mean = np.mean(rouge_result)
bleu_mean = np.mean(bleu_scores)
print(f'rouge score : {rouge_mean}')
print(f'BLEU score : {bleu_mean}')
print(f'f1 measure: {2 * (bleu_mean * rouge_mean) / (bleu_mean + rouge_mean)}')


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

