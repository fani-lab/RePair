import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
import re
import string
from collections import Counter
datasets = ['diamond', 'platinum', 'gold']
t5_refinement = '../../output/aol-ia/t5.base.gc.docs.query.title.url/base.refinement'

# compute f1 score

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) == 0 else 0.0
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


for ds in datasets:
    queries_dev = pd.read_csv(
        f'../../output/aol-ia/t5.base.gc.docs.query.title.url/bm25.map.datasets/{ds}.test.tsv', sep='\t', names=['query', 'query_'], usecols=['query_'])
    queries_pred = pd.read_csv(f'{t5_refinement}/{ds}.test.pred-1000100', skip_blank_lines=False, sep='\r\r', names=['query'])
    queries_dev['pred_query'] = queries_pred['query']
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_result = list()
    bleu_scores = list()
    f1_scores = list()
    for i, row in tqdm(queries_dev.iterrows(), total=queries_dev.shape[0]):
        scores = scorer.score(str(row.query_), str(row.pred_query))
        rouge_result.append(scores["rougeL"].fmeasure)
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(references=[(str(row.query_)).split(' ')], hypothesis=(str(row.pred_query)).split(' '), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method7)
        bleu_scores.append(BLEUscore)
        f1_scores.append(f1_score(str(row.query_), str(row.pred_query)))
    rouge_mean = np.mean(rouge_result)
    bleu_mean = np.mean(bleu_scores)
    f1_mean = np.mean(f1_scores)
    print(f'rouge score for {ds}: {rouge_mean}')
    print(f'BLEU score {ds}: {bleu_mean}')
    print(f'f1 measure {ds}: {f1_mean}')

# msmarco base refinement
# rouge score for diamond: 0.42568057136051424
# BLEU score diamond: 0.21029683850476916
# f1 measure diamond: 0.416822260588614

# rouge score for platinum: 0.42969398167649014
# BLEU score platinum: 0.2115697317809715
# f1 measure platinum: 0.41915060043151015

# rouge score for gold: 0.4356421447877607
# BLEU score gold: 0.21678563445709004
# f1 measure gold: 0.42532275952238546

# msmarco transfer refinement
# rouge score for diamond: 0.4186038646735186
# BLEU score diamond: 0.2061915039658764
# f1 measure diamond: 0.41743805966063974

# rouge score for platinum: 0.4180435950210718
# BLEU score platinum: 0.20547429381774826
# f1 measure platinum: 0.41576693292575345

# rouge score for gold: 0.4256894215931806
# BLEU score gold: 0.21156194047222282
# f1 measure gold: 0.42353650076526406



# AOL title transfer
# rouge score for diamond: 0.24785768028983485
# BLEU score diamond: 0.1057627554917162
# f1 measure diamond: 0.23602467853502

# rouge score for platinum: 0.2148899068052302
# BLEU score platinum: 0.08699215515241193
# f1 measure platinum: 0.2011023023056243


# AOL title Base
# rouge score for diamond: 0.17408527383170763
# BLEU score diamond: 0.07404099482985602
# f1 measure diamond: 0.16509779706595218

# rouge score for platinum: 0.1555684917625179
# BLEU score platinum: 0.06250855995300625
# f1 measure platinum: 0.14425863392143193

# rouge score for gold: 0.17282032172417106
# BLEU score gold: 0.07026203522067144
# f1 measure gold: 0.16097593193386475


# AOL title URL base
# rouge score for diamond: 0.25625260529488425
# BLEU score diamond: 0.1079382534269401
# f1 measure diamond: 0.24463295875478222

# rouge score for platinum: 0.2213520954721022
# BLEU score platinum: 0.08731537824992998
# f1 measure platinum: 0.20731018675681637

# rouge score  gold: 0.2493297981807377
# BLEU score gold: 0.0997131509862043
# f1 measure gold: 0.23478355839064657

# AOL TITLE URL Transfer

# rouge score diamond: 0.25625260529488425
# BLEU score diamond: 0.1079382534269401
# f1 measure diamond: 0.24463295875478222

# rouge score platinum: 0.2213520954721022
# BLEU score platinum: 0.08731537824992998
# f1 measure platinum: 0.20731018675681637

# rouge score  gold: 0.2493297981807377
# BLEU score gold: 0.0997131509862043
# f1 measure gold: 0.23478355839064657
