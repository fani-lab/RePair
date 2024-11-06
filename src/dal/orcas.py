from os.path import isfile,join
import json, os, pandas as pd,numpy as np
from tqdm import tqdm
from dal.ds import Dataset
#import evaluate
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu

import seaborn as sns
import pickle
import random
tqdm.pandas()
import csv
from ftfy import fix_text
class MsMarcoDoc(Dataset):

    def __init__(self, settings): super(MsMarcoDoc, self).__init__(settings=settings)

    @classmethod
    def pair(cls, input, output,data_split, ctx, cat=True):
        queries = pd.read_csv(f'{input}/toy.orcas.train.tsv', sep="\t", index_col=False, encoding='utf-8',  usecols=['qid', 'query', 'label', 'data_split'], converters={'query': str.lower})
        queries.drop_duplicates(subset=['qid'], inplace=True)
        if data_split == "test": queries = queries[queries["data_split"] == "test"]
        qrels = pd.read_csv(f'{input}/orcas-doctrain-qrels.tsv', sep=" ",encoding='utf-8', index_col=False, names=['qid', 'fake', 'did', 'relevancy'], header=None)
        qrels.drop_duplicates(subset=['qid', 'did'], inplace=True)  # qrels have duplicates!!
        qrels.to_csv(f'{input}/orcas-doctrain-qrels.tsv_', index=False, sep='\t', header=False,encoding='utf-8')  # trec_eval.9.0.4 does not accept duplicate rows!!
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)
        doccol = 'docs' if cat else 'doc'
        queries_qrels[doccol] = queries_qrels['did'].progress_apply(cls._txt)  # training_set: 100%|██████████| 6880075/6880075 [17:58:30<00:00, 106.32it/s]
        queries_qrels['ctx'] = '' # add context
        if cat: queries_qrels = queries_qrels.groupby(['qid', 'query','label'], as_index=False, observed=True).agg({'did': list, doccol: ' '.join})
        if ctx: queries_qrels[doccol] = queries_qrels['label'] + ": " + queries_qrels[doccol].astype(str) #without ctx
        queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
        batch_size = 55000 #1000000  # need to make this dynamic
        index_item_str = '.'.join(cls.settings['index_item'])
        # create dirs:
        if not os.path.isdir(
            f'../output/orcas/{cls.user_pairing}t5.base.gc.docs.query.{"ctx." if ctx else ""}{index_item_str}/original_queries'): os.makedirs(
            f'../output/orcas/{cls.user_pairing}t5.base.gc.docs.query.{"ctx." if ctx else ""}{index_item_str}/original_queries')
        if not os.path.isdir(
            f'../output/toy.orcas/{cls.user_pairing}t5.base.gc.docs.query.{"ctx." if ctx else ""}{index_item_str}/qrels'): os.makedirs(
            f'../output/orcas/{cls.user_pairing}t5.base.gc.docs.query.{"ctx." if ctx else ""}{index_item_str}/qrels')
        if len(queries_qrels) > batch_size:
            for _, chunk in queries_qrels.groupby(np.arange(queries_qrels.shape[0]) // batch_size):
                chunk.drop_duplicates(subset=['qid'], inplace=True)
                chunk.to_csv(f'../output/orcas/{cls.user_pairing}docs.query.{"ctx." if ctx else ""}{index_item_str}.{data_split}.{_}.tsv',
                             columns=['docs', 'query'], header=False, sep='\t', encoding='utf-8', index=False)
                chunk.to_csv(f'../output/orcas/{cls.user_pairing}t5.base.gc.docs.query.{"ctx." if ctx else ""}{index_item_str}/original_queries/original.{_}.tsv', sep='\t', encoding='utf-8', index=False, columns=['query'], header=False)
                qrels_splits = chunk[['qid', 'query']].merge(qrels, on='qid', how='inner')
                qrels_splits.to_csv(f'../output/orcas/{cls.user_pairing}t5.base.gc.docs.query.{"ctx." if ctx else ""}{index_item_str}/qrels/queries.qrels.docs.{"ctx." if ctx else ""}{data_split}.{_}.tsv',
                    sep='\t',encoding='utf-8', index=False, header=False, columns=['qid', 'query', 'fake', 'did', 'relevancy'])
        return queries_qrels



    @classmethod
    def stat(cls, agg_all, output_path, raw_path, ranker, metric,ctx):
        groups = agg_all.groupby("qid")
        dic_len=dict()
        dic_qt=dict()
        dic_qid=dict()
        dict_blue=dict()
        mean_blue_score=dict()

        #blue score
        # scorer = rouge_scorer.RougeScorer(['rouge1'])
        # for _, group in tqdm(groups, total=len(groups)):#[3:15:07<00:00, 119.32it/s]
        #     if len(group) > 1:
        #         pred_queries= group.iloc[1:len(group)]
        #         matrix = []
        #         for row in range(len(pred_queries)):
        #             a = []
        #             for column in range(len(pred_queries)):
        #                 #a.append(sentence_bleu([str(pred_queries["query"].values[row]).split()],str(pred_queries["query"].values[column]).split() , weights=(1,0,0,0)) )
        #                 a.append(scorer.score(str(pred_queries["query"].values[row]),str(pred_queries["query"].values[column]))["rouge1"][2])
        #                 #a.append(rouge.compute(predictions=[str(pred_queries["query"].values[column])], references=[str(pred_queries["query"].values[row])])["rougeL"])
        #             matrix.append(a)
        #         dict_blue[_]=matrix
        #         mean_blue_score[_]= np.array(matrix).mean()
        # with open(f'{output_path}/rouge1_score.{"ctx." if ctx else ""}pickle', 'wb') as bleu_score:
        #     pickle.dump(dict_blue, bleu_score, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f'{output_path}/rouge1_score_mean.{"ctx." if ctx else ""}pickle', 'wb') as bleu_score_mean:
        #     pickle.dump(mean_blue_score, bleu_score_mean, protocol=pickle.HIGHEST_PROTOCOL)
        # columns=["qid", "mean_rouge",]
        # with open(f'{output_path}/rouge1_score_mean.test.pickle', "rb") as f:
        #     object = pickle.load(f)
        # with open(f'{output_path}/rouge1_score_mean.test.csv', 'w',newline='') as csv_file:
        #     csvwriter = csv.DictWriter(csv_file, fieldnames=columns,delimiter='\t')
        #     csvwriter.writeheader()
        #     for session in object:
        #             csvwriter.writerow({"qid":session, "mean_rouge":object[session]})

        #columns = ["qid","pred_query_len","pred_query_uniq_len","count"]
        #counting stat
        # ref_queries = pd.read_csv(f'{output_path}/bm25.map.agg.gold.tsv', sep="\t", usecols=["qid","order","query","bm25.map"])
        # for _, group in tqdm(groups, total=len(groups)):#[3:15:07<00:00, 119.32it/s]
        #     if len(group) > 1:
        #         pred_query_len = len(group.iloc[1:len(group)]["query"])
        #         pred_query_uniq_len = len(group.iloc[1:len(group)]["query"].unique())
        #         dic_len[group["qid"].values[0]] = {"pred_query_len": pred_query_len, "pred_query_uniq_len": pred_query_uniq_len}
        # with open(f'{output_path}/pred_query_len.{"ctx." if ctx else ""}pickle', 'wb') as max_mean_qid:
        #     pickle.dump(dic_len, max_mean_qid, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f'{output_path}/pred_query_len.pickle', "rb") as f:
        #    object = pickle.load(f)
        # with open(f'{output_path}/pred_query_len.csv', 'w', newline='') as csv_file:
        #     csvwriter = csv.DictWriter(csv_file, fieldnames=columns,delimiter='\t')
        #     csvwriter.writeheader()
        #     for qid in object :
        #         count=0
        #         if object[qid]["pred_query_len"]==object[qid]["pred_query_uniq_len"]:
        #             count=False
        #         else:count=True
        #         csvwriter.writerow({"qid": qid,"pred_query_len":object[qid]["pred_query_len"],"pred_query_uniq_len":object[qid]["pred_query_uniq_len"],"count":count })

        # qt_pickle to scv
        # Factual,Navigational, Abstain, Instrumental, Transactional
        # columns = ["qid", "mean_diff", "max_diff"]
        # with open(f'{output_path}/qid_diff_test_gold.pickle', "rb") as f:
        #     object = pickle.load(f)
        # with open(f'{output_path}/qid_diff_test_gold.csv', 'w', newline='') as csv_file:
        #     csvwriter = csv.DictWriter(csv_file, fieldnames=columns, delimiter='\t')
        #     csvwriter.writeheader()
        #     for i in range(len(object["Transactional"])):
        #         for key in (object["Transactional"][i]):
        #             if object["Transactional"][i][key]["mean_diff"]>0:
        #                csvwriter.writerow({"qt":"Transactional","qid": key, "mean_diff": object["Transactional"][i][key]["mean_diff"],"max_diff": object["Transactional"][i][key]["max_diff"]})

        #Factual,Navigational, Abstain, Instrumental, Transactional
        queries = pd.read_csv(f'{raw_path}/orcas.tsv', sep="\t", index_col=False, encoding='utf-8',
                                usecols=['qid', 'query', 'label', 'data_split'], converters={'query': str.lower})
        queries = queries[queries["data_split"] == "validation"]
        queries.drop_duplicates(subset=['qid'], inplace=True)
        queries = queries[queries["data_split"] == "validation"]
        types = queries["label"].unique()

        for t in types:
            dic_qt.update({t: []})
        for _, group in tqdm(groups, total=len(groups)):#[3:15:07<00:00, 119.32it/s]
            if len(group) > 1:
                max_diff = group.iloc[1:len(group)][f'{ranker}.{metric}'].max() - group.iloc[0][f'{ranker}.{metric}']
                mean_diff = group.iloc[1:len(group)][group[f'{ranker}.{metric}'] > 0][f'{ranker}.{metric}'].mean() - (group.iloc[0][f'{ranker}.{metric}'])
                dic_qt[queries.loc[(queries['qid'] == int(group["qid"].values[0]))]["label"].values[0]].append({group["qid"].values[0]: {"mean_diff": mean_diff,"max_diff": max_diff}})
                #dic_qid[group["qid"].values[0]] = {"mean_diff": mean_diff, "max_diff": max_diff}
        # with open(f'{output_path}/qid_diff_test_gold.{ranker}.{metric}.{"ctx." if ctx else ""}pickle', 'wb') as max_mean_qid:
        #    pickle.dump(dic_qid, max_mean_qid, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{output_path}/qt-stat/qt_diff.{ranker}.{metric}.{"ctx." if ctx else ""}pickle', 'wb') as max_mean_qt:
           pickle.dump(dic_qt, max_mean_qt, protocol=pickle.HIGHEST_PROTOCOL)

        #qid_pickle to scv
        # columns=["qid", "mean_diff", "max_diff" ]
        # with open(f'{output_path}/qid_diff_test_gold.{ranker}.{metric}.{"ctx." if ctx else ""}pickle', "rb") as f:
        #     object = pickle.load(f)
        # with open(f'{output_path}/zero_qid_diff_test_gold.{ranker}.{metric}.{"ctx." if ctx else ""}csv', 'w',newline='') as csv_file:
        #     csvwriter = csv.DictWriter(csv_file, fieldnames=columns,delimiter='\t')
        #     csvwriter.writeheader()
        #     for session in object:
        #             csvwriter.writerow({"qid":session, "mean_diff":object[session]["mean_diff"],"max_diff":object[session]["max_diff"]})

        #qt_pickle toscv
        types=["Factual","Navigational", "Abstain", "Instrumental", "Transactional"]
        for t in types:
            columns = ["qt", "qid", "mean_diff", "max_diff"]
            with open(f'{output_path}/qt-stat/qt_diff.{ranker}.{metric}.{"ctx." if ctx else ""}pickle', "rb") as f:
                object = pickle.load(f)
            with open(f'{output_path}/qt-stat/qt_diff_{t}.{ranker}.{metric}.{"ctx." if ctx else ""}csv', 'w',
                      newline='') as csv_file:
                csvwriter = csv.DictWriter(csv_file, fieldnames=columns, delimiter='\t')
                csvwriter.writeheader()
                for i in range(len(object[f'{t}'])):
                    for key in (object[f'{t}'][i]):
                        if object[f'{t}'][i][key]["mean_diff"] >= 0:
                            csvwriter.writerow({"qt": f'{t}', "qid": key,
                                                "mean_diff": object[f'{t}'][i][key]["mean_diff"],
                                                "max_diff": object[f'{t}'][i][key]["max_diff"]})


        # plt.rcParams["figure.figsize"] = [7.00, 3.50]
        # plt.rcParams["figure.autolayout"] = True
        # Make a list of columns
        # data={
        #     #"map_diff":[-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # }
        # new_def= pd.DataFrame(data)
        # mean_diff_count=[]
        # ctx_mean_diff_count=[]
        # # Read a CSV file
        # df = pd.read_csv(f'{output_path}/qid_mean_max_diff_gold.csv')
        # mean_diff_count.append(df[(df["mean_diff"]<=0)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] >0)&(df["mean_diff"] <= 0.1)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.1) & (df["mean_diff"] <= 0.2)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.2) & (df["mean_diff"] <= 0.3)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.3) & (df["mean_diff"] <= 0.4)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.4) & (df["mean_diff"] <= 0.5)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.5) & (df["mean_diff"] <= 0.6)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.6) & (df["mean_diff"] <= 0.7)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.7) & (df["mean_diff"] <= 0.8)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.8) & (df["mean_diff"] <= 0.9)]["qid"].count())
        # mean_diff_count.append(df[(df["mean_diff"] > 0.9) & (df["mean_diff"] <= 1)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] <= 0)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0) & (df["ctx_mean_diff"] <= 0.1)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.1) & (df["ctx_mean_diff"] <= 0.2)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.2) & (df["ctx_mean_diff"] <= 0.3)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.3) & (df["ctx_mean_diff"] <= 0.4)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.4) & (df["ctx_mean_diff"] <= 0.5)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.5) & (df["ctx_mean_diff"] <= 0.6)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.6) & (df["ctx_mean_diff"] <= 0.7)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.7) & (df["ctx_mean_diff"] <= 0.8)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.8) & (df["ctx_mean_diff"] <= 0.9)]["qid"].count())
        # ctx_mean_diff_count.append(df[(df["ctx_mean_diff"] > 0.9) & (df["ctx_mean_diff"] <= 1)]["qid"].count())
        # data["mean_diff_count"]=mean_diff_count
        #data["ctx_mean_diff_count"]=ctx_mean_diff_count
        #
        # columns = ["ctx_diff_mean"]
        # with open(f'{output_path}/mean_ctx__diff_quantity.csv', 'w', newline='') as csv_file:
        #     csvwriter = csv.DictWriter(csv_file, fieldnames=columns,delimiter='\t')
        #     csvwriter.writeheader()
        #     for diff in data["ctx_mean_diff_count"]:
        #             csvwriter.writerow({"ctx_diff_mean": diff})
        #Plot the lines
        # df.plot()
        # plt.show()


