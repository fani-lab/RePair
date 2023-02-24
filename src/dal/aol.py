import json, os, pandas as pd, shutil
from os.path import isfile, join
from tqdm import tqdm
tqdm.pandas()

from pyserini.search.lucene import LuceneSearcher

import param
from dal.ds import Dataset

class Aol(Dataset):

    @staticmethod
    def init(homedir, index_item, indexdir, ncore):
        # AOL requires us to construct the Index, Qrels and Queries file from IR_dataset
        try:
            Dataset.searcher = LuceneSearcher(f"{param.settings['aol']['index']}/{'.'.join(param.settings['aol']['index_item'])}")
        except:
            print(f"No index found at {param.settings['aol']['index']}! Creating index from scratch using ir-dataset ...")
            #https://github.com/allenai/ir_datasets
            os.environ['IR_DATASETS_HOME'] = homedir
            if not os.path.isdir(os.environ['IR_DATASETS_HOME']): os.makedirs(os.environ['IR_DATASETS_HOME'])
            import ir_datasets
            from cmn.lucenex import lucenex

            print('Setting up aol corpos using ir-datasets ...')
            aolia = ir_datasets.load("aol-ia")
            print('Getting queries and qrels ...')
            # the column order in the file is [qid, uid, did, uid]!!!! STUPID!!
            qrels = pd.DataFrame.from_records(aolia.qrels_iter(), columns=['qid', 'did', 'rel', 'uid'], nrows=1)  # namedtuple<query_id, doc_id, relevance, iteration>
            queries = pd.DataFrame.from_records(aolia.queries_iter(), columns=['qid', 'query'], nrows=1)# namedtuple<query_id, text>

            print('Creating jsonl collections for indexing ...')
            print(f'Raw documents should be downloaded already at {homedir}/aol-ia/downloaded_docs/ as explained here: https://github.com/terrierteam/aolia-tools')
            index_item_str = '.'.join(index_item)
            Aol.create_jsonl(aolia, index_item, f'{homedir}/aol-ia/{index_item_str}')
            lucenex(f'{homedir}/aol-ia/{index_item_str}/', f'{indexdir}/{index_item_str}/', ncore)
            # do NOT rename qrel to qrel.tsv or anything else as aol-ia hardcoded it
            # if os.path.isfile('./../data/raw/aol-ia/qrels'): os.rename('./../data/raw/aol-ia/qrels', '../data/raw/aol-ia/qrels')
            Dataset.searcher = LuceneSearcher(f"{param.settings['aol']['index']}/{'.'.join(param.settings['aol']['index_item'])}")
            if not Dataset.searcher: raise ValueError(f"Lucene searcher cannot find aol index at {param.settings['aol']['index']}/{'.'.join(param.settings['aol']['index_item'])}!")
            # dangerous cleaning!
            # for d in os.listdir('./../data/raw/'):
            #     if not (d.find('aol-ia') > -1 or d.find('msmarco') > -1) and os.path.isdir(f'./../data/raw/{d}'): shutil.rmtree(f'./../data/raw/{d}')

    @staticmethod
    def create_jsonl(aolia, index_item, output):
        """
        https://github.com/castorini/anserini-tools/blob/7b84f773225b5973b4533dfa0aa18653409a6146/scripts/msmarco/convert_collection_to_jsonl.py
        :param index_item: defaults to title_and_text, use the params to create specified index
        :param output: folder name to create docs
        :return: list of docids that have empty body based on the index_item
        """
        print(f'Converting aol docs into jsonl collection for {index_item}')
        if not os.path.isdir(output): os.makedirs(output)
        output_jsonl_file = open(f'{output}/docs.json', 'w', encoding='utf-8', newline='\n')
        for i, doc in enumerate(aolia.docs_iter()):  # doc returns doc_id, title, text, url, ia_url
            did = doc.doc_id
            doc = {'title': doc.title, 'url': doc.url, 'text': doc.text}
            doc = ' '.join([doc[item] for item in index_item])
            output_jsonl_file.write(json.dumps({'id': did, 'contents': doc}) + '\n')
            if i % 100000 == 0: print(f'Converted {i:,} docs, writing into file {output_jsonl_file.name} ...')
        output_jsonl_file.close()

    @staticmethod
    def pair(input, output, cat=True):
        queries = pd.read_csv(f'{input}/queries.tsv', sep='\t', index_col=False, names=['qid', 'query'], converters={'query': str.lower}, header=None)
        # the column order in the file is [qid, uid, did, uid]!!!! STUPID!!
        qrels = pd.read_csv(f'{input}/qrels', encoding='UTF-8', sep='\t', index_col=False, names=['qid', 'uid', 'did', 'rel'], header=None)
        #not considering uid
        # docid is a hash of the URL. qid is the a hash of the *noramlised query* ==> two uid may have same qid then, same docid.
        queries_qrels = pd.merge(queries, qrels, on='qid', how='inner', copy=False)

        doccol = 'docs' if cat else 'doc'
        del queries, qrels
        queries_qrels['ctx'] = ''
        queries_qrels = queries_qrels.astype('category')
        queries_qrels[doccol] = queries_qrels['did'].progress_apply(Aol.to_txt)

        # no uid for now + some cleansings ...
        queries_qrels.drop_duplicates(subset=['qid', 'did'], inplace=True)  # two users with same click for same query
        queries_qrels['uid'] = -1
        queries_qrels.dropna(inplace=True) #empty doctxt, query, ...
        queries_qrels.drop(queries_qrels[queries_qrels['query'].str.strip().str.len() <= param.settings['aol']['filter']['minql']].index,inplace=True)
        queries_qrels.drop(queries_qrels[queries_qrels[doccol].str.strip().str.len() < param.settings["aol"]["filter"]['mindocl']].index,inplace=True)  # remove qrels whose docs are less than mindocl
        queries_qrels.to_csv(f'{input}/qrels.tsv_', index=False, sep='\t', header=False, columns=['qid', 'uid', 'did', 'rel'])

        if cat: queries_qrels = queries_qrels.groupby(['qid', 'query'], as_index=False, observed=True).agg({'uid': list, 'did': list, doccol: ' '.join})
        queries_qrels.to_csv(output, sep='\t', encoding='utf-8', index=False)
        return queries_qrels

    @staticmethod
    def search_df(queries, out_docids, qids, index_item,ranker='bm25', topk=100, batch=None):
        if ranker == 'bm25': Dataset.searcher.set_bm25(0.82, 0.68)
        if ranker == 'qld': Dataset.searcher.set_qld()
        assert len(queries) == len(qids)
        if batch: raise ValueError('Trec_eval does not accept more than 2GB files! So, we need to break it into several files. No batch search then!')
        else:
            def to_docids(row, o):
                if pd.isna(row.query): return
                hits = Dataset.searcher.search(row.query, k=topk, remove_dups=True)
                for i, h in enumerate(hits): o.write(f'{qids[row.name]}\tQ0\t{h.docid}\t{i + 1:2}\t{h.score:.5f}\tPyserini\n')

            #queries.progress_apply(to_docids, axis=1)
            max_docs_per_file = 400000
            file_index = 0
            for i, doc in tqdm(queries.iterrows(), total=len(queries)):
                if i % max_docs_per_file == 0:
                    if i > 0: out_file.close()
                    output_path = join(f'{out_docids.replace(ranker,"split_"+str(file_index)+"."+ranker)}')
                    out_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                    file_index += 1
                to_docids(doc, out_file)
                if i % 100000 == 0: print(f'wrote {i} files to {out_docids.replace(ranker,"split_"+ str(file_index - 1) + "." + ranker)}')