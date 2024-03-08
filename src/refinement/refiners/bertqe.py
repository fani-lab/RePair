import sys

sys.path.extend(['../refinement'])
sys.path.extend(['../pygaggle'])

import string
import torch

from pyserini.index.lucene import IndexReader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
from src.refinement.refiners.relevancefeedback import RelevanceFeedback


# @inproceedings{zheng-etal-2020-bert,
#    title = "{BERT-QE}: {C}ontextualized {Q}uery {E}xpansion for {D}ocument {R}e-ranking",
#    author = "Zheng, Zhi  and Hui, Kai  and  He, Ben  and Han, Xianpei  and  Sun, Le  and Yates, Andrew",
#    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
#    month = nov,
#    year = "2020",
#    address = "Online",
#    publisher = "Association for Computational Linguistics",
#    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.424",
#    pages = "4718--4728",
# }


class BertQE(RelevanceFeedback):
    def __init__(self, ranker, prels, index):
        RelevanceFeedback.__init__(self, ranker, prels, index, topn=10)
        self.model_name = "bert-base-uncased"

    def get_refined_query(self, q, args):
        q = q.translate(str.maketrans('', '', string.punctuation))
        # Phase 1
        raw_docs = self.get_raw_doc(args[0])
        scores_qd, sorted_docs = self.rerank_bert(q, raw_docs)

        # Phase 2
        chunks = self.make_chunks(sorted_docs[:3], 5)
        score_qc, sorted_chunks = self.rerank_bert(q, chunks)

        # Phase 3
        rel_qCd = []
        alpha = 0.5
        # Evaluate relevance of chunks for each document
        for i, chunk in enumerate(sorted_chunks[:10]):
            score_cd, _ = self.rerank_bert(chunk, raw_docs)
            score_cd = tuple(element * score_qc[i] for element in score_cd)
            rel_qCd.append(tuple((1 - alpha) * qd + alpha * cd for qd, cd in zip(scores_qd, score_cd)))

        # Choose the max score and select the corresponding index
        max_score_index = rel_qCd.index(max(rel_qCd))
        # Retrieve the corresponding chunk
        return chunks[max_score_index]

    def make_chunks(self, raw_docs, m):
        chunks = []
        terms = []
        [terms.extend(raw_doc.split()) for raw_doc in raw_docs]

        # Ensure the specified chunk size (m) is valid
        if m <= 0: raise ValueError("Chunk size (m) must be a positive integer.")

        # Calculate the overlap (up to m/2 words)
        overlap = m // 2

        # Iterate through the terms using a sliding window
        for i in range(0, len(terms) - m + 1, m - overlap):
            chunk = ' '.join(terms[i:i + m])
            chunks.append(chunk)

        return chunks

    def get_raw_doc(self, qid):
        topn_docs = self.get_topn_relevant_docids(qid=qid)
        self.index_reader = IndexReader(self.index)
        # topn_docs = random.sample(topn_docs, 5)
        raw_docs = []
        for docid in topn_docs:
            term_positions = self.index_reader.get_term_positions(docid)
            doc = []
            for term, positions in term_positions.items():
                for p in positions: doc.append((term, p))
            doc = ' '.join([t for t, p in sorted(doc, key=lambda x: x[1])])
            doc = ''.join([i if ord(i) < 128 else ' ' for i in doc])
            raw_docs.append(doc)
        return raw_docs

    def rerank_bert(self, q, docs):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=1)  # 1 label for ranking score
        inputs = self.tokenizer([q] * len(docs), docs, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad(): outputs = self.model(**inputs).logits
        scores = softmax(outputs, dim=0).squeeze().tolist()
        return zip(*sorted(zip(scores, docs), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    qe = BertQE(ranker='qld',
                prels='./original.qld',
                index='./lucene-index.gov2.pos+docvectors+rawdocs')
    print(qe.get_model_name())
    print(qe.get_refined_query('Dam removal', [752]))


