import csv, json
import pandas as pd
from ftfy import fix_text
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

def msmarco(input, output):
    queries_source = pd.read_csv(f'{input}/queries.target.tsv', sep="\t", index_col=False)
    with open(f'{input}/qrels.target.tsv') as f:
        qrels = csv.reader(f, delimiter='\t')
        next(qrels)
        with open(f'{output}/qrels.target.tsv', 'w') as pf, open(f'{input}/queries.target.tsv', 'w') as qf:
            pf.write("pid\tpassage\n")
            qf.write("qid\tquery\n")
            for line in pf:
                fetch_qid = queries_source.loc[queries_source['qid'] == int(line[0])]
                try:
                    doc = searcher.doc(line[2])#docid
                    json_doc = json.loads(doc.raw())
                    retrieved_passage = fix_text(json_doc['contents'])
                except:
                    raise ValueError('the provided docid is not valid')

                qf.write(f"{fetch_qid['qid'].squeeze()}\t{fetch_qid['query'].squeeze()}\n")
                qp.write(f'{line[2]}\t{retrieved_passage}\n')

def convert_to_features(self, example_batch):
    # Tokenize contexts and questions (as pairs of inputs)

    if self.print_text:
        print("Input Text: ", self.clean_text(example_batch['text']))
    #         input_ = self.clean_text(example_batch['text']) + " </s>"
    #         target_ = self.clean_text(example_batch['headline']) + " </s>"

    input_ = self.clean_text(example_batch['text'])
    target_ = self.clean_text(example_batch['headline'])

    source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                              padding='max_length', truncation=True, return_tensors="pt")

    targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                               padding='max_length', truncation=True, return_tensors="pt")

    return source, targets


def __getitem__(self, index):
    source, targets = convert_to_features(self.dataset[index])

    source_ids = source["input_ids"].squeeze()
    target_ids = targets["input_ids"].squeeze()

    src_mask = source["attention_mask"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
