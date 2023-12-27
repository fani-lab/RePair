import ir_datasets
import pandas as pd
import os


def save(dataset, filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # # Queries
    # queries = pd.DataFrame(columns=["id", "text"])
    # for query in dataset.queries_iter():
    #     queries.loc[len(queries)] = [query.query_id, query.text]
    # queries.to_csv(filepath+"topics.antique.txt", index=False)

    # # Documents
    # docs = pd.DataFrame(columns=["id", "text"])
    # ddocs = dataset.docs_cls()
    # for doc in dataset.docs_iter():
    #     docs.loc[len(docs)] = [doc.doc_id, doc.text]
    # docs.to_csv(filepath+"docs.antique.txt", index=False)

    # Qrels
    qrels = pd.DataFrame(columns=["query_id", "doc_id", "relevance", "iteration"])
    for qrel in dataset.qrels_iter():
        qrels.loc[len(qrels)] = [qrel.query_id, qrel.doc_id, qrel.relevance, qrel.iteration]
    qrels.to_csv(filepath+"qrels.antique.txt", index=False)


if __name__ == '__main__':
    # Dbpedia
    # dataset = ir_datasets.load("beir/dbpedia-entity")
    # save(dataset, "topics.Dbpedia.txt")
    # # Antique
    dataset = ir_datasets.load("antique/train")
    save(dataset, "./antique/")