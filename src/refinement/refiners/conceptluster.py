import tagme
tagme.GCUBE_TOKEN = "10df41c6-f741-45fc-88dd-9b24b2568a7b"
from src.refinement.refiners.termluster import Termluster
import os


class Conceptluster(Termluster):
    def __init__(self, ranker, prels, index, topn=5, topw=3):
        Termluster.__init__(self, ranker, prels, index, topn=topn, topw=topw)

    def get_refined_query(self, q, args):
        qid = args[0]
        list_of_concept_lists = []
        docids = self.get_topn_relevant_docids(qid)
        for docid in docids:
            doc_text = self.get_document(docid)
            concept_list = self.get_concepts(doc_text, score=0.1)
            list_of_concept_lists.append(concept_list)

        G, cluster_dict = self.make_graph_document(list_of_concept_lists, min_edge=10)
        expanded_query = self.refine_query_concept_cluster(q, G, cluster_dict, k_relevant_words=self.topw)
        return super().get_refined_query(expanded_query, args[0])

    def refine_query_concept_cluster(self, q, G, cluster_dict, k_relevant_words):
        q += ' ' + ' '.join(self.get_concepts(q, 0.1))
        return super().refined_query_term_cluster(q, G, cluster_dict, k_relevant_words)

    def get_document(self, docid):
        command = f'\"./anserini/target/appassembler/bin/IndexUtils\" -index \"{self.index}\" -dumpRawDoc \"{docid}\"'
        stream = os.popen(command)
        return stream.read()

    def get_concepts(self, text, score):
        concepts = tagme.annotate(text).get_annotations(score)
        return list(set([c.entity_title for c in concepts if c.entity_title not in text]))


if __name__ == "__main__":
    qe = Conceptluster(ranker='bm25',
                   prels='./output/robust04/topics.robust04.abstractqueryexpansion.bm25.txt',
                   index='../ds/robust04/index-robust04-20191213')

    for i in range(5):
        print(qe.get_model_name())
        print(qe.get_refined_query('HosseinFani International Crime Organization', [301]))
        print(qe.get_refined_query('Agoraphobia', [698]))
        print(qe.get_refined_query('Unsolicited Faxes', [317]))
