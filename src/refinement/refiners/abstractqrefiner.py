import traceback
import pandas as pd
import sys

sys.path.extend(['../refinement'])
from refinement import utils


class AbstractQRefiner:
    def __init__(self, replace=False, topn=None):
        self.replace = replace
        self.topn = topn
        self.query_set = pd.DataFrame(columns=['qid'], dtype=int)

    # all children expanders must call this in the returning line
    def get_refined_query(self, q, args=None):
        return q, args

    def get_model_name(self):
        # this is for backward compatibility for renaming this class
        if self.__class__.__name__ == 'AbstractQExpander': return 'AbstractQueryExpansion'.lower()
        return f"{self.__class__.__name__.lower()}{f'.topn{self.topn}' if self.topn else ''}{'.replace' if self.replace else ''}"

    def get_query_set(self):
        return self.query_set

    def generate_queue(self, Qfilename, Q_filename):
        query_df = self.read_queries(Qfilename)
        query_df.apply(lambda row: self.preprocess_expanded_query(q=row[self.get_model_name().lower()], qid=row['qid'], clean=True), axis=1)
        self.write_queries(Q_filename)

    def preprocess_expanded_query(self, q, qid, clean):
        try:
            q_, args = self.get_expanded_query(q, args=[qid])
            q_ = utils.clean(q_) if clean else q_
        except:
            print(f'WARNING: MAIN: {self.get_model_name()}: Expanding query [{qid}:{q}] failed!')
            print(traceback.format_exc())
            q_.args = q, [0]

        new_line = {'qid': qid, self.get_model_name(): q_}
        if self.get_model_name().__contains__('backtranslation'): new_line['semsim'] = args[0]
        self.query_set = pd.concat([self.query_set, pd.DataFrame([new_line])], ignore_index=True)
        print(f'INFO: MAIN: {self.get_model_name()}: {qid}: {q} -> {q_}')

    def write_queries(self, Q_filename):
        self.query_set.to_csv(Q_filename, sep='\t', index=False, header=False)

    def read_queries(self, Qfilename):
        model_name = self.get_model_name().lower()
        is_tag_file = False
        q, qid = '', ''
        query_df = pd.DataFrame(columns=['qid'])
        with open(Qfilename, 'r', encoding='UTF-8') as Qfile:
            for line in Qfile:
                if '<top>' in line and not is_tag_file: is_tag_file = True
                if '<num>' in line: qid = int(line[line.index(':') + 1:])
                elif line[:7] == '<title>':
                    q = line[8:].strip()
                    if not q: q = next(Qfile).strip()
                elif '<topic' in line:
                    s = line.index('\"') + 1
                    e = line.index('\"', s + 1)
                    qid = int(line[s:e])
                elif line[2:9] == '<query>': q = line[9:-9]
                elif len(line.split('\t')) >= 2 and not is_tag_file:
                    qid = line.split('\t')[0].rstrip()
                    q = line.split('\t')[1].rstrip()
                if q != '' and qid != '':
                    new_line = {'qid': qid, model_name: q}
                    query_df = pd.concat([query_df, pd.DataFrame([new_line])], ignore_index=True)
                    q, qid = '', ''
        return query_df.astype({'qid': 'str'})

    def write_expanded_queries(self, Qfilename, Q_filename, clean=True):
        # prevent to clean the original query
        if self.__class__.__name__ == 'AbstractQExpander': clean = False
        model_name = self.get_model_name().lower()
        Q_ = pd.DataFrame()
        is_tag_file = False
        with open(Qfilename, 'r', encoding='UTF-8') as Qfile:
            with open(Q_filename, 'w', encoding='UTF-8') as Q_file:
                print(f'INFO: MAIN: {self.get_model_name()}: Expanding queries in {Qfilename} ...')
                for line in Qfile:
                    # For txt files
                    if '<top>' in line and not is_tag_file: is_tag_file = True
                    if '<num>' in line:
                        qid = int(line[line.index(':') + 1:])
                        Q_file.write(line)
                    # For robust & gov2
                    elif line[:7] == '<title>':
                        q = line[8:].strip()
                        if not q: q = next(Qfile).strip()
                        q_, args, Q_ = self.preprocess_expanded_query(q, qid, clean, Q_)
                        if model_name.__contains__('backtranslation'): Q_file.write(
                            f'<semsim> {args[0]:.4f} </semsim>\n')
                        Q_file.write('<title> ' + str(q_) + '\n')
                    elif '<topic' in line:
                        s = line.index('\"') + 1
                        e = line.index('\"', s + 1)
                        qid = int(line[s:e])
                        Q_file.write(line)
                    # For clueweb09b & clueweb12b13
                    elif line[2:9] == '<query>':
                        q = line[9:-9]
                        q_, args, Q_ = self.preprocess_expanded_query(q, qid, clean, Q_)
                        if model_name.__contains__('backtranslation'): Q_file.write(
                            f'<semsim> {args[0]:.4f} </semsim>\n')
                        Q_file.write('  <query>' + str(q_) + '</query>' + '\n')
                    # For tsv files
                    elif len(line.split('\t')) >= 2 and not is_tag_file:
                        qid = line.split('\t')[0].rstrip()
                        q = line.split('\t')[1].rstrip()
                        q_, args, Q_ = self.preprocess_expanded_query(q, qid, clean, Q_)
                        Q_file.write(qid + '\t' + str(q_))
                        Q_file.write('\t' + str(args[0]) + '\n') if model_name.__contains__('backtranslation') else Q_file.write('\n')
                    else:
                        Q_file.write(line)
        return Q_

    def read_expanded_queries(self, Q_filename):
        model_name = self.get_model_name().lower()
        Q_ = pd.DataFrame(columns=['qid'], dtype=int)
        is_tag_file = False
        with open(Q_filename, 'r', encoding='UTF-8') as Q_file:
            print(f'INFO: MAIN: {self.get_model_name()}: Reading expanded queries in {Q_filename} ...')
            for line in Q_file:
                q_ = None
                # for files with tag
                if '<top>' in line and not is_tag_file: is_tag_file = True
                if '<num>' in line:
                    qid = line[line.index(':') + 1:].strip()
                elif '<semsim>' in line:
                    score = line[8:-10] + ' '
                # for robust & gov2
                elif line[:7] == '<title>':
                    q_ = line[8:].strip() + ' '
                elif '<topic' in line:
                    s = line.index('\"') + 1
                    e = line.index('\"', s + 1)
                    qid = line[s:e].strip()
                # for clueweb09b & clueweb12b13
                elif line[2:9] == '<query>':
                    q_ = line[9:-9] + ' '
                elif len(line.split('\t')) >= 2 and not is_tag_file:
                    qid = line.split('\t')[0].rstrip()
                    q_ = line.split('\t')[1].rstrip()
                    if model_name.__contains__('backtranslation'): score = line.split('\t')[2].rstrip()
                else:
                    continue
                if q_:
                    new_line = {'qid': qid, model_name: q_}
                    # For backtranslation expander add a new column as semsim
                    if model_name.__contains__('backtranslation'): new_line['semsim'] = score
                    Q_ = pd.concat([Q_, pd.DataFrame([new_line])], ignore_index=True)
        return Q_.astype({'qid': 'str'})


if __name__ == "__main__":
    qe = AbstractQRefiner()
    print(qe.get_model_name())
    print(qe.get_expanded_query('International Crime Organization'))

    # from expanders.abstractqexpander import AbstractQExpander
    # from expanders.sensedisambiguation import SenseDisambiguation
    # from expanders.thesaurus import Thesaurus
    # from expanders.wordnet import Wordnet
    # from expanders.word2vec import Word2Vec
    from expanders.anchor import Anchor
    # from expanders.glove import Glove
    # from expanders.conceptnet import Conceptnet
    # from expanders.relevancefeedback import RelevanceFeedback
    # from expanders.stem import Stem  # Stem expander is the wrapper for all stemmers as an expnader :)
    # from stemmers.krovetz import KrovetzStemmer
    # from stemmers.lovins import LovinsStemmer
    # from stemmers.paicehusk import PaiceHuskStemmer
    # from stemmers.porter import PorterStemmer
    # from stemmers.porter2 import Porter2Stemmer
    # from stemmers.sstemmer import SRemovalStemmer
    # from stemmers.trunc4 import Trunc4Stemmer
    # from stemmers.trunc5 import Trunc5Stemmer
    from expanders.docluster import Docluster
    from expanders.termluster import Termluster
    from expanders.conceptluster import Conceptluster

    expanders = [AbstractQRefiner(),
                 #              Thesaurus(),
                 #              Wordnet(),
                 #              Word2Vec('../pre/wiki-news-300d-1M.vec', topn=3),
                 #              Glove('../pre/glove.6B.300d', topn=3),
                 #              SenseDisambiguation(),
                 #              Conceptnet(),
                 #              Thesaurus(replace=True),
                 #              Wordnet(replace=True),
                 #              Word2Vec('../pre/wiki-news-300d-1M.vec', topn=3, replace=True),
                 #              Glove('../pre/glove.6B.300d', topn=3, replace=True),
                 #              SenseDisambiguation(replace=True),
                 #              Conceptnet(replace=True),
                 #              Stem(KrovetzStemmer(jarfile='stemmers/kstem-3.4.jar')),
                 #              Stem(LovinsStemmer()),
                 #              Stem(PaiceHuskStemmer()),
                 #              Stem(PorterStemmer()),
                 #              Stem(Porter2Stemmer()),
                 #              Stem(SRemovalStemmer()),
                 #              Stem(Trunc4Stemmer()),
                 #              Stem(Trunc5Stemmer()),
                 #              RelevanceFeedback(ranker='bm25', prels='./output/robust04/topics.robust04.abstractqueryexpansion.bm25.txt',anserini='../anserini/',index='../ds/robust04/index-robust04-20191213'),
                 #              Docluster(ranker='bm25', prels='./output/robust04/topics.robust04.abstractqueryexpansion.bm25.txt',anserini='../anserini/', index='../ds/robust04/index-robust04-20191213'),
                 #              Termluster(ranker='bm25', prels='./output/robust04/topics.robust04.abstractqueryexpansion.bm25.txt',anserini='../anserini/', index='../ds/robust04/index-robust04-20191213'),
                 #              Conceptluster(ranker='bm25', prels='./output/robust04/topics.robust04.abstractqueryexpansion.bm25.txt', anserini='../anserini/', index='../ds/robust04/index-robust04-20191213'),
                 #              Anchor(anchorfile='../pre/anchor_text_en_sample.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d-sample.vec', topn=3),
                 #              Anchor(anchorfile='../pre/anchor_text_en_sample.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d-sample.vec', topn=3, replace=True)
                 ]
    for expander in expanders: expander.write_expanded_queries('../ds/robust04/topics.robust04.txt', 'dummy.txt')
    # expanders.write_expanded_queries('../ds/gov2/topics.terabyte05.751-800.txt', 'dummy')
    # expanders.write_expanded_queries('../ds/clueweb09b/topics.web.101-150.txt', 'dummy')
    # expanders.write_expanded_queries('../ds/clueweb12b13/topics.web.201-250.txt', 'dummy')
