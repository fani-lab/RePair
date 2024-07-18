from refinement.refiners.abstractqrefiner import AbstractQRefiner
from refinement.refiners.stem import Stem   # Stem refiner is the wrapper for all stemmers as an refiner :)
from refinement import refiner_param
from itertools import product
from param import settings
import os


#global analysis
def get_nrf_refiner():
    refiners_list = [AbstractQRefiner()]
    refiners_name = refiner_param.refiners['global']
    if refiners_name['Thesaurus']: from refinement.refiners.thesaurus import Thesaurus; refiners_list.append(Thesaurus())
    if refiners_name['Thesaurus']: from refinement.refiners.thesaurus import Thesaurus; refiners_list.append(Thesaurus(replace=True))
    if refiners_name['Wordnet']: from refinement.refiners.wordnet import Wordnet; refiners_list.append(Wordnet())
    if refiners_name['Wordnet']: from refinement.refiners.wordnet import Wordnet; refiners_list.append(Wordnet(replace=True))
    if refiners_name['Word2Vec']: from refinement.refiners.word2vec import Word2Vec; refiners_list.append(Word2Vec('./refinement/pre/wiki-news-300d-1M.vec'))
    if refiners_name['Word2Vec']: from refinement.refiners.word2vec import Word2Vec; refiners_list.append(Word2Vec('./refinement/pre/wiki-news-300d-1M.vec', replace=True))
    if refiners_name['Glove']: from refinement.refiners.glove import Glove; refiners_list.append(Glove('./refinement/pre/glove.6B.300d'))
    if refiners_name['Glove']: from refinement.refiners.glove import Glove; refiners_list.append(Glove('./refinement/pre/glove.6B.300d', replace=True))
    if refiners_name['Anchor']: from refinement.refiners.anchor import Anchor; refiners_list.append(Anchor(anchorfile='./refinement/pre/anchor_text_en.ttl', vectorfile='./refinement/pre/wiki-anchor-text-en-ttl-300d.vec'))
    if refiners_name['Anchor']: from refinement.refiners.anchor import Anchor; refiners_list.append(Anchor(anchorfile='./refinement/pre/anchor_text_en.ttl', vectorfile='./refinement/pre/wiki-anchor-text-en-ttl-300d.vec', replace=True))
    if refiners_name['Wiki']: from refinement.refiners.wiki import Wiki; refiners_list.append(Wiki('./refinement/pre/temp_model_Wiki'))
    if refiners_name['Wiki']: from refinement.refiners.wiki import Wiki; refiners_list.append(Wiki('./refinement/pre/temp_model_Wiki', replace=True))
    if refiners_name['Tagmee']: from refinement.refiners.tagmee import Tagmee; refiners_list.append(Tagmee())
    if refiners_name['Tagmee']: from refinement.refiners.tagmee import Tagmee; refiners_list.append(Tagmee(replace=True))
    if refiners_name['SenseDisambiguation']: from refinement.refiners.sensedisambiguation import SenseDisambiguation; refiners_list.append(SenseDisambiguation())
    if refiners_name['SenseDisambiguation']: from refinement.refiners.sensedisambiguation import SenseDisambiguation; refiners_list.append(SenseDisambiguation(replace=True))
    if refiners_name['Conceptnet']: from refinement.refiners.conceptnet import Conceptnet; refiners_list.append(Conceptnet())
    if refiners_name['Conceptnet']: from refinement.refiners.conceptnet import Conceptnet; refiners_list.append(Conceptnet(replace=True))
    if refiners_name['KrovetzStemmer']: from refinement.stemmers.krovetz import KrovetzStemmer; refiners_list.append(Stem(KrovetzStemmer(jarfile='./refinement/pre/kstem-3.4.jar')))
    if refiners_name['LovinsStemmer']: from refinement.stemmers.lovins import LovinsStemmer; refiners_list.append(Stem(LovinsStemmer()))
    if refiners_name['PaiceHuskStemmer']: from refinement.stemmers.paicehusk import PaiceHuskStemmer; refiners_list.append(Stem(PaiceHuskStemmer()))
    if refiners_name['PorterStemmer']: from refinement.stemmers.porter import PorterStemmer; refiners_list.append(Stem(PorterStemmer()))
    if refiners_name['Porter2Stemmer']: from refinement.stemmers.porter2 import Porter2Stemmer; refiners_list.append(Stem(Porter2Stemmer()))
    if refiners_name['SRemovalStemmer']: from refinement.stemmers.sstemmer import SRemovalStemmer; refiners_list.append(Stem(SRemovalStemmer()))
    if refiners_name['Trunc4Stemmer']: from refinement.stemmers.trunc4 import Trunc4Stemmer; refiners_list.append(Stem(Trunc4Stemmer()))
    if refiners_name['Trunc5Stemmer']: from refinement.stemmers.trunc5 import Trunc5Stemmer; refiners_list.append(Stem(Trunc5Stemmer()))
    if refiners_name['BackTranslation']: from refinement.refiners.backtranslation import BackTranslation; refiners_list.extend([BackTranslation(trans, lang) for lang, trans in product(refiner_param.backtranslation['tgt_lng'], refiner_param.backtranslation['translator'])])
    # since RF needs index and search output which depends on ir method and topics corpora, we cannot add this here. Instead, we run it individually
    # RF assumes that there exist abstractqueryexpansion files

    return refiners_list


#local analysis
def get_rf_refiner(output, corpus, ext_corpus=None, ds=None, domain=None):
    refiners_list = []
    refiners_name = refiner_param.refiners['local']
    for ranker in settings['ranker']:
        ranker_folder = next((folder for folder in [folder for folder in os.listdir(output) if os.path.isdir(os.path.join(output, folder))] if ranker in folder), None)
        prels = f'{output}/{ranker_folder}/original.{ranker}'
        if refiners_name['RM3']: from refinement.refiners.rm3 import RM3; refiners_list.append(RM3(ranker=ranker, index=corpus['index']))
        if refiners_name['RelevanceFeedback']: from refinement.refiners.relevancefeedback import RelevanceFeedback; refiners_list.append(RelevanceFeedback(ranker=ranker, prels=prels, index=corpus['index']))
        if refiners_name['Docluster']: from refinement.refiners.docluster import Docluster; refiners_list.append(Docluster(ranker=ranker, prels=prels, index=corpus['index'])),
        if refiners_name['Termluster']: from refinement.refiners.termluster import Termluster; refiners_list.append(Termluster(ranker=ranker, prels=prels, index=corpus['index']))
        if refiners_name['Conceptluster']: from refinement.refiners.conceptluster import Conceptluster; refiners_list.append(Conceptluster(ranker=ranker, prels=prels, index=corpus['index']))
        if refiners_name['BertQE']: from refinement.refiners.bertqe import BertQE; refiners_list.append(BertQE(ranker=ranker, prels=prels, index=corpus['index']))
        if refiners_name['OnFields']: from refinement.refiners.onfields import OnFields; refiners_list.append(OnFields(ranker=ranker, prels=prels, index=refiner_param.corpora[corpus]['index'], w_t=corpus['w_t'], w_a=corpus['w_a'], corpus_size=corpus['size']))
        if refiners_name['AdapOnFields']: from refinement.refiners.adaponfields import AdapOnFields; refiners_list.append(AdapOnFields(ranker=ranker, prels=prels, index=corpus['index'], w_t=corpus['w_t'], w_a=corpus['w_a'], corpus_size=corpus['size'], collection_tokens=corpus['tokens'], ext_corpus=ext_corpus, ext_index=ext_corpus['index'], ext_collection_tokens=ext_corpus['tokens'], ext_w_t=ext_corpus['w_t'], ext_w_a=ext_corpus['w_a'], ext_corpus_size=ext_corpus['size'], adap=True))

        if refiners_name['T5Transformer']: from refinement.refiners.t5transformer import T5Transformer; refiners_list.append(Stem(T5Transformer(domain=domain, corpus=corpus, ds=ds, output=output)))

    return refiners_list
