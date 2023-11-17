from src.refinement.refiners.abstractqrefiner import AbstractQRefiner
from src.refinement.refiners.stem import Stem # Stem refiner is the wrapper for all stemmers as an refiner :)
from src.refinement import refiner_param
from src.refinement import utils

#global analysis
def get_nrf_refiner():
    refiners_list = [AbstractQRefiner()]
    if refiner_param.refiners['Thesaurus']: from src.refinement.refiners.thesaurus import Thesaurus; refiners_list.append(Thesaurus())
    if refiner_param.refiners['Thesaurus']: from src.refinement.refiners.thesaurus import Thesaurus; refiners_list.append(Thesaurus(replace=True))
    if refiner_param.refiners['Wordnet']: from src.refinement.refiners.wordnet import Wordnet; refiners_list.append(Wordnet())
    if refiner_param.refiners['Wordnet']: from src.refinement.refiners.wordnet import Wordnet; refiners_list.append(Wordnet(replace=True))
    if refiner_param.refiners['Word2Vec']: from src.refinement.refiners.word2vec import Word2Vec; refiners_list.append(Word2Vec('../pre/wiki-news-300d-1M.vec'))
    if refiner_param.refiners['Word2Vec']: from src.refinement.refiners.word2vec import Word2Vec; refiners_list.append(Word2Vec('../pre/wiki-news-300d-1M.vec', replace=True))
    if refiner_param.refiners['Glove']: from src.refinement.refiners.glove import Glove; refiners_list.append(Glove('../pre/glove.6B.300d'))
    if refiner_param.refiners['Glove']: from src.refinement.refiners.glove import Glove; refiners_list.append(Glove('../pre/glove.6B.300d', replace=True))
    if refiner_param.refiners['Anchor']: from src.refinement.refiners.anchor import Anchor; refiners_list.append(Anchor(anchorfile='../pre/anchor_text_en.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d.vec'))
    if refiner_param.refiners['Anchor']: from src.refinement.refiners.anchor import Anchor; refiners_list.append(Anchor(anchorfile='../pre/anchor_text_en.ttl', vectorfile='../pre/wiki-anchor-text-en-ttl-300d.vec', replace=True))
    if refiner_param.refiners['Wiki']: from src.refinement.refiners.wiki import Wiki; refiners_list.append(Wiki('../pre/temp_model_Wiki'))
    if refiner_param.refiners['Wiki']: from src.refinement.refiners.wiki import Wiki; refiners_list.append(Wiki('../pre/temp_model_Wiki', replace=True))
    if refiner_param.refiners['Tagmee']: from src.refinement.refiners.tagmee import Tagmee; refiners_list.append(Tagmee())
    if refiner_param.refiners['Tagmee']: from src.refinement.refiners.tagmee import Tagmee; refiners_list.append(Tagmee(replace=True))
    if refiner_param.refiners['SenseDisambiguation']: from src.refinement.refiners.sensedisambiguation import SenseDisambiguation; refiners_list.append(SenseDisambiguation())
    if refiner_param.refiners['SenseDisambiguation']: from src.refinement.refiners.sensedisambiguation import SenseDisambiguation; refiners_list.append(SenseDisambiguation(replace=True))
    if refiner_param.refiners['Conceptnet']: from src.refinement.refiners.conceptnet import Conceptnet; refiners_list.append(Conceptnet())
    if refiner_param.refiners['Conceptnet']: from src.refinement.refiners.conceptnet import Conceptnet; refiners_list.append(Conceptnet(replace=True))
    if refiner_param.refiners['KrovetzStemmer']: from stemmers.krovetz import KrovetzStemmer; refiners_list.append(Stem(KrovetzStemmer(jarfile='stemmers/kstem-3.4.jar')))
    if refiner_param.refiners['LovinsStemmer']: from stemmers.lovins import LovinsStemmer; refiners_list.append(Stem(LovinsStemmer()))
    if refiner_param.refiners['PaiceHuskStemmer']: from stemmers.paicehusk import PaiceHuskStemmer; refiners_list.append(Stem(PaiceHuskStemmer()))
    if refiner_param.refiners['PorterStemmer']: from stemmers.porter import PorterStemmer; refiners_list.append(Stem(PorterStemmer()))
    if refiner_param.refiners['Porter2Stemmer']: from stemmers.porter2 import Porter2Stemmer; refiners_list.append(Stem(Porter2Stemmer()))
    if refiner_param.refiners['SRemovalStemmer']: from stemmers.sstemmer import SRemovalStemmer; refiners_list.append(Stem(SRemovalStemmer()))
    if refiner_param.refiners['Trunc4Stemmer']: from stemmers.trunc4 import Trunc4Stemmer; refiners_list.append(Stem(Trunc4Stemmer()))
    if refiner_param.refiners['Trunc5Stemmer']: from stemmers.trunc5 import Trunc5Stemmer; refiners_list.append(Stem(Trunc5Stemmer()))
    if refiner_param.refiners['BackTranslation']: from src.refinement.refiners.backtranslation import BackTranslation; refiners_list.extend([BackTranslation(each_lng) for index, each_lng in enumerate(refiner_param.backtranslation['tgt_lng'])])
    # since RF needs index and search output which depends on ir method and topics corpora, we cannot add this here. Instead, we run it individually
    # RF assumes that there exist abstractqueryexpansion files

    return refiners_list

#local analysis
def get_rf_refiner(rankers, corpus, output, ext_corpus=None):
    refiners_list = []
    for ranker in rankers:
        ranker_name = utils.get_ranker_name(ranker)
        if refiner_param.refiners['RM3']: from src.refinement.refiners.rm3 import RM3; refiners_list.append(RM3(ranker=ranker_name, index=corpus['index']))
        if refiner_param.refiners['RelevanceFeedback']: from src.refinement.refiners.relevancefeedback import RelevanceFeedback; refiners_list.append(RelevanceFeedback(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=refiner_param.anserini['path'], index=corpus['index']))
        if refiner_param.refiners['Docluster']: from src.refinement.refiners.docluster import Docluster; refiners_list.append(Docluster(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=refiner_param.anserini['path'], index=corpus['index'])),
        if refiner_param.refiners['Termluster']: from src.refinement.refiners.termluster import Termluster; refiners_list.append(Termluster(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=refiner_param.anserini['path'], index=corpus['index']))
        if refiner_param.refiners['Conceptluster']: from src.refinement.refiners.conceptluster import Conceptluster; refiners_list.append(Conceptluster(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=refiner_param.anserini['path'], index=corpus['index']))
        if refiner_param.refiners['BertQE']: from src.refinement.refiners.bertqe import BertQE; refiners_list.append(BertQE(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), index=corpus['index'], anserini=refiner_param.anserini['path']))
        if refiner_param.refiners['OnFields']: from src.refinement.refiners.onfields import OnFields; refiners_list.append(OnFields(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=refiner_param.anserini['path'], index=refiner_param.corpora[corpus]['index'], w_t=corpus['w_t'], w_a=corpus['w_a'], corpus_size=corpus['size']))
        if refiner_param.refiners['AdapOnFields']: from src.refinement.refiners.adaponfields import AdapOnFields; refiners_list.append(AdapOnFields(ranker=ranker_name, prels='{}.abstractqueryexpansion.{}.txt'.format(output, ranker_name), anserini=refiner_param.anserini['path'], index=corpus['index'], w_t=corpus['w_t'], w_a=corpus['w_a'], corpus_size=corpus['size'], collection_tokens=corpus['tokens'], ext_corpus=ext_corpus, ext_index=ext_corpus['index'], ext_collection_tokens=ext_corpus['tokens'], ext_w_t=ext_corpus['w_t'], ext_w_a=ext_corpus['w_a'], ext_corpus_size=ext_corpus['size'], adap=True))

    return refiners_list
