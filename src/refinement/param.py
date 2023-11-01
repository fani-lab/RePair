import sys, platform

extension = '.exe' if platform.system() == 'Windows' else ""

refiners = {
        'SenseDisambiguation':  0,
        'Thesaurus':            0,
        'Wordnet':              0,
        'Conceptnet':           0,
        'Tagmee':               0,

        'Word2Vec':             0,
        'Glove':                0,
        'Anchor':               0,
        'Wiki':                 0,

        'KrovetzStemmer':       0,
        'LovinsStemmer':        0,
        'PaiceHuskStemmer':     0,
        'PorterStemmer':        0,
        'Porter2Stemmer':       0,
        'SRemovalStemmer':      0,
        'Trunc4Stemmer':        0,
        'Trunc5Stemmer':        0,

        'RelevanceFeedback':    0,
        'Docluster':            0,
        'Termluster':           0,
        'Conceptluster':        0,
        'OnFields':             0,  # make sure that the index for 'extcorpus' is available
        'AdapOnFields':         0,  # make sure that the index for 'extcorpus' is available
        'BertQE':               0,
        'RM3':                  0,

        'BackTranslation':      1,
    }

# Backtranslation settings
backtranslation = {
    'src_lng': 'eng_Latn',
    'tgt_lng': ['fra_Latn'], # ['yue_Hant', 'kor_Hang', 'arb_Arab', 'pes_Arab', 'fra_Latn', 'deu_Latn', 'rus_Cyrl', 'zsm_Latn', 'tam_Taml', 'swh_Latn']
    'max_length': 512,
    'device': 'cpu',
    'model_card': 'facebook/nllb-200-distilled-600M',
    'transformer_model': 'johngiorgi/declutr-small',
}
