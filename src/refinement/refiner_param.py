import sys, platform

extension = '.exe' if platform.system() == 'Windows' else ""

settings = {
    'transformer_model': 'johngiorgi/declutr-small',
}

refiners = {
        'global': {
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

            'BackTranslation':      1,
        },
        'local': {
            'RelevanceFeedback':    0,
            'Docluster':            0,
            'Termluster':           0,
            'Conceptluster':        0,
            'OnFields':             0,  # make sure that the index for 'extcorpus' is available
            'AdapOnFields':         0,  # make sure that the index for 'extcorpus' is available
            'BertQE':               0,
            'RM3':                  0,
            'T5Transformer':        0,
        },
    }

# Backtranslation settings
backtranslation = {
    'src_lng': 'eng_Latn',
    'tgt_lng': ['yue_Hant', 'kor_Hang', 'arb_Arab', 'pes_Arab', 'fra_Latn', 'deu_Latn', 'rus_Cyrl', 'zsm_Latn', 'tam_Taml', 'swh_Latn'], # ['yue_Hant', 'kor_Hang', 'arb_Arab', 'pes_Arab', 'fra_Latn', 'deu_Latn', 'rus_Cyrl', 'zsm_Latn', 'tam_Taml', 'swh_Latn']
    'max_length': 512,
    'device': 'cpu',
    'model_card': 'facebook/nllb-200-distilled-600M',
}

t5transformer = {
    'pair': 1,
    'finetune': 1,
    'predict': 1,
    't5model': 'small.local',  # 'base.gc' on google cloud tpu, 'small.local' on local machine
    'iter': 5,  # number of finetuning iteration for t5
    'nchanges': 5,  # number of changes to a query
    'pretrained_dir': f'./../output/t5-data/pretrained_models/',
}

