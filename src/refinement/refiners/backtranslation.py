from refinement.refiners.abstractqrefiner import AbstractQRefiner
from refinement.refiner_param import backtranslation
from refinement.lang_code import other, nllb

class BackTranslation(AbstractQRefiner):
    def __init__(self, translator, tgt):
        AbstractQRefiner.__init__(self)

        # Initialization
        self.src = backtranslation['src_lng']
        self.tgt = tgt
        self.translator_name = translator

        # Translation models
        # Meta's NLLB
        if self.translator_name == 'nllb':
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(backtranslation['model_card'])
            tokenizer = AutoTokenizer.from_pretrained(backtranslation['model_card'])
            self.translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=nllb[self.src], tgt_lang=nllb[self.tgt], max_length=backtranslation['max_length'], device=backtranslation['device'])
            self.back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=nllb[self.tgt], tgt_lang=nllb[self.src], max_length=backtranslation['max_length'], device=backtranslation['device'])
        # Other translation models: https://github.com/uliontse/translators?tab=readme-ov-file#supported-translation-services
        else:
            import translators as ts
            self.translator = ts
    '''
    Generates the backtranslated query then calculates the semantic similarity of the two queries
    '''
    def get_refined_query(self, query, args=None):
        if self.translator_name == 'nllb':
            translated_query = self.translator(query)
            backtranslated_query = (self.back_translator(translated_query[0]['translation_text']))[0]['translation_text']
        else:
            translated_query = self.translator.translate_text(query_text=query, translator=self.translator_name, from_language=other[self.src], to_language=other[self.tgt])
            backtranslated_query = self.translator.translate_text(query_text=translated_query, translator=self.translator_name, from_language=other[self.tgt], to_language=other[self.src])
        return super().get_refined_query(backtranslated_query)

    def get_refined_query_batch(self, queries, args=None):
        try:
            translated_queries = self.translator([query for query in queries])
            backtranslated_queries = self.backtranslator([tq_['translation_text'] for tq_ in translated_queries])
            q_s = [q_['translation_text'] for q_ in backtranslated_queries]
        except:
            q_s = [None] * len(queries)
        return q_s

    '''
    Returns the name of the model ('backtranslation) with name of the target language
    Example: 'backtranslation_fra_latn'
    '''
    def get_model_name(self):
        return 'bt_' + self.translator_name + '_' + self.tgt.lower()


if __name__ == "__main__":
    qe = BackTranslation()
    print(qe.get_model_name())
    print(qe.get_refined_query('This is my pc'))
