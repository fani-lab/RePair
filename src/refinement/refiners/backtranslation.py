from refinement.refiners.abstractqrefiner import AbstractQRefiner
from refinement.refiner_param import backtranslation
from refinement.lang_code import google, nllb

class BackTranslation(AbstractQRefiner):
    def __init__(self, translator, tgt):
        AbstractQRefiner.__init__(self)

        # Initialization
        self.src = backtranslation['src_lng']
        self.tgt = tgt
        self.translator_name = translator

        # Translation models
        # Google
        if self.translator_name == 'google':
            from googletrans import Translator
            self.translator = Translator(service_urls=['translate.google.com'])
        # Meta's NLLB
        else:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(backtranslation['model_card'])
            tokenizer = AutoTokenizer.from_pretrained(backtranslation['model_card'])
            self.translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=nllb[self.src], tgt_lang=nllb[self.tgt], max_length=backtranslation['max_length'], device=backtranslation['device'])
            self.back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=nllb[self.tgt], tgt_lang=nllb[self.src], max_length=backtranslation['max_length'], device=backtranslation['device'])

    '''
    Generates the backtranslated query then calculates the semantic similarity of the two queries
    '''
    def get_refined_query(self, query, args=None):
        if self.translator_name == 'google':
            translated_query = self.translator.translate(query, src=google[self.src], dest=google[self.tgt])
            backtranslated_query = (self.translator.translate(translated_query.text, src=google[self.tgt], dest=google[self.src])).text
        else:
            translated_query = self.translator(query)
            backtranslated_query = (self.back_translator(translated_query[0]['translation_text']))[0]['translation_text']
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
