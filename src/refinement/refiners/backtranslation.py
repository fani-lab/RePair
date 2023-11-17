from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from src.refinement.refiners.abstractqrefiner import AbstractQRefiner
from src.refinement.refiner_param import backtranslation


class BackTranslation(AbstractQRefiner):
    def __init__(self, tgt):
        AbstractQRefiner.__init__(self)

        # Initialization
        self.tgt = tgt
        model = AutoModelForSeq2SeqLM.from_pretrained(backtranslation['model_card'])
        tokenizer = AutoTokenizer.from_pretrained(backtranslation['model_card'])

        # Translation models
        self.translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=backtranslation['src_lng'], tgt_lang=self.tgt, max_length=backtranslation['max_length'], device=backtranslation['device'])
        self.back_translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=self.tgt, tgt_lang=backtranslation['src_lng'], max_length=backtranslation['max_length'], device=backtranslation['device'])
        # Model use for calculating semsim

    '''
    Generates the backtranslated query then calculates the semantic similarity of the two queries
    '''
    def get_refined_query(self, query, args=None):
        translated_query = self.translator(query.q)
        back_translated_query = self.back_translator(translated_query[0]['translation_text'])
        return back_translated_query[0]['translation_text']
        # return super().get_expanded_query(q, [0])

    def get_refined_query_batch(self, queries, args=None):
        try:
            translated_queries = self.translator([query.q for query in queries])
            back_translated_queries = self.back_translator([tq_['translation_text'] for tq_ in translated_queries])
            q_s = [q_['translation_text'] for q_ in back_translated_queries]
        except:
            q_s = [None] * len(queries)
        return q_s

    '''
    Returns the name of the model ('backtranslation) with name of the target language
    Example: 'backtranslation_fra_latn'
    '''
    def get_model_name(self):
        return super().get_model_name() + '_' + self.tgt.lower()


if __name__ == "__main__":
    qe = BackTranslation()
    print(qe.get_model_name())
    print(qe.get_refined_query('This is my pc'))
