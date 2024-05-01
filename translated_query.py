def nllb_translate(q, src, tgt):
    # ['yue_Hant', 'kor_Hang', 'arb_Arab', 'pes_Arab', 'fra_Latn', 'deu_Latn', 'rus_Cyrl', 'zsm_Latn', 'tam_Taml', 'swh_Latn']
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
    tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src, tgt_lang=tgt, max_length=512, device='cpu')
    return translator(q)

def google_translate(q, src, tgt):
    from googletrans import Translator
    translator = Translator(service_urls=['translate.google.com'])
    translated_query = translator.translate(q, src=src, dest=tgt)
    return translated_query.text

# NLLB
# q = "murals"
# l = 'deu_Latn'
# t = nllb_translate(q, 'eng_Latn', l)
# bt = nllb_translate(q, l, 'eng_Latn')
# print(t[0]['translation_text'])
# print(bt[0]['translation_text'])

# Google
q = "sick building syndrome"
l = 'fr'
t = google_translate(q, 'en', l)
bt = google_translate(t, l, 'en')
print(t)
print(bt)