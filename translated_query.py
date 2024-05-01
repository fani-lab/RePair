from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')

# ['yue_Hant', 'kor_Hang', 'arb_Arab', 'pes_Arab', 'fra_Latn', 'deu_Latn', 'rus_Cyrl', 'zsm_Latn', 'tam_Taml', 'swh_Latn']
def translate(q, l):
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='eng_Latn', tgt_lang=l, max_length=512, device='cpu')
    return translator(q)

q = "murals"
l = 'deu_Latn'
t = translate(q, l)
bt = translate(q, 'eng_Latn')
print(t[0]['translation_text'], )
print(bt[0]['translation_text'])


