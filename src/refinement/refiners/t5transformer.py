from refinement.refiners.abstractqrefiner import AbstractQRefiner
from refinement.refiner_param import t5transformer
from param import settings, corpora
from refinement.mdl import mt5w
import os


class T5Transformer(AbstractQRefiner):
    def __init__(self, domain, corpus, ds, output):
        # Initialization
        AbstractQRefiner.__init__(self)
        self.ds = ds
        self.domain = domain
        self.output = output
        self.datapath = settings['datalist'][settings['datalist'].index(self.domain)]
        self.prep_output = f'./../data/preprocessed/{os.path.split(self.datapath)[-1]}'
        if not os.path.isdir(self.prep_output): os.makedirs(self.prep_output)
        self.t5_model = t5transformer['t5model']  # {"small", "base", "large", "3B", "11B"} cross {"local", "gc"}
        self.index_item_str = '.'.join(corpus['index_item'])
        self.in_type, self.out_type = corpus['pairing'][1], corpus['pairing'][2]
        self.tsv_path = {'train': f'{self.prep_output}/{self.ds.user_pairing}{self.in_type}.{self.out_type}.{self.index_item_str}.train.tsv',
                         'test': f'{self.prep_output}/{self.ds.user_pairing}{self.in_type}.{self.out_type}.{self.index_item_str}.test.tsv'}
        if t5transformer['finetune']: self.fine_tuning()

    def get_refined_query(self, query, args=None):
        # TODO: Change it in a way to take one query at a time
        print(f"Predicting {t5transformer['nchanges']} query changes using {self.t5_model} and storing the results at {self.output} ...")
        mt5w.predict(
            iter=t5transformer['nchanges'],
            split='test',
            tsv_path=self.tsv_path,
            output=self.output,
            lseq=corpora[self.domain]['lseq'],
            model_name=self.get_model_name(),
            gcloud=False)

    def fine_tuning(self):
        print(f"Finetuning {self.t5_model} for {t5transformer['iter']} iterations and storing the checkpoints at {self.output} ...")
        mt5w.finetune(
            tsv_path=self.tsv_path,
            pretrained_dir=t5transformer['pretrained_dir'] + self.t5_model.split(".")[0],
            # "gs://t5-data/pretrained_models/{"small", "base", "large", "3B", "11B"}
            steps=t5transformer['iter'],
            output=f'{self.output}/model',
            task_name=f"{self.domain.replace('-', '')}_cf",
            # :DD Task name must match regex: ^[\w\d\.\:_]+$
            lseq=corpora[self.domain]['lseq'],
            nexamples=None, in_type=self.in_type, out_type=self.out_type, gcloud=False)

    def paring(self):
        print('Pairing queries and relevant passages for training set ...')
        cat = True if 'docs' in {self.in_type, self.out_type} else False
        query_qrel_doc = self.ds.pair(self.datapath, f'{self.prep_output}/{self.ds.user_pairing}queries.qrels.doc{"s" if cat else ""}.ctx.{self.index_item_str}.train.no_dups.tsv', cat=cat)
        # TODO: query_qrel_doc = pair(datapath, f'{prep_output}/queries.qrels.doc.ctx.{index_item_str}.test.tsv')
        query_qrel_doc.to_csv(self.tsv_path['train'], sep='\t', encoding='utf-8', index=False, columns=[self.in_type, self.out_type], header=False)
        query_qrel_doc.to_csv(self.tsv_path['test'], sep='\t', encoding='utf-8', index=False, columns=[self.in_type, self.out_type], header=False)

    def get_model_name(self):
        # return f'{self.ds.user_pairing}t5.{self.t5_model}.{self.in_type}.{self.out_type}.{self.ds.index_item_str}'
        return f't5.{self.t5_model}.pred'


if __name__ == "__main__":
    qe = T5Transformer()
    print(qe.get_model_name())
    print(qe.get_refined_query('This is my pc'))