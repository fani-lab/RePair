import sys
sys.path.extend(['../refinement'])

from src.refinement.refiners.abstractqrefiner import AbstractQRefiner
from src.refinement.stemmers.abstractstemmer import AbstractStemmer
class Stem(AbstractQRefiner):
    def __init__(self, stemmer:AbstractStemmer):
        AbstractQRefiner.__init__(self, replace=False)
        self.stemmer = stemmer

    def get_model_name(self):
        return super().get_model_name() + '.' + self.stemmer.basename

    def get_refined_query(self, q, args=None):
        return super().get_refined_query(self.stemmer.stem_query(q))

if __name__ == "__main__":
    from src.refinement.stemmers.krovetz import KrovetzStemmer
    qe = Stem(KrovetzStemmer(jarfile='stemmers/kstem-3.4.jar'))
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.lovins import LovinsStemmer
    qe = Stem(LovinsStemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.paicehusk import PaiceHuskStemmer
    qe = Stem(PaiceHuskStemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.porter import PorterStemmer
    qe = Stem(PorterStemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.porter2 import Porter2Stemmer
    qe = Stem(Porter2Stemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.sstemmer import SRemovalStemmer
    qe = Stem(SRemovalStemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.trunc4 import Trunc4Stemmer
    qe = Stem(Trunc4Stemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
    from refinement.stemmers.trunc5 import Trunc5Stemmer
    qe = Stem(Trunc5Stemmer())
    print(qe.get_model_name())
    print(qe.get_refined_query('International Crime Organization'))
