from pywsd import disambiguate
from refinement.refiners.abstractqrefiner import AbstractQRefiner


class SenseDisambiguation(AbstractQRefiner):
    def __init__(self, replace=False):
        AbstractQRefiner.__init__(self, replace)

    def get_refined_query(self, q, args=None):
        res = []
        disamb = disambiguate(q)
        for i,t in enumerate(disamb):
            if t[1] is not None:
                if not self.replace:
                    res.append(t[0])
                x=t[1].name().split('.')[0].split('_')
                if t[0].lower() != (' '.join(x)).lower() or self.replace:
                    res.append(' '.join(x))
            else:
                res.append(t[0])
        return super().get_refined_query(' '.join(res))


if __name__ == "__main__":
    qe = SenseDisambiguation()
    print(qe.get_model_name())
    print(qe.get_refined_query('obama family tree'))
    print(qe.get_refined_query('HosseinFani International Crime Organization'))

    qe = SenseDisambiguation(replace=True)
    print(qe.get_model_name())
    print(qe.get_refined_query('maryland department of natural resources'))
    print(qe.get_refined_query('HosseinFani International Crime Organization'))
