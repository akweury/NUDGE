# Created by jing at 31.01.24


class VarianceFact():
    """ initiate one fact object """

    def __init__(self,data, mask, obj_combs, prop_comb, preds, delta):
        super().__init__()
        self.data = data
        self.mask = mask
        self.obj_comb = obj_combs
        self.prop_comb = prop_comb
        self.preds = preds
        self.delta = delta
    def fact_variance(self):
        pass


class ProbFact:
    def __init__(self,preds, mask, obj_combs, prop_comb, delta):

        super().__init__()
        self.preds = preds
        self.mask = mask
        self.obj_comb = obj_combs
        self.prop_comb = prop_comb
        self.delta = delta
    def fact_prob(self):
        pass