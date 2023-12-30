# Created by jing at 28.11.23
import torch

class Ge():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "greater_or_equal_than"

    def fit(self, t1, t2, objs):
        if t1.sum() == 0 or t2.sum() == 0:
            return False

        # If all of A greater than B
        th = 1e-1
        var, mean = torch.var_mean(torch.ge(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.round(mean - var, decimals=2)
            self.p_bound['max'] = torch.round(mean + var, decimals=2)

        return satisfy

    def eval(self, t1, t2, p_space):
        satisfy = torch.ge(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        return satisfy, p_values


class Le():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "less_or_equal_than"

    def fit(self, t1, t2, objs):
        if t1.sum() == 0 or t2.sum() == 0:
            return False

        # If all of A less than B
        th = 1e-1
        var, mean = torch.var_mean(torch.le(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.round(mean - var, decimals=2)
            self.p_bound['max'] = torch.round(mean + var, decimals=2)

        return satisfy

    def eval(self, t1, t2, p_space):
        satisfy = torch.le(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        return satisfy, p_values


class Similar():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "as_similar_as"

    def fit(self, t1, t2, objs):
        # repeat situation
        if objs[1] < objs[0] or t1.sum() == 0 or t2.sum() == 0:
            return False

        th = 0.6
        # If x distance between A and B is similar for all
        var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))
        satisfy = False
        if torch.abs(var / mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.zeros(1)
            self.p_bound['max'] = torch.round(mean + var * 0.5, decimals=2)
        return satisfy

    def eval(self, t1, t2, p_space):
        p_values = torch.round(torch.abs(torch.sub(t1, t2)), decimals=2)
        satisfy = torch.zeros(p_values.size(), dtype=torch.bool)
        for v_i, value in enumerate(p_values):
            if value in p_space:
                satisfy[v_i] = True

        return satisfy, p_values

class Different():
    """ generate one micro-program """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "differ_from"

    def fit(self, t1, t2, objs):
        # repeat situation
        if objs[1] < objs[0] or t1.sum() == 0 or t2.sum() == 0:
            return False

        th = 0.6
        # If x distance between A and B is different for all
        var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))
        satisfy = False
        if torch.abs(var / mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.zeros(1)
            self.p_bound['max'] = torch.round(mean + var * 0.5, decimals=2)
        return satisfy

    def eval(self, t1, t2, p_space):
        p_values = torch.round(torch.abs(torch.sub(t1, t2)), decimals=2)
        satisfy = torch.zeros(p_values.size(), dtype=torch.bool)
        for v_i, value in enumerate(p_values):
            if value in p_space:
                satisfy[v_i] = True

        return satisfy, p_values


def get_preds():
    return [Ge(), Le(), Similar()]

preds = [Ge(), Le(), Similar()]

pred_dict = {"greater_or_equal_than": Ge(),
             "less_or_equal_than": Le(),
             "as_similar_as": Similar()
             }
