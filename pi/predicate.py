# Created by jing at 28.11.23
import torch


class GT():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "greater_or_equal_than"
        self.p_spaces = []

    def fit(self, t1, t2, objs):
        th = 1e-1

        if t1.sum() == 0 or t2.sum() == 0:
            return False

        # single state
        if len(t1) == 1:
            var, mean = torch.tensor(0), torch.gt(t1, t2).float()
        # multiple states
        else:
            th = 1e-1
            var, mean = torch.var_mean(torch.gt(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.round(mean - var, decimals=2)
            self.p_bound['max'] = torch.round(mean + var, decimals=2)

        return satisfy

    def update_space(self, t1, t2):
        return

    def eval(self, t1, t2, p_space):
        satisfy = torch.gt(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        return satisfy, p_values

    def eval_batch(self, t1, t2, p_space):
        th = 1e-1
        if t1.sum() == 0 or t2.sum() == 0:
            return False

        # single state
        if len(t1) == 1:
            var, mean = torch.tensor(0), torch.gt(t1, t2).float()
        # multiple states
        else:
            th = 1e-1
            var, mean = torch.var_mean(torch.gt(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True
        return satisfy


class LT():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "less_or_equal_than"
        self.p_spaces = []

    def fit(self, t1, t2, objs):
        th = 1e-1
        if t1.sum() == 0 or t2.sum() == 0:
            return False

        if len(t1) == 1:
            var, mean = torch.tensor(0), torch.lt(t1, t2).float()
        else:
            var, mean = torch.var_mean(torch.lt(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.round(mean - var, decimals=2)
            self.p_bound['max'] = torch.round(mean + var, decimals=2)

        return satisfy

    def update_space(self, t1, t2):
        return

    def eval_batch(self, t1, t2, p_space):
        th = 1e-1
        if t1.sum() == 0 or t2.sum() == 0:
            return False
        if len(t1) == 1:
            var, mean = torch.tensor(0), torch.lt(t1, t2).float()
        else:
            var, mean = torch.var_mean(torch.lt(t1, t2).float())

        satisfy = False
        if var < th and (1 - mean) < th:
            satisfy = True

        return satisfy

    def eval(self, t1, t2, p_space):
        satisfy = torch.lt(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        return satisfy, p_values


class Similar():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "as_similar_as"
        self.p_spaces = []

    def fit(self, t1, t2, objs):
        th = 0.6

        # repeat situation
        if objs[1] < objs[0] or t1.sum() == 0 or t2.sum() == 0:
            return False

        if len(t1) == 1:
            var, mean = torch.tensor(0), torch.abs(torch.sub(t1, t2))
        else:
            var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))
        satisfy = False
        if torch.abs(var / mean) < th:
            satisfy = True
            self.p_bound['min'] = torch.zeros(1)
            self.p_bound['max'] = torch.round(mean + var * 0.5, decimals=2)
        return satisfy

    def update_space(self, t1, t2):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        p_values = torch.round(torch.abs(torch.sub(t1, t2)), decimals=2)
        for value in p_values:
            if value not in self.p_spaces:
                self.p_spaces.append(value)

    def eval_batch(self, t1, t2, p_space):
        th = 0.6

        if len(t1) == 1:
            var, mean = torch.tensor(0), torch.abs(torch.sub(t1, t2))
        else:
            var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))
        satisfy = False
        if torch.abs(var / mean) < th:
            satisfy = True
        return satisfy

    def eval(self, t1, t2, p_space):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
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


def get_preds(repeats=1):
    all_preds = []
    for i in range(repeats):
        all_preds += [GT(), LT(), Similar()]
    return all_preds


preds = [GT(), LT(), Similar()]

pred_dict = {"greater_or_equal_than": GT(),
             "less_or_equal_than": LT(),
             "as_similar_as": Similar()
             }
