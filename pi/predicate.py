# Created by jing at 28.11.23
import torch
class Ge():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {}
        self.p_space = torch.zeros(1)
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

    def eval(self, t1, t2):
        satisfy = torch.ge(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        return satisfy, p_values

    def update_p_space(self, p_space):
        pass


class Similar():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {}
        self.p_space = torch.zeros(1)
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

    def update_p_space(self, p_space):
        self.p_space = p_space


    def eval(self, t1, t2):
        p_values = torch.round(torch.abs(torch.sub(t1, t2)), decimals=2)
        satisfy = torch.zeros(p_values.size(), dtype=torch.bool)
        for v_i, value in enumerate(p_values):
            if value in self.p_space:
                satisfy[v_i] = True

        return satisfy, p_values


def similar(t1, t2, sr, p_values, batch_data=True, avg_data=True, given_parameters=None):
    p_bound = {'min': 0, 'max': 0}
    p_values = torch.zeros(1)
    # repeat situation
    if sr[1] < sr[0] or t1.sum() == 0 or t2.sum() == 0:
        return False, p_bound, p_values

    th = 0.6
    if avg_data:
        # If x distance between A and B is similar for all
        var, mean = torch.var_mean(torch.abs(torch.sub(t1, t2)))

        satisfy = False
        if torch.abs(var / mean) < th:
            satisfy = True
            p_bound = {'min': torch.zeros(1),
                       'max': torch.round(mean + var * 0.5, decimals=2)}
    else:
        dist = torch.abs(torch.sub(t1, t2))
        p_values = dist

        t1_t2 = torch.cat((t1.unsqueeze(1), t2.unsqueeze(1)), 1)

        # if no parameter is given, predict in a heuristic way
        if given_parameters is None:
            satisfy = torch.abs(dist / t1_t2.min(dim=1)[0]) < th  # might be not accurate, has to be improved
        else:
            satisfy = dist < given_parameters["max"]

    return satisfy, p_bound, p_values


preds = [Ge(), Similar()]
pred_dict = {"greater_or_equal_than": Ge(),
             "as_similar_as": Similar()}
