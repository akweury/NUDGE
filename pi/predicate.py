# Created by jing at 28.11.23
import torch

pass_th = 0.8


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
        try:
            if var < th and (1 - mean) < th:
                satisfy = True
                self.p_bound['min'] = torch.round(mean - var, decimals=2)
                self.p_bound['max'] = torch.round(mean + var, decimals=2)
        except RuntimeError:
            print("Warning:")

        return satisfy

    def refine_space(self, t1, t2):
        pass

    def update_space(self, t1, t2):
        return

    def eval(self, t1, t2, objs):
        satisfy = torch.gt(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        satisfy_percent = satisfy.sum() / len(satisfy)
        if satisfy_percent > 0.9:
            satisfy = True
        else:
            satisfy = False
        return satisfy

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

    def expand_space(self, t1, t2):
        pass


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

    def refine_space(self, t1, t2):
        pass

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

    def eval(self, t1, t2, objs):
        satisfy = torch.lt(t1, t2).float().bool()
        p_values = torch.zeros(size=satisfy.size())
        satisfy = satisfy.prod()

        return satisfy

    def expand_space(self, t1, t2):
        pass


class Similar():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.max_dist = 20
        self.dist_num = 40
        self.name = None
        self.unit_dist = self.max_dist / self.dist_num
        self.p_bound = {"min": 0, "max": 0}
        self.dist = None
        self.score = None

    def fit(self, t1, t2, objs):

        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))

        satisfy_percents = []
        for d_i in range(self.dist_num):
            satisfy = (self.unit_dist * (d_i + 1) > dist) * (dist > self.unit_dist * d_i)
            satisfy_percent = satisfy.sum() / len(satisfy)
            satisfy_percents.append(satisfy_percent)

        highest_percent = torch.tensor(satisfy_percents).max()
        highest_dist = torch.tensor(satisfy_percents).argmax()
        if highest_percent > pass_th:
            pred_satisfy = True

            self.name = f"is_{highest_dist}_away_from"
            self.dist = highest_dist
            self.score = highest_percent
            self.p_bound["min"] = self.unit_dist * highest_dist
            self.p_bound["max"] = self.unit_dist * (highest_dist + 1)
        else:
            pred_satisfy = False

        return pred_satisfy

    def eval(self, t1, t2, objs):
        if self.name is None:
            return False

        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))

        d_i = self.dist
        satisfy = (self.unit_dist * (d_i + 1) > dist) * (dist > self.unit_dist * d_i)
        satisfy_percent = satisfy.sum() / len(satisfy)

        if satisfy_percent > pass_th:
            pred_satisfy = True
        else:
            pred_satisfy = False

        return pred_satisfy


class At_Least():
    """ generate one micro-program """

    def __init__(self):
        super().__init__()
        self.max_dist = 20
        self.dist_num = 40
        self.unit_dist = self.max_dist / self.dist_num
        self.p_bound = {"min": 0.0, "max": "infinity"}
        self.name = None
        self.dist = None
        self.score = None

    def fit(self, t1, t2, objs):

        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))

        pred_satisfy = False
        for d_i in range(self.dist_num - 1, -1, -1):
            satisfy = dist > self.unit_dist * d_i
            satisfy_percent = satisfy.sum() / len(satisfy)
            if satisfy_percent > pass_th:
                pred_satisfy = True
                self.name = f"is_at_least_{d_i}_away_from"
                self.dist = d_i
                self.score = satisfy_percent
                self.p_bound["min"] = self.unit_dist * d_i
                break

        return pred_satisfy

    def eval(self, t1, t2, objs):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))

        d_i = self.dist
        if d_i is None:
            return False
        pred_satisfy = dist > self.unit_dist * d_i

        satisfy_percent = pred_satisfy.sum() / len(pred_satisfy)

        if satisfy_percent > pass_th:
            pred_satisfy = True
        else:
            pred_satisfy = False

        return pred_satisfy


class At_Most():
    """ generate one micro-program """

    def __init__(self):
        super().__init__()
        self.max_dist = 20
        self.dist_num = 40
        self.unit_dist = self.max_dist / self.dist_num
        self.p_bound = {"min": 0.0, "max": "infinity"}
        self.name = None
        self.dist = None
        self.score = None

    def fit(self, t1, t2, objs):

        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))

        pred_satisfy = False
        for d_i in range(1, self.dist_num + 1):
            satisfy = dist < self.unit_dist * d_i
            satisfy_percent = satisfy.sum() / len(satisfy)
            if satisfy_percent > pass_th:
                pred_satisfy = True
                self.name = f"is_at_most_{d_i}_away_from"
                self.dist = d_i
                self.score = satisfy_percent
                self.p_bound["min"] = self.unit_dist * d_i
                break

        return pred_satisfy

    def eval(self, t1, t2, objs):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))

        d_i = self.dist
        if d_i is None:
            return False
        pred_satisfy = dist < self.unit_dist * d_i

        satisfy_percent = pred_satisfy.sum() / len(pred_satisfy)

        if satisfy_percent > pass_th:
            pred_satisfy = True
        else:
            pred_satisfy = False

        return pred_satisfy


class Dist():
    """ generate one micro-program
    """

    def __init__(self, dist_factor, dist_num, max_dist):
        super().__init__()
        self.name = f"is_{dist_factor}_away_from_"
        self.dist_factor = dist_factor
        self.dist_num = dist_num
        self.max_dist = max_dist
        self.unit_dist = max_dist / self.dist_num

    def eval(self, t1, t2, objs):

        # repeat situation
        if objs[1] < objs[0] or t1.sum() == 0 or t2.sum() == 0:
            return False

        if len(t1) == 1:
            dist = torch.tensor(0), torch.abs(torch.sub(t1, t2))
        else:
            dist = torch.abs(torch.sub(t1, t2))

        satisfy = (self.unit_dist * (self.dist_factor + 1) > dist) * (dist > self.unit_dist * self.dist_factor)
        satisfy_percent = satisfy.sum() / len(satisfy)
        if satisfy_percent > pass_th:
            pred_satisfy = True
        else:
            pred_satisfy = False
        return pred_satisfy


def get_preds(repeats=1):
    all_preds = []
    for i in range(repeats):
        # distance_preds = get_dist_preds(repeats)
        all_preds += [GT(), LT(), Similar(), At_Least(), At_Most()]
    return all_preds
