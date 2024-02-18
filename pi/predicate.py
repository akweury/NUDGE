# Created by jing at 28.11.23
import torch
import numpy as np

from pi.utils import draw_utils, math_utils
from pi.neural import nn_model

pass_th = 0.8


class LT():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "less_or_equal_than"
        self.p_spaces = []

    def fit_batch(self, t1, t2):
        satisfy = torch.lt(t1, t2)
        return satisfy

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

    def eval(self, t1, t2):
        satisfy = torch.lt(t1, t2).float().bool()
        # p_values = torch.zeros(size=satisfy.size())
        # satisfy = satisfy.prod()

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

    def fit_batch(self, t1, t2):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))
        pred_satisfy = torch.zeros(len(t1), dtype=torch.bool)

        dist_satisfactions = torch.zeros(len(t1), self.dist_num)
        for d_i in range(self.dist_num):
            satisfy = (self.unit_dist * (d_i + 1) > dist) * (dist > self.unit_dist * d_i)
            dist_satisfactions[:, d_i] = satisfy

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

    def eval(self, t1, t2):
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

    def __init__(self, dist_factor, dist_num, max_dist):
        super().__init__()
        self.unit_dist = max_dist / dist_num
        self.name = f"is_at_least_{dist_factor}_away_from"
        self.dist = dist_factor
        self.p_bound = {"min": self.unit_dist * dist_factor, "max": "infinity"}

    def eval(self, t1, t2):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))
        pred_satisfy = dist > self.unit_dist * self.dist
        return pred_satisfy


class At_Most():
    """ generate one micro-program """

    def __init__(self, dist_factor, dist_num, max_dist):
        super().__init__()
        self.unit_dist = max_dist / dist_num
        self.name = f"is_at_most_{dist_factor}_away_from"
        self.dist = dist_factor
        self.p_bound = {"min": 0.0, "max": self.unit_dist * dist_factor}

    def eval(self, t1, t2):
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        dist = torch.abs(torch.sub(t1, t2))
        pred_satisfy = dist < self.unit_dist * self.dist
        return pred_satisfy


class Dist_Closest():
    """ generate one micro-program
    """

    def __init__(self, args, X_0, X_1, name, plot_path):
        super().__init__()
        self.args = args
        self.X_0 = X_0
        self.X_1 = X_1
        self.model = None
        self.num_epochs = args.train_nn_epochs
        self.var, self.mean = torch.var_mean(X_0, dim=0)
        self.var = self.var.sum()
        self.mean = self.mean.sum()
        self.name = f"{name}_ep_{self.num_epochs}_var_{self.var:.1f}_mean_{self.mean:.1f}"
        self.plot_path = plot_path

        self.y_0 = 0
        self.y_1 = 1

    def add_item(self, x):
        self.X_0 = torch.cat((self.X_0, x), dim=0)

    def gen_data(self):
        X_0 = self.X_0
        # generate enough data
        if len(self.X_1) > len(X_0):
            try:
                generated_points = nn_model.generate_data(X_0, gen_num=len(self.X_1) - len(X_0))
            except:
                X_0 += torch.rand(X_0.shape) * 0.001
                generated_points = nn_model.generate_data(X_0, gen_num=len(self.X_1) - len(X_0))
            X_0 = torch.cat((X_0, generated_points), dim=0)

        # prepare training data
        X = torch.cat((X_0, self.X_1), dim=0)
        y = torch.zeros(len(X), 2)

        X_0_indices = torch.cat(
            (torch.ones(len(X_0), dtype=torch.bool), torch.zeros(len(self.X_1), dtype=torch.bool)), dim=0)
        y[X_0_indices, self.y_0] = 1
        y[~X_0_indices, self.y_1] = 1
        return X, y

    def fit_pred(self):
        X, y = self.gen_data()
        # fit a classifier using neural network
        self.model = nn_model.fit_classifier(x_tensor=X, y_tensor=y,
                                             num_epochs=self.num_epochs, device=self.args.device,
                                             classifier_type=self.name, plot_path=self.plot_path)
        if self.args.with_explain:
            pos_indices = torch.argmax(y, dim=1) == 0
            pos_data = X[pos_indices]
            neg_data = X[~pos_indices]

            data = [[pos_data[:, 0], neg_data[:, 0]],
                    [pos_data[:, 1], neg_data[:, 1]],
                    [pos_data[:, 2], neg_data[:, 2]]
                    ]
            # draw_utils.plot_histogram(data, [[["x_pos", "x_neg"]], [["y_pos", "y_neg"]], [["dir_pos", "dir_neg"]]],
            #                           self.name, self.plot_path, figure_size=(20, 10))

    def eval(self, t1, t2, action):
        direction = torch.tensor([math_utils.action_to_deg(self.args.action_names[action])] * t2.shape[1]).to(t2.device)
        t1_move_one_step = torch.repeat_interleave(math_utils.one_step_move(t1[0], direction[0], self.args.step_dist),
                                                   t2.shape[1], dim=0)
        assert t2.shape[0] == 1
        dist = math_utils.dist_a_and_b(t1_move_one_step, t2[0])
        dir = [math_utils.dir_a_and_b(t1_move_one_step[i], t2[0, i]).item() for i in range(t2.shape[1])]
        dir = torch.tensor(dir).unsqueeze(1).to(t2.device)
        dist_dir = torch.cat((dist, dir), dim=1)
        # Use the trained model to predict the new value
        new_value_prediction = self.model(dist_dir).detach()
        satisfactions = new_value_prediction.argmax(dim=1) == self.y_0
        satisfaction = satisfactions.sum() > 0
        return satisfaction


class GT():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "greater_or_equal_than"
        self.p_spaces = []

    def eval(self, t1, t2):
        satisfy = torch.gt(t1, t2)
        return satisfy


class GT_Closest():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.name = "greater_or_equal_than_closest"

    def eval(self, t1, t2):
        closest_indices = torch.abs(t2 - t1).argmin(dim=1)
        t2_closest = torch.tensor([t2[i, v_i] for i, v_i in enumerate(closest_indices)])
        satisfy = torch.gt(t1.squeeze(), t2_closest.squeeze())
        return satisfy
