# Created by jing at 28.11.23
import torch

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pass_th = 0.8


class GT():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "greater_or_equal_than"
        self.p_spaces = []

    def fit_batch(self, t1, t2):
        satisfy = torch.gt(t1, t2)
        return satisfy

    def fit(self, t1, t2, objs):
        satisfy = torch.zeros(len(t1), dtype=torch.bool)
        th = 1e-1

        if t1.sum() == 0 or t2.sum() == 0:
            return satisfy

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

    def eval(self, t1, t2):
        satisfy = torch.gt(t1, t2).float().bool()
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


class Dist():
    """ generate one micro-program
    """

    def __init__(self, data, data_pos, var, mean):
        super().__init__()
        self.name = f"distance_var_{var:.2f}_mean_{mean:.2f}"
        self.data = data
        self.data_pos = data_pos
        self.model = None
        self.fit()

    def fit(self):
        X = self.data + self.data_pos
        y = [1] * len(self.data) + [0] * len(self.data_pos)

        # Assume X is your feature matrix, and y is the corresponding labels (Group A or not)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a logistic regression model
        model = LogisticRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")

        self.model = model

    def eval(self, t1, t2):
        dist = torch.abs(torch.sub(t1, t2))
        # Use the trained model to predict the new value
        new_value_prediction = self.model.predict(dist.reshape(1, -1))
        print(f"Prediction for the new value: {new_value_prediction}")
        return new_value_prediction


class GT():
    """ generate one micro-program
    """

    def __init__(self):
        super().__init__()
        self.p_bound = {"min": 0, "max": 0}
        self.name = "greater_or_equal_than"
        self.p_spaces = []

    def eval(self, t1, t2):
        satisfy = torch.gt(t1, t2).float()
        return satisfy
