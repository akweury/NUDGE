# Created by jing at 28.11.23
import torch
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split

from pi.utils import draw_utils, math_utils
from pi.neural import nn_model

pass_th = 0.8


# class LT():
#     """ generate one micro-program
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.p_bound = {"min": 0, "max": 0}
#         self.name = "less_or_equal_than"
#         self.p_spaces = []
#
#     def fit_batch(self, t1, t2):
#         satisfy = torch.lt(t1, t2)
#         return satisfy
#
#     def fit(self, t1, t2, objs):
#         th = 1e-1
#         if t1.sum() == 0 or t2.sum() == 0:
#             return False
#
#         if len(t1) == 1:
#             var, mean = torch.tensor(0), torch.lt(t1, t2).float()
#         else:
#             var, mean = torch.var_mean(torch.lt(t1, t2).float())
#
#         satisfy = False
#         if var < th and (1 - mean) < th:
#             satisfy = True
#             self.p_bound['min'] = torch.round(mean - var, decimals=2)
#             self.p_bound['max'] = torch.round(mean + var, decimals=2)
#
#         return satisfy
#
#     def refine_space(self, t1, t2):
#         pass
#
#     def update_space(self, t1, t2):
#         return
#
#     def eval_batch(self, t1, t2, p_space):
#         th = 1e-1
#         if t1.sum() == 0 or t2.sum() == 0:
#             return False
#         if len(t1) == 1:
#             var, mean = torch.tensor(0), torch.lt(t1, t2).float()
#         else:
#             var, mean = torch.var_mean(torch.lt(t1, t2).float())
#
#         satisfy = False
#         if var < th and (1 - mean) < th:
#             satisfy = True
#
#         return satisfy
#
#     def eval(self, t1, t2):
#         satisfy = torch.lt(t1, t2).float().bool()
#         # p_values = torch.zeros(size=satisfy.size())
#         # satisfy = satisfy.prod()
#
#         return satisfy
#
#     def expand_space(self, t1, t2):
#         pass


# class Similar():
#     """ generate one micro-program
#     """
#
#     def __init__(self):
#         super().__init__()
#         self.max_dist = 20
#         self.dist_num = 40
#         self.name = None
#         self.unit_dist = self.max_dist / self.dist_num
#         self.p_bound = {"min": 0, "max": 0}
#         self.dist = None
#         self.score = None
#
#     def fit_batch(self, t1, t2):
#         t1 = t1.reshape(-1)
#         t2 = t2.reshape(-1)
#         dist = torch.abs(torch.sub(t1, t2))
#         pred_satisfy = torch.zeros(len(t1), dtype=torch.bool)
#
#         dist_satisfactions = torch.zeros(len(t1), self.dist_num)
#         for d_i in range(self.dist_num):
#             satisfy = (self.unit_dist * (d_i + 1) > dist) * (dist > self.unit_dist * d_i)
#             dist_satisfactions[:, d_i] = satisfy
#
#         highest_percent = torch.tensor(satisfy_percents).max()
#         highest_dist = torch.tensor(satisfy_percents).argmax()
#         if highest_percent > pass_th:
#             pred_satisfy = True
#
#             self.name = f"is_{highest_dist}_away_from"
#             self.dist = highest_dist
#             self.score = highest_percent
#             self.p_bound["min"] = self.unit_dist * highest_dist
#             self.p_bound["max"] = self.unit_dist * (highest_dist + 1)
#         else:
#             pred_satisfy = False
#         return pred_satisfy
#
#     def fit(self, t1, t2, objs):
#
#         t1 = t1.reshape(-1)
#         t2 = t2.reshape(-1)
#         dist = torch.abs(torch.sub(t1, t2))
#
#         satisfy_percents = []
#         for d_i in range(self.dist_num):
#             satisfy = (self.unit_dist * (d_i + 1) > dist) * (dist > self.unit_dist * d_i)
#             satisfy_percent = satisfy.sum() / len(satisfy)
#             satisfy_percents.append(satisfy_percent)
#
#         highest_percent = torch.tensor(satisfy_percents).max()
#         highest_dist = torch.tensor(satisfy_percents).argmax()
#         if highest_percent > pass_th:
#             pred_satisfy = True
#
#             self.name = f"is_{highest_dist}_away_from"
#             self.dist = highest_dist
#             self.score = highest_percent
#             self.p_bound["min"] = self.unit_dist * highest_dist
#             self.p_bound["max"] = self.unit_dist * (highest_dist + 1)
#         else:
#             pred_satisfy = False
#
#         return pred_satisfy
#
#     def eval(self, t1, t2):
#         if self.name is None:
#             return False
#
#         t1 = t1.reshape(-1)
#         t2 = t2.reshape(-1)
#         dist = torch.abs(torch.sub(t1, t2))
#
#         d_i = self.dist
#         satisfy = (self.unit_dist * (d_i + 1) > dist) * (dist > self.unit_dist * d_i)
#         satisfy_percent = satisfy.sum() / len(satisfy)
#
#         if satisfy_percent > pass_th:
#             pred_satisfy = True
#         else:
#             pred_satisfy = False
#
#         return pred_satisfy

#
# class At_Least():
#     """ generate one micro-program """
#
#     def __init__(self, dist_factor, dist_num, max_dist):
#         super().__init__()
#         self.unit_dist = max_dist / dist_num
#         self.name = f"is_at_least_{dist_factor}_away_from"
#         self.dist = dist_factor
#         self.p_bound = {"min": self.unit_dist * dist_factor, "max": "infinity"}
#
#     def eval(self, t1, t2):
#         t1 = t1.reshape(-1)
#         t2 = t2.reshape(-1)
#         dist = torch.abs(torch.sub(t1, t2))
#         pred_satisfy = dist > self.unit_dist * self.dist
#         return pred_satisfy
#
#
# class At_Most():
#     """ generate one micro-program """
#
#     def __init__(self, dist_factor, dist_num, max_dist):
#         super().__init__()
#         self.unit_dist = max_dist / dist_num
#         self.name = f"is_at_most_{dist_factor}_away_from"
#         self.dist = dist_factor
#         self.p_bound = {"min": 0.0, "max": self.unit_dist * dist_factor}
#
#     def eval(self, t1, t2):
#         t1 = t1.reshape(-1)
#         t2 = t2.reshape(-1)
#         dist = torch.abs(torch.sub(t1, t2))
#         pred_satisfy = dist < self.unit_dist * self.dist
#         return pred_satisfy
#

class Skill_Dist_Dir():
    """ generate one micro-program
    """

    def __init__(self, args, x_range, x_conf, y_range, y_conf, dir_range, dir_conf, name, plot_path, skill_len):
        super().__init__()
        self.args = args
        self.x_range = [x_range[i].to(args.device) for i in range(skill_len)]
        self.y_range = [y_range[i].to(args.device) for i in range(skill_len)]
        self.x_conf = [x_conf[i].to(args.device) for i in range(skill_len)]
        self.y_conf = [y_conf[i].to(args.device) for i in range(skill_len)]
        self.dir_range = [dir_range[i].to(args.device) for i in range(skill_len)]
        self.dir_conf = [dir_conf[i].to(args.device) for i in range(skill_len)]
        self.model = None

        self.name = f"{name}"
        self.plot_path = plot_path

        self.y_0 = 0
        self.y_1 = 1

    def eval(self, t1, t2, actions, beh_type, skill_stage):
        action = self.args.action_names[actions[skill_stage]]
        direction = torch.tensor([math_utils.action_to_deg(action)] * t2.shape[1]).to(t2.device)
        t1_move_one_step = torch.repeat_interleave(math_utils.one_step_move(t1[0], direction[0], self.args.step_dist),
                                                   t2.shape[1], dim=0)
        assert t2.shape[0] == 1
        dists = math_utils.dist_a_and_b(t1_move_one_step, t2[0])
        dists_aligned = math_utils.closest_one_percent(dists)
        input_t1 = t1_move_one_step[0:1]
        input_t2 = t2[0]
        dirs = math_utils.dir_a_and_b_with_alignment(input_t1, input_t2).to(t2.device)

        conf_dir = torch.zeros(len(input_t2)).to(t2.device)
        conf_x = torch.zeros(len(input_t2)).to(t2.device)
        conf_y = torch.zeros(len(input_t2)).to(t2.device)
        mask_dir = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        mask_x = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        mask_y = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        mask = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        for i in range(t2.shape[1]):
            if dirs[i] in self.dir_range[skill_stage]:
                mask_dir[i] = True
                conf_dir[i] = self.dir_conf[skill_stage][self.dir_range[skill_stage] == dirs[i]]

            if beh_type != "path_finding":
                if dists_aligned[i][0] in self.x_range[skill_stage]:
                    mask_x[i] = True
                    conf_x[i] = self.x_conf[skill_stage][self.x_range[skill_stage] == dists_aligned[i][0]]
                if dists_aligned[i][1] in self.y_range[skill_stage]:
                    mask_y[i] = True
                    conf_y[i] = self.y_conf[skill_stage][self.y_range[skill_stage] == dists_aligned[i][1]]

                mask[i] = mask_x[i] * mask_y[i] * mask_dir[i]
            else:
                mask[i] = mask_dir[i]

        conf_dir[~mask_dir] = 0
        conf_x[~mask_x] = 0
        conf_y[~mask_y] = 0

        # Use the trained model to predict the new value
        valid_conf = ((conf_dir + conf_x + conf_y) / 3)[mask]
        if len(valid_conf) > 0:
            pred_conf = valid_conf.max()
        else:
            pred_conf = 0
        # satisfactions = new_value_prediction.argmax(dim=1) == self.y_0

        return pred_conf


class Dist_Closest():
    """ generate one micro-program
    """

    def __init__(self, args, x_range, x_conf, y_range, y_conf, dir_range, dir_conf, name, plot_path):
        super().__init__()
        self.args = args
        self.x_range = x_range.to(args.device)
        self.y_range = y_range.to(args.device)
        self.x_conf = x_conf.to(args.device)
        self.y_conf = y_conf.to(args.device)
        self.dir_range = dir_range.to(args.device)
        self.dir_conf = dir_conf.to(args.device)
        self.model = None

        self.name = f"{name}"
        self.plot_path = plot_path

        self.y_0 = 0
        self.y_1 = 1

    # def gen_data(self):
    #     X_0 = self.X_0
    #     # generate enough data
    #     if len(self.X_1) > len(X_0):
    #         try:
    #             generated_points = nn_model.generate_data(X_0, gen_num=len(self.X_1) - len(X_0))
    #         except:
    #             X_0 += torch.rand(X_0.shape) * 0.001
    #             generated_points = nn_model.generate_data(X_0, gen_num=len(self.X_1) - len(X_0))
    #         X_0 = torch.cat((X_0, generated_points), dim=0)
    #
    #     # prepare training data
    #     X = torch.cat((X_0, self.X_1), dim=0)
    #     # y = torch.zeros(len(X), 2)
    #
    #     y = torch.cat(
    #         (torch.ones(len(X_0), dtype=torch.bool), torch.zeros(len(self.X_1), dtype=torch.bool)), dim=0)
    #     return X, y

    # def fit_pred(self):
    #     X, y = s/elf.gen_data()
    # fit a classifier using neural network
    # self.model = nn_model.fit_classifier(x_tensor=X, y_tensor=y,
    #                                      num_epochs=self.num_epochs, device=self.args.device,
    #                                      classifier_type=self.name, plot_path=self.plot_path)

    # Generate sample 3D data (replace this with your actual data)
    # Scatter plot for 3D data

    # Scatter plot for 2D data
    # from sklearn.mixture import GaussianMixture
    # n_components = 3  # Number of Gaussian components (you can adjust this)
    # self.model = GaussianMixture(n_components=n_components, random_state=42)
    # training_data = self.X_0
    # self.model.fit(training_data)
    #
    # # Plot the training data
    # plt.scatter(training_data[:, 0], training_data[:, 2], alpha=0.5)
    #
    # # Plot Gaussian components
    # for mean, cov in zip(gmm.means_, gmm.covariances_):
    #     v, w = np.linalg.eigh(cov)
    #     v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    #     u = w[0] / np.linalg.norm(w[0])
    #     angle = np.arctan(u[1] / u[0])
    #     angle = 180.0 * angle / np.pi  # convert to degrees
    #     ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[2], 180.0 + angle, color='red', alpha=0.5)
    #     plt.gca().add_patch(ell)
    #
    # plt.title('Gaussian Mixture Model')
    # plt.show()
    #
    # print("")

    # if self.args.with_explain:
    #     pos_indices = torch.argmax(y, dim=1) == 0
    #     pos_data = self.X_0[pos_indices]
    # neg_data = X[~pos_indices]

    # data = [[pos_data[:, 0], neg_data[:, 0]],
    #         [pos_data[:, 1], neg_data[:, 1]],
    #         [pos_data[:, 2], neg_data[:, 2]]
    #         ]
    # draw_utils.plot_histogram(data, [[["x_pos", "x_neg"]], [["y_pos", "y_neg"]], [["dir_pos", "dir_neg"]]],
    #                           self.name, self.plot_path, figure_size=(20, 10))
    def eval_o2o(self, t1, t2, action, beh_type, skill_stage=None):
        direction = torch.tensor([math_utils.action_to_deg(self.args.action_names[action])] * t2.shape[1]).to(t2.device)
        t1_move_one_step = torch.repeat_interleave(math_utils.one_step_move_o2o(t1, direction[0], self.args.step_dist),
                                                   t2.shape[1], dim=1)
        dists = math_utils.dist_a_and_b(t1_move_one_step, t2)
        dists_aligned = math_utils.closest_one_percent(dists)

        dirs = math_utils.dir_a_and_b_with_alignment_o2o(t1_move_one_step, t2).to(t2.device)

        conf_dir = torch.zeros(t2.shape[1], t2.shape[0]).to(t2.device)
        conf_x = torch.zeros(t2.shape[1], t2.shape[0]).to(t2.device)
        conf_y = torch.zeros(t2.shape[1], t2.shape[0]).to(t2.device)
        mask_dir = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        mask_x = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        mask_y = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        mask = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        for i in range(t2.shape[1]):
            self.dir_range = self.dir_range.to(t2.device)
            result_tensor = torch.eq(dirs[i, 0, :], self.dir_range)
            # result= torch.any(result_tensor, dim=0)
            mask_dir[i, result_tensor] = True
            dir_weight = torch.repeat_interleave(self.dir_conf.reshape(1, -1), dirs.shape[2], dim=1)

            conf_dir[i, result_tensor] = dir_weight[self.dir_range == dirs[i]]
            mask[i] = mask_dir[i]

        conf_dir[~mask_dir] = 0
        conf_x[~mask_x] = 0
        conf_y[~mask_y] = 0

        # Use the trained model to predict the new value
        conf = ((conf_dir + conf_x + conf_y) / 3)
        pred_conf = torch.zeros((conf_dir.shape[1])).to(t2.device)
        for i in range(conf_dir.shape[1]):
            if len(conf[mask[:, i], i]) > 0:
                pred_conf[i] = conf[mask[:, i], i].max()
            else:
                pred_conf[i] = 0
        # satisfactions = new_value_prediction.argmax(dim=1) == self.y_0

        return pred_conf

    def eval(self, t1, t2, action, beh_type, skill_stage=None):
        direction = torch.tensor([math_utils.action_to_deg(self.args.action_names[action])] * t2.shape[1]).to(t2.device)
        t1_move_one_step = torch.repeat_interleave(math_utils.one_step_move(t1[0], direction[0], self.args.step_dist),
                                                   t2.shape[1], dim=0)
        assert t2.shape[0] == 1
        dists = math_utils.dist_a_and_b(t1_move_one_step, t2[0])
        dists_aligned = math_utils.closest_one_percent(dists)
        input_t1 = t1_move_one_step[0:1]
        input_t2 = t2[0]
        dirs = math_utils.dir_a_and_b_with_alignment(input_t1, input_t2).to(t2.device)

        conf_dir = torch.zeros(len(input_t2)).to(t2.device)
        conf_x = torch.zeros(len(input_t2)).to(t2.device)
        conf_y = torch.zeros(len(input_t2)).to(t2.device)
        mask_dir = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        mask_x = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        mask_y = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        mask = torch.zeros(len(input_t2), dtype=torch.bool).to(t2.device)
        for i in range(t2.shape[1]):
            if dirs[i] in self.dir_range:
                mask_dir[i] = True
                conf_dir[i] = self.dir_conf[self.dir_range == dirs[i]]

            if beh_type != "path_finding":
                if dists_aligned[i][0] in self.x_range:
                    mask_x[i] = True
                    conf_x[i] = self.x_conf[self.x_range == dists_aligned[i][0]]
                if dists_aligned[i][1] in self.y_range:
                    mask_y[i] = True
                    conf_y[i] = self.y_conf[self.y_range == dists_aligned[i][1]]

                mask[i] = mask_x[i] * mask_y[i] * mask_dir[i]
            else:
                mask[i] = mask_dir[i]

        conf_dir[~mask_dir] = 0
        conf_x[~mask_x] = 0
        conf_y[~mask_y] = 0

        # Use the trained model to predict the new value
        valid_conf = ((conf_dir + conf_x + conf_y) / 3)[mask]
        if len(valid_conf) > 0:
            pred_conf = valid_conf.max()
        else:
            pred_conf = 0
        # satisfactions = new_value_prediction.argmax(dim=1) == self.y_0

        return pred_conf


class O2ODir():
    """ generate one micro-program
    """

    def __init__(self, args, x_type, y_type, dir_type, name, plot_path):
        super().__init__()
        self.args = args
        self.x_type = x_type.to(args.device)
        self.y_type = y_type.to(args.device)
        self.dir_type = dir_type.to(args.device)
        self.name = f"{name}"
        self.plot_path = plot_path

    def eval_o2o(self, t1, t2, action, beh_type, skill_stage=None):
        direction = torch.tensor([math_utils.action_to_deg(self.args.action_names[action])] * t2.shape[1]).to(t2.device)
        t1_move_one_step = torch.repeat_interleave(math_utils.one_step_move_o2o(t1, direction[0], self.args.step_dist),
                                                   t2.shape[1], dim=1)
        dists = math_utils.dist_a_and_b(t1_move_one_step, t2)
        dists_aligned = math_utils.closest_one_percent(dists, 0.1).permute(1,2,0)

        dirs = math_utils.dir_a_and_b_with_alignment_o2o(t1_move_one_step, t2).to(t2.device)

        # mask_dir = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        # mask_x = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        # mask_y = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        mask = torch.zeros(t2.shape[1], t2.shape[0], dtype=torch.bool).to(t2.device)
        for i in range(t2.shape[1]):
            mask_dir_eq = torch.eq(dirs[i, 0, :], self.dir_type)
            mask_x_eq = torch.eq(dists_aligned[i, 0, :], self.x_type)
            mask_y_eq = torch.eq(dists_aligned[i, 1, :], self.y_type)
            # result= torch.any(result_tensor, dim=0)

            # dir_weight = torch.repeat_interleave(self.dir_type.reshape(1, -1), dirs.shape[2], dim=1)
            mask[i] = mask_dir_eq * mask_x_eq * mask_y_eq
        mask = mask.prod(dim=0)
        return mask


def o2o_model(move_direction, p1, p2, step_dist, dir_types, x_types, y_types):
    p1_moved = math_utils.one_step_move_o2o(p1, move_direction, step_dist)
    dists = math_utils.dist_a_and_b(p1_moved, p2)
    dists = math_utils.closest_one_percent(dists, 0.1).permute(1,2,0)
    dirs = math_utils.dir_a_and_b_with_alignment(p1,p2)

    mask_dir_eq = torch.eq(dirs[:,0,:], dir_types)
    mask_x_eq = torch.eq(dists[:,0,:], x_types)
    mask_y_eq = torch.eq(dists[:,1,:], y_types)
    mask = mask_dir_eq * mask_x_eq * mask_y_eq
    return mask


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
