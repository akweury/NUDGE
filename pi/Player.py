# Created by jing at 18.01.24
import os
import random
import torch
from torch import nn as nn
from torch.distributions import Categorical

from src.agents.neural_agent import ActorCritic, NeuralPPO
from src.agents.utils_getout import extract_neural_state_getout
from pi import Reasoner
from pi.utils import smp_utils, beh_utils, file_utils, oc_utils, reason_utils
from pi.utils import game_utils, draw_utils


class SymbolicMicroProgramModel(nn.Module):
    def __init__(self, args, rng=None):
        super(SymbolicMicroProgramModel, self).__init__()
        self.rng = random.Random() if rng is None else rng
        self.args = args
        self.actor = Reasoner.SmpReasoner(args)

    def forward(self):
        raise NotImplementedError

    def act(self, logic_state, epsilon=0.0):
        action_probs = self.actor(logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.args.device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        action_probs = self.actor(logic_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()


class SymbolicMicroProgramPlayer:
    def __init__(self, args):
        self.args = args
        self.preds = None
        self.model = SymbolicMicroProgramModel(args).actor.to(args.device)
        self.position_norm_factor = None
        self.prop_indices = None
        self.def_behaviors = []
        self.att_behaviors = []
        self.pf_behaviors = []
        self.skill_att_behaviors = []
        self.o2o_behaviors = []
        self.win_states = []
        self.win_actions = []
        self.win_rewards = []
        self.score_states = []
        self.score_rewards = []
        self.score_actions = []
        self.lost_states = []
        self.lost_actions = []
        self.lost_rewards = []
        self.lost_game_ids = []

        self.learned_jump = False
        self.learn_jump_at = None
        self.win_two_in_a_row = None
        self.win_three_in_a_row = None
        self.win_five_in_a_row = None

    def load_atari_buffer(self, args):
        if os.path.exists(args.buffer_tensor_filename):
            if args.device != "cpu":
                args.device = f"cuda:{args.device}"
            data = torch.load(args.buffer_tensor_filename, map_location=args.device)
            self.states = data["states"].to(self.args.device)
            self.actions = data["actions"].to(self.args.device)
            self.rewards = data["rewards"].to(self.args.device)

            self.win_states = data["win_states"].to(self.args.device)
            self.win_actions = data["win_actions"].to(self.args.device)
            self.win_rewards = data["win_rewards"].to(self.args.device)

            self.lost_states = data["lost_states"].to(self.args.device)
            self.lost_rewards = data["lost_rewards"].to(self.args.device)
            self.lost_actions = data["lost_actions"].to(self.args.device)
            self.data = {'win_states': self.win_states,
                         'win_actions': self.win_actions,
                         'win_rewards': self.win_rewards,
                         'lost_states': self.lost_states,
                         'lost_actions': self.lost_actions,
                         'lost_rewards': self.lost_rewards}

        else:
            buffer = game_utils.load_buffer(args)
            print(f'- Loaded game history : {len(buffer.logic_states)}')
            self.buffer_win_rates = buffer.win_rates
            game_num = len(buffer.actions)
            self.actions = []
            self.rewards = []
            self.states = []
            for g_i in range(game_num):
                self.actions += buffer.actions[g_i]
                self.rewards += buffer.rewards[g_i]
                self.states += buffer.logic_states[g_i].tolist()
                if self.buffer_win_rates[g_i] > 0:
                    self.win_actions += buffer.actions[g_i]
                    self.win_rewards += buffer.rewards[g_i]
                    self.win_states += buffer.logic_states[g_i].tolist()
                elif self.buffer_win_rates[g_i] < 0:
                    self.lost_actions += buffer.actions[g_i]
                    self.lost_rewards += buffer.rewards[g_i]
                    self.lost_states += buffer.logic_states[g_i].tolist()
                else:
                    raise ValueError
            self.states = torch.tensor(self.states).to(self.args.device)
            self.actions = torch.tensor(self.actions).to(self.args.device)
            self.rewards = torch.tensor(self.rewards).to(self.args.device)
            self.win_states = torch.tensor(self.win_states).to(self.args.device)
            self.win_actions = torch.tensor(self.win_actions).to(self.args.device)
            self.win_rewards = torch.tensor(self.win_rewards).to(self.args.device)
            self.lost_states = torch.tensor(self.lost_states).to(self.args.device)
            self.lost_actions = torch.tensor(self.lost_actions).to(self.args.device)
            self.lost_rewards = torch.tensor(self.lost_rewards).to(self.args.device)
            self.data = {'win_states': self.win_states,
                         'win_actions': self.win_actions,
                         'win_rewards': self.win_rewards,
                         'lost_states': self.lost_states,
                         'lost_actions': self.lost_actions,
                         'lost_rewards': self.lost_rewards}
            train_data = {"states": self.states,
                          "actions": self.actions,
                          "rewards": self.rewards,
                          "win_states": self.win_states,
                          "win_actions": self.win_actions,
                          "win_rewards": self.win_rewards,
                          "lost_states": self.lost_states,
                          "lost_actions": self.lost_actions,
                          "lost_rewards": self.lost_rewards}
            torch.save(train_data, args.buffer_tensor_filename)

    def load_buffer(self, buffer):

        print(
            f'- Loaded game history, win games : {len(buffer.logic_states)}, loss games : {len(buffer.lost_logic_states)}')

        self.buffer_win_rates = buffer.win_rates

        for g_i in range(len(buffer.actions)):
            self.win_actions += buffer.actions[g_i]
            self.win_rewards += buffer.rewards[g_i]
            self.win_states += buffer.logic_states[g_i].tolist()

        for g_i in range(len(buffer.lost_actions)):
            self.lost_actions += buffer.lost_actions[g_i]
            self.lost_rewards += buffer.lost_rewards[g_i]
            self.lost_states += buffer.lost_logic_states[g_i].tolist()
            self.lost_game_ids += [g_i] * len(buffer.lost_logic_states[g_i].tolist())

        self.win_states = torch.tensor(self.win_states).to(self.args.device)
        self.win_actions = torch.tensor(self.win_actions).to(self.args.device)
        self.win_rewards = torch.tensor(self.win_rewards).to(self.args.device)
        self.lost_states = torch.tensor(self.lost_states).to(self.args.device)
        self.lost_actions = torch.tensor(self.lost_actions).to(self.args.device)
        self.lost_rewards = torch.tensor(self.lost_rewards).to(self.args.device)
        self.data = {'win_states': self.win_states,
                     'win_actions': self.win_actions,
                     'win_rewards': self.win_rewards,
                     'lost_states': self.lost_states,
                     'lost_actions': self.lost_actions,
                     'lost_rewards': self.lost_rewards}

    def update_lost_buffer(self, logic_states, actions, rewards):
        new_lost_states = torch.tensor(logic_states).squeeze()
        new_lost_actions = torch.tensor(actions)
        new_lost_rewards = torch.tensor(rewards)
        if new_lost_states.shape[1] != self.lost_states.shape[1]:
            self.lost_states = new_lost_states
            self.lost_actions = new_lost_actions
            self.lost_rewards = new_lost_rewards
        else:
            self.lost_states = torch.cat((self.lost_states, new_lost_states), 0)
            self.lost_actions = torch.cat((self.lost_actions, new_lost_actions), 0)
            self.lost_rewards = torch.cat((self.lost_rewards, new_lost_rewards), 0)

    def update_behaviors(self, pf_behaviors, def_behaviors, att_behaviors, skill_att_behavior, o2o_behaviors,
                         args=None):
        if args is not None:
            self.args = args
        if skill_att_behavior is not None:
            self.skill_att_behavior = skill_att_behavior
        if o2o_behaviors is not None:
            self.o2o_behaviors = o2o_behaviors
        if pf_behaviors is not None:
            self.pf_behaviors = pf_behaviors
        if def_behaviors is not None:
            self.def_behaviors = def_behaviors
        if att_behaviors is not None:
            self.att_behaviors = att_behaviors
        if self.pf_behaviors is None:
            self.pf_behaviors = []
        if self.def_behaviors is None:
            self.def_behaviors = []
        if self.att_behaviors is None:
            self.att_behaviors = []
        if self.skill_att_behaviors is None:
            self.skill_att_behavior = []
        if self.o2o_behaviors is None:
            self.o2o_behaviors = []
        try:
            self.behaviors = self.pf_behaviors + self.def_behaviors + self.att_behaviors + self.skill_att_behaviors + self.o2o_behaviors
        except TypeError:
            print('Type Error (update behaviors)')
        self.model.update(args, self.behaviors, o2o_data=self.o2o_data, action_delta=self.action_delta)
        #
        # weights = torch.zeros(len(self.behaviors)).to(self.args.device)
        # for beh_i, beh in enumerate(self.behaviors):
        #     weights[beh_i] = beh.weight
        # self.model.weights = weights
        # if not self.learned_jump:
        #     for beh in self.behaviors:
        #         if "jump" == beh.clause.head.pred.name:
        #             self.learned_jump = True
        #             return True

        return False

    def reasoning_def_behaviors(self, use_ckp=True, show_log=True):
        # print(f"Reasoning defensive behaviors...")
        # if no data for defensive behaviors exist
        if len(self.lost_states) == 0:
            return [], []

        neg_states_stat_file = self.args.check_point_path / "defensive" / f"defensive_stats.json"
        if use_ckp and os.path.exists(neg_states_stat_file):
            def_beh_data = file_utils.load_json(neg_states_stat_file)
        else:
            def_beh_data = smp_utils.stat_rewards(self.lost_states,
                                                  self.lost_actions,
                                                  self.lost_rewards,
                                                  self.args.zero_reward,
                                                  self.args.obj_info,
                                                  self.prop_indices,
                                                  self.args.var_th,
                                                  "defense", self.args.action_names,
                                                  self.args.step_dist, self.args.max_dist)
            for beh_i, beh_data in enumerate(def_beh_data):
                data = [[torch.tensor(beh_data["dists_pos"])[:, 0], torch.tensor(beh_data["dists_neg"])[:, 0]],
                        [torch.tensor(beh_data["dists_pos"])[:, 1], torch.tensor(beh_data["dists_neg"])[:, 1]],
                        [torch.tensor(beh_data["dir_pos"]).squeeze(), torch.tensor(beh_data["dir_ab_neg"]).squeeze()]
                        ]
                beh_name = f"{beh_i}_{self.args.action_names[beh_data['action_type']]}_var_{beh_data['variance']:.2f}"
                draw_utils.plot_histogram(data, [[["x_pos", "x_neg"]], [["y_pos", "y_neg"]], [["dir_pos", "dir_neg"]]],
                                          beh_name, self.args.check_point_path / "defensive", figure_size=(30, 10))
            file_utils.save_json(neg_states_stat_file, def_beh_data)

        neg_beh_file = self.args.check_point_path / "defensive" / f"defensive_behaviors.pkl"
        if os.path.exists(neg_beh_file):
            defense_behaviors = file_utils.load_pickle(neg_beh_file)
            defense_behaviors = beh_utils.update_negative_behaviors(self.args, defense_behaviors,
                                                                    def_beh_data)
            if self.args.with_explain:
                for def_beh in defense_behaviors:
                    print(f"# defense behavior: {def_beh.clause}")
            file_utils.save_pkl(neg_beh_file, defense_behaviors)

        else:
            defense_behaviors = []
            db_plots = []
            for beh_i, beh in enumerate(def_beh_data):
                # print(f"- Creating defense behavior {beh_i + 1}/{len(def_beh_data)}...")
                behavior = beh_utils.create_negative_behavior(self.args, beh_i, beh)
                db_plots.append({"plot_i": beh_i})
                defense_behaviors.append(behavior)
            file_utils.save_pkl(neg_beh_file, defense_behaviors)

        self.def_behaviors = defense_behaviors
        return defense_behaviors

    def reasoning_att_skills(self, use_ckp=True):
        if len(self.pos_data) == 0:
            return []
        stat_file = self.args.check_point_path / "skill_attack" / f"attack_skill_stats.json"
        if use_ckp and os.path.exists(stat_file):
            att_behavior_data = file_utils.load_json(stat_file)
        else:
            att_behavior_data = smp_utils.stat_series_rewards(self.win_states,
                                                              self.win_actions,
                                                              self.win_rewards,
                                                              self.prop_indices,
                                                              self.args, "attack")
            for beh_i, beh_data in enumerate(att_behavior_data):
                skill_len = beh_data["skill_len"]
                for frame_i in range(skill_len):
                    data = [[torch.tensor(beh_data["dists_pos"])[frame_i, :, 0],
                             torch.tensor(beh_data["dists_neg"])[frame_i, :, 0]],
                            [torch.tensor(beh_data["dists_pos"])[frame_i, :, 1],
                             torch.tensor(beh_data["dists_neg"])[frame_i, :, 1]],
                            [torch.tensor(beh_data["dir_pos"]).squeeze()[frame_i],
                             torch.tensor(beh_data["dir_ab_neg"]).squeeze()[frame_i]]
                            ]
                    beh_name = f"{beh_i}_{skill_len}_{frame_i}_{self.args.action_names[beh_data['action_type'][frame_i]]}_var_{beh_data['variance']:.2f}"
                    # draw_utils.plot_histogram(data, [[["x_pos", "x_neg"]], [["y_pos", "y_neg"]], [["dir_pos", "dir_neg"]]],
                    #                           beh_name, self.args.check_point_path / "skill_attack", figure_size=(30, 10))
            file_utils.save_json(stat_file, att_behavior_data)
        att_behavior_file = self.args.check_point_path / "skill_attack" / f"attack_behaviors.pkl"
        if os.path.exists(att_behavior_file):
            attack_behaviors = file_utils.load_pickle(att_behavior_file)
            attack_behaviors = beh_utils.update_skill_attack_behaviors(self.args, attack_behaviors, att_behavior_data)
        else:
            attack_behaviors = []
            # print(f'- Create {len(att_behavior_data)} attack behaviors')
            for beh_i, beh in enumerate(att_behavior_data):
                attack_behaviors.append(beh_utils.create_skill_attack_behavior(self.args, beh_i, beh))
            file_utils.save_pkl(att_behavior_file, attack_behaviors)

        # for attack_behavior in attack_behaviors:
        #     print(f"# attack behavior: {attack_behavior.clause}")
        self.skill_att_behaviors = attack_behaviors
        return attack_behaviors

    def reasoning_att_behaviors(self, use_ckp=True):
        if len(self.pos_data) == 0:
            return []
        stat_file = self.args.check_point_path / "attack" / f"attack_stats.json"
        if use_ckp and os.path.exists(stat_file):
            att_behavior_data = file_utils.load_json(stat_file)
        else:
            att_behavior_data = smp_utils.stat_rewards(self.win_states,
                                                       self.win_actions,
                                                       self.win_rewards,
                                                       self.args.zero_reward,
                                                       self.args.obj_info,
                                                       self.prop_indices,
                                                       self.args.var_th, "attack",
                                                       self.args.action_names,
                                                       self.args.step_dist,
                                                       self.args.max_dist)
            for beh_i, beh_data in enumerate(att_behavior_data):
                data = [[torch.tensor(beh_data["dists_pos"])[:, 0], torch.tensor(beh_data["dists_neg"])[:, 0]],
                        [torch.tensor(beh_data["dists_pos"])[:, 1], torch.tensor(beh_data["dists_neg"])[:, 1]],
                        [torch.tensor(beh_data["dir_pos"]).squeeze(), torch.tensor(beh_data["dir_ab_neg"]).squeeze()]
                        ]
                beh_name = f"{beh_i}_{self.args.action_names[beh_data['action_type']]}_var_{beh_data['variance']:.2f}"
                draw_utils.plot_histogram(data, [[["x_pos", "x_neg"]], [["y_pos", "y_neg"]], [["dir_pos", "dir_neg"]]],
                                          beh_name, self.args.check_point_path / "attack", figure_size=(30, 10))
            file_utils.save_json(stat_file, att_behavior_data)
        att_behavior_file = self.args.check_point_path / "attack" / f"attack_behaviors.pkl"
        if os.path.exists(att_behavior_file):
            attack_behaviors = file_utils.load_pickle(att_behavior_file)
            attack_behaviors = beh_utils.update_attack_behaviors(self.args, attack_behaviors, att_behavior_data)
        else:
            attack_behaviors = []
            # print(f'- Create {len(att_behavior_data)} attack behaviors')
            for beh_i, beh in enumerate(att_behavior_data):
                attack_behaviors.append(beh_utils.create_attack_behavior(self.args, beh_i, beh))
            file_utils.save_pkl(att_behavior_file, attack_behaviors)

        # for attack_behavior in attack_behaviors:
        #     print(f"# attack behavior: {attack_behavior.clause}")
        self.att_behaviors = attack_behaviors
        return attack_behaviors

    def reasoning_o2o_behaviors(self):
        if os.path.exists(self.args.o2o_data_file):
            o2o_dict = file_utils.load_json(self.args.o2o_data_file)
        else:
            o2o_dict = reason_utils.reason_o2o_states(self.args, self.data)
        self.o2o_data = o2o_dict['behavior_data']
        self.action_delta = o2o_dict['action_data']
        return self.o2o_data

    def reasoning_path_behaviors(self, use_ckp=True):
        if len(self.pos_data) == 0:
            return []
        stat_file = self.args.check_point_path / "path_finding" / f"pf_stats.json"
        if use_ckp and os.path.exists(stat_file):
            pf_behavior_data = file_utils.load_json(stat_file)
        else:
            pf_behavior_data = smp_utils.stat_zero_rewards(self.win_states,
                                                           self.win_actions,
                                                           self.win_rewards,
                                                           self.args.zero_reward,
                                                           self.args.obj_info,
                                                           self.prop_indices,
                                                           self.args.var_th,
                                                           "path_finding",
                                                           self.args.action_names,
                                                           self.args.step_dist)
            for beh_i, beh_data in enumerate(pf_behavior_data):
                data = [[torch.tensor(beh_data["dists_pos"])[:, 0], torch.tensor(beh_data["dists_neg"])[:, 0]],
                        [torch.tensor(beh_data["dists_pos"])[:, 1], torch.tensor(beh_data["dists_neg"])[:, 1]],
                        [torch.tensor(beh_data["dir_pos"]).squeeze(), torch.tensor(beh_data["dir_ab_neg"]).squeeze()]
                        ]
                beh_name = f"{beh_i}_{self.args.action_names[beh_data['action_type']]}_{beh_data['obj_combs']}_var_{beh_data['variance']:.2f}"
                draw_utils.plot_histogram(data, [[["x_pos", "x_neg"]], [["y_pos", "y_neg"]], [["dir_pos", "dir_neg"]]],
                                          beh_name, self.args.check_point_path / "path_finding", figure_size=(30, 10))
            file_utils.save_json(stat_file, pf_behavior_data)
        pf_behavior_file = self.args.check_point_path / "path_finding" / f"pf_behaviors.pkl"
        if os.path.exists(pf_behavior_file):
            pf_behaviors = file_utils.load_pickle(pf_behavior_file)
            pf_behaviors = beh_utils.update_pf_behaviors(self.args, pf_behaviors, pf_behavior_data)
        else:
            pf_behaviors = []
            # print(f'- Create {len(pf_behavior_data)} path_finding behaviors')
            for beh_i, beh in enumerate(pf_behavior_data):
                pf_behaviors.append(beh_utils.create_pf_behavior(self.args, beh_i, beh))
            file_utils.save_pkl(pf_behavior_file, pf_behaviors)

        # for pf_behavior in pf_behaviors:
        #     print(f"# Path Finding behavior: {pf_behavior.clause}")
        self.pf_behaviors = pf_behaviors
        return pf_behaviors

    def reasoning_pf_behaviors(self):
        # print(f"Reasoning defensive behaviors...")
        ############# learn from positive rewards
        pos_states_stat_file = self.args.check_point_path / "path_finding" / f"pf_stats.json"
        if os.path.exists(pos_states_stat_file):
            pos_beh_data = file_utils.load_json(pos_states_stat_file)
        else:
            pos_beh_data = smp_utils.stat_pos_data(self.args, self.win_states, self.win_actions, self.win_rewards,
                                                   self.args.obj_info, self.prop_indices)
            file_utils.save_json(pos_states_stat_file, pos_beh_data)

        pos_behavior_data = smp_utils.best_pos_data_comb(pos_beh_data)
        path_behaviors = beh_utils.create_positive_behaviors(self.args, pos_behavior_data)

        self.pf_behaviors = path_behaviors
        return path_behaviors

    def revise_win(self, history, game_states):
        print("")

        #
        # # punish the last action
        # assert history[-1]["reward"][0] < -1
        # for frame_i in range(len(history) - 1, 0, -1):
        #     frame_action = history[frame_i]['action']
        #     behavior_indices = history[frame_i]['behavior_index']
        #     frame_reward = history[frame_i]['reward'][0]
        #     frame_state = history[frame_i]['state']
        #     frame_mask = smp_utils.mask_tensors_from_states(frame_state, game_info)
        #
        #     # search activated behaviors
        #     frame_action_neg = False
        #     beh_pos_indices = []
        #     pf_data = []
        #     for behavior_index in behavior_indices:
        #         behavior = self.model.behaviors[behavior_index]
        #         # activated path finding behaviors
        #         if not behavior.neg_beh and behavior.action == frame_action:
        #             beh_pos_indices.append(behavior_index)
        #             pf_data.append({'obj_comb': behavior.fact[0].obj_comb, 'prop_comb': behavior.fact[0].prop_comb})
        #
        #         # activated defense behaviors
        #         if behavior.neg_beh and behavior.action == frame_action:
        #             frame_action_neg = True
        #     # revise behavior
        #     if len(beh_pos_indices) > 0 and not frame_action_neg:
        #         for behavior in self.model.behaviors:
        #             if behavior.action == frame_action and behavior.neg_beh:
        #                 for fact in behavior.fact:
        #                     if torch.equal(torch.tensor(fact.mask).reshape(-1), frame_mask.reshape(-1)):
        #                         for pred in fact.preds:
        #                             print(f"- revise frame {frame_i} (last non-defensed frame)")
        #                             print(f'- update behavior: {behavior.clause}')
        #                             data_X = smp_utils.extract_fact_data(fact, frame_state)
        #                             pred.add_item(data_X)
        #                             pred.fit_pred()
        #                             revised = True
        #         # create new behavior
        #         if not revised:
        #             print(f"- require a new defense behavior.")
        #             # defense behavior
        #             lost_states = []
        #             lost_actions = []
        #             lost_rewards = []
        #             for f_i, frame_data in enumerate(history):
        #                 if f_i <= frame_i:
        #                     lost_states.append(history[f_i]['state'].tolist())
        #                     lost_actions.append(history[f_i]['action'])
        #                     if f_i == frame_i:
        #                         lost_rewards.append(frame_reward)
        #                     else:
        #                         lost_rewards.append(history[f_i]['reward'][0])
        #
        #             lost_game_data = {'states': lost_states, 'actions': lost_actions, 'rewards': lost_rewards}
        #         break
        #
        # return lost_game_data

    def revise_loss(self, args, env_args):
        self.update_lost_buffer(env_args.logic_states, env_args.actions, env_args.rewards)
        def_behaviors = self.reasoning_def_behaviors(use_ckp=False, show_log=args.with_explain)
        self.update_behaviors(None, def_behaviors, None, args)

    def revise_timeout(self, history):
        print("")

    def act(self, state):
        if self.args.m == 'getout' or self.args.m == "getoutplus":
            action, explaining = self.getout_actor(state)
        elif self.args.m == 'Assault':
            action, explaining = self.assault_actor(state)
        elif self.args.m in ["Asterix", "Boxing", "Breakout", "Freeway", "Kangaroo", "Pong"]:
            action, explaining = self.asterix_actor(state)
        elif self.args.m == 'threefish':
            action, explaining = self.threefish_actor(state)
        elif self.args.m == 'loot':
            action, explaining = self.loot_actor(state)
        else:
            raise ValueError

        return action, explaining

    def reasoning_act(self, state):
        if self.args.m == 'getout':
            action, explaining = self.getout_reasoning_actor(state)
        else:
            raise ValueError
        return action, explaining

    def get_probs(self):
        probs = self.model.get_probs()
        return probs

    def get_explaining(self):
        explaining = 0
        return explaining

    def get_state(self, state):
        if self.args.m == 'Assault':
            logic_state = oc_utils.extract_logic_state_assault(state, self.args).squeeze(0)
        else:
            raise ValueError
        logic_state = logic_state.tolist()
        result = []
        for list in logic_state:
            obj_state = [round(num, 2) for num in list]
            result.append(obj_state)
        return result

    def getout_actor(self, getout):
        extracted_state = oc_utils.extract_logic_state_getout(getout, self.args)
        predictions, explains = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()

        action = prediction + 1
        explains['action'] = prediction
        return action, explains

    def assault_actor(self, getout):
        extracted_state = oc_utils.extract_logic_state_assault(getout, self.args).unsqueeze(0)
        predictions, explains = self.model(extracted_state)
        # predictions, explains = self.action_combine_assault(predictions, explains)
        prediction = torch.argmax(predictions).cpu().item()
        explains['action'] = prediction

        return prediction, explains

    def asterix_actor(self, objs):
        extracted_state, _ = oc_utils.extract_logic_state_atari(objs, self.args.game_info, self.position_norm_factor)
        predictions, explains = self.model(torch.tensor(extracted_state).unsqueeze(0).to(self.args.device))
        # predictions, explains = self.action_combine_assault(predictions, explains)
        prediction = torch.argmax(predictions).cpu().item()
        explains['action'] = prediction

        return prediction, explains

    def getout_reasoning_actor(self, getout):
        extracted_state = extract_logic_state_getout(getout, self.args)
        predictions, explain = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        action = prediction + 1
        return action, explain


class PpoPlayer:
    def __init__(self, args):
        self.args = args
        self.preds = None
        self.model = self.load_model(args.model_path, args)
        self.buffer_win_rates = torch.zeros(10000)

    def update(self, args=None, behaviors=None, prop_indices=None, explains=None, preds=None):
        if args is not None:
            self.args = args
        if preds is not None:
            self.preds = preds
        self.model.update(args, behaviors, prop_indices, explains, preds)

    def act(self, state):
        if self.args.m == 'getout' or self.args.m == "getoutplus":
            action = self.getout_actor(state)
        else:
            raise ValueError
        return action

    def reasoning_act(self, state):
        if self.args.m == 'getout':
            action = self.getout_reasoning_actor(state)
        else:
            raise ValueError
        return action

    def get_probs(self):
        probs = self.model.get_probs()
        return probs

    def get_explaining(self):
        explaining = 0
        return explaining

    def get_state(self, state):
        if self.args.m == 'Assault':
            logic_state = oc_utils.extract_logic_state_assault(state, self.args).squeeze(0)
        else:
            raise ValueError
        logic_state = logic_state.tolist()
        result = []
        for list in logic_state:
            obj_state = [round(num, 2) for num in list]
            result.append(obj_state)
        return result

    def getout_actor(self, getout):
        extracted_state = extract_neural_state_getout(getout, self.args).to(self.args.device)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        # explaining = explains[prediction]
        explaining = None
        action = prediction + 1
        return action

    def getout_reasoning_actor(self, getout):
        extracted_state = extract_neural_state_getout(getout, self.args).to(self.args.device)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()

        action = prediction + 1
        return action

    def load_model(self, model_path, args, set_eval=True):

        with open(model_path, "rb") as f:
            # model = ActorCritic(args).to(args.device)
            model = NeuralPPO(args).to(args.device)

            model.load_state_dict(state_dict=torch.load(f, map_location="cpu"))

        print(f"- loaded player model from {model_path}")
        model = model.actor
        model.as_dict = True

        if set_eval:
            model = model.eval()

        return model
