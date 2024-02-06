# Created by jing at 18.01.24
import os
import random
import torch
from torch import nn as nn
from torch.distributions import Categorical

from src.agents.neural_agent import ActorCritic
from src.agents.utils_getout import extract_neural_state_getout
from pi import Reasoner
from pi.utils import smp_utils, beh_utils, file_utils, oc_utils


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
        self.def_behaviors = []
        self.pf_behaviors = []
        self.win_states = []
        self.win_actions = []
        self.win_rewards = []
        self.lost_states = []
        self.lost_actions = []
        self.lost_rewards = []
        self.lost_game_ids = []

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

        self.win_states = torch.tensor(self.win_states)
        self.win_actions = torch.tensor(self.win_actions)
        self.win_rewards = torch.tensor(self.win_rewards)
        self.lost_states = torch.tensor(self.lost_states)
        self.lost_actions = torch.tensor(self.lost_actions)
        self.lost_rewards = torch.tensor(self.lost_rewards)

    def update_lost_buffer(self, lost_game_data):
        self.lost_states = torch.cat((self.lost_states, torch.tensor(lost_game_data['states']).squeeze()), 0)
        self.lost_actions = torch.cat((self.lost_actions, torch.tensor(lost_game_data['actions'])), 0)
        self.lost_rewards = torch.cat((self.lost_rewards, torch.tensor(lost_game_data['rewards'])), 0)

    def update_behaviors(self, pf_behaviors, def_behaviors, args=None):
        if args is not None:
            self.args = args
        if pf_behaviors is not None:
            self.pf_behaviors = pf_behaviors
        if def_behaviors is not None:
            self.def_behaviors = def_behaviors

        self.behaviors = self.pf_behaviors + self.def_behaviors
        self.model.update(args, self.behaviors)

    def reasoning_def_behaviors(self, use_ckp=True):
        # if no data for defensive behaviors exist
        if len(self.lost_states) == 0:
            self.def_behaviors = []
            return

        neg_states_stat_file = self.args.check_point_path / f"{self.args.m}_neg_stats.json"
        if use_ckp and os.path.exists(neg_states_stat_file):
            def_beh_data = file_utils.load_json(neg_states_stat_file)
        else:
            def_beh_data = smp_utils.stat_negative_rewards(self.lost_states, self.lost_actions, self.lost_rewards,
                                                           self.args.zero_reward, self.args.obj_info)
            file_utils.save_json(neg_states_stat_file, def_beh_data)

        neg_beh_file = self.args.check_point_path / f"{self.args.m}_neg_beh.pkl"
        if os.path.exists(neg_beh_file):
            defense_behaviors = file_utils.load_pickle(neg_beh_file)
            defense_behaviors = beh_utils.update_negative_behaviors(self.args, defense_behaviors, def_beh_data)
            for def_beh in defense_behaviors:
                print(f"# defense behavior: {def_beh.clause}")

        else:
            defense_behaviors = []
            for beh_i, beh in enumerate(def_beh_data):
                defense_behaviors.append(beh_utils.create_negative_behavior(self.args, beh_i, beh))
            file_utils.save_pkl(neg_beh_file, defense_behaviors)

        self.def_behaviors = defense_behaviors

    def reasoning_pf_behaviors(self, prop_indices):
        ############# learn from positive rewards
        pos_states_stat_file = self.args.check_point_path / f"{self.args.m}_pos_states.json"
        if os.path.exists(pos_states_stat_file):
            pos_beh_data = file_utils.load_json(pos_states_stat_file)
        else:
            pos_beh_data = smp_utils.stat_pos_data(self.args, self.win_states, self.win_actions, self.win_rewards,
                                                   self.args.obj_info, prop_indices)
            file_utils.save_json(pos_states_stat_file, pos_beh_data)

        pos_behavior_data = smp_utils.best_pos_data_comb(pos_beh_data)
        path_behaviors = beh_utils.create_positive_behaviors(self.args, pos_behavior_data)

        self.pf_behaviors = path_behaviors

    def revise_win(self, history, game_states):
        print("")

    def revise_loss(self, history):
        revised = False
        lost_states = []
        lost_actions = []
        lost_rewards = []

        for f_i, frame_data in enumerate(history):
            lost_states.append(history[f_i]['state'].tolist())
            lost_actions.append(history[f_i]['action'])
            lost_rewards.append(history[f_i]['reward'][0])
        lost_game_data = {'states': lost_states, 'actions': lost_actions, 'rewards': lost_rewards}
        return lost_game_data

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

    def revise_timeout(self, history):
        print("")

    def act(self, state):
        if self.args.m == 'getout' or self.args.m == "getoutplus":
            action, explaining = self.getout_actor(state)
        elif self.args.m == 'Assault':
            action, explaining = self.assault_actor(state)
        elif self.args.m == 'threefish':
            action, explaining = self.threefish_actor(state)
        elif self.args.m == 'loot':
            action, explaining = self.loot_actor(state)
        elif self.args.m == 'atari':
            action, explaining = self.atari_actor(state)
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

    def action_combine_assault(self, action_prob, explains):

        if explains == [-1]:
            return action_prob, explains

        if action_prob.sum() <= 1:
            explains = [b_i for action_prob, b_i in explains]
            return action_prob, explains

        if action_prob[0, 0] == 1:
            action_prob[0, 0] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not action_prob[0] == 1]

        if action_prob[0, 1] == action_prob[0, 2] == 1:
            action_prob[0, 2] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not action_prob[2] == 1]

        # left and right
        if action_prob[0, 3] == action_prob[0, 4] == 1:
            action_prob[0, 3] = 0
            action_prob[0, 4] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if
                        not (action_prob[3] == 1 or action_prob[4] == 1)]

        # left fire and right fire
        if action_prob[0, 5] == action_prob[0, 6] == 1:
            action_prob[0, 5] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not (action_prob[5] == 1)]

        if sum(action_prob[0, [5, 6]]) > 0 and action_prob[0, 1] == 1:
            action_prob[0, 1] = 0
            explains = [(action_prob, b_i) for action_prob, b_i in explains if not (action_prob[1] == 1)]

        if action_prob.sum() > 1:
            raise ValueError
        explains = [b_i for action_prob, b_i in explains]
        return action_prob, explains

    def assault_actor(self, getout):
        extracted_state = oc_utils.extract_logic_state_assault(getout, self.args).unsqueeze(0)
        predictions, explains = self.model(extracted_state)
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
        extracted_state = extract_neural_state_getout(getout, self.args)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        # explaining = explains[prediction]
        explaining = None
        action = prediction + 1
        return action

    def getout_reasoning_actor(self, getout):
        extracted_state = extract_neural_state_getout(getout, self.args)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()

        action = prediction + 1
        return action

    def load_model(self, model_path, args, set_eval=True):

        with open(model_path, "rb") as f:
            model = ActorCritic(args)
            model.load_state_dict(state_dict=torch.load(f, map_location=torch.device('cpu')))

        print(f"- loaded player model from {model_path}")
        model = model.actor
        model.as_dict = True

        if set_eval:
            model = model.eval()

        return model
