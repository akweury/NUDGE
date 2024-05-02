import os
import pickle
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from nsfr.nsfr.utils import get_nsfr_model

from .MLPController.mlpatari import MLPAtari
from .MLPController.mlpgetout import MLPGetout
from .MLPController.mlploot import MLPLoot
from .MLPController.mlpthreefish import MLPThreefish
from .utils_atari import (action_map_atari, extract_logic_state_atari,
                          extract_neural_state_atari, preds_to_action_atari)
from .utils_getout import (action_map_getout, extract_logic_state_getout,
                           extract_neural_state_getout, preds_to_action_getout)
from .utils_loot import (action_map_loot, extract_logic_state_loot,
                         extract_neural_state_loot, preds_to_action_loot)
from .utils_threefish import (action_map_threefish,
                              extract_logic_state_threefish,
                              extract_neural_state_threefish,
                              preds_to_action_threefish)


class NSFR_ActorCritic(nn.Module):
    def __init__(self, args, rng=None):
        super(NSFR_ActorCritic, self).__init__()
        self.rng = random.Random() if rng is None else rng
        self.args = args
        self.actor = get_nsfr_model(self.args, train=True)
        self.prednames = self.get_prednames()
        if self.args.m == 'threefish':
            self.critic = MLPThreefish(out_size=1, logic=True)
        elif self.args.m == 'getout':
            self.critic = MLPGetout(out_size=1, logic=True)
        elif self.args.m == 'loot':
            self.critic = MLPLoot(out_size=1, logic=True)
        elif self.args.m == 'atari':
            self.critic = MLPAtari(out_size=1, logic=True)
        self.num_actions = len(self.prednames)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=self.args.device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions - 1) for _ in range(self.num_actions - 1)],
                         device=self.args.device))

    def forward(self):
        raise NotImplementedError

    def act(self, logic_state, epsilon=0.0, norm_factor=None):
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


from nesy_pi.aitk import ai_interface
from nesy_pi import ilp
from nesy_pi.aitk.utils import file_utils
from nesy_pi.aitk.utils.fol import bk

from src import config


class NSFR_PI_ActorCritic(nn.Module):
    def __init__(self, args, rng=None):
        super(NSFR_PI_ActorCritic, self).__init__()
        self.rng = random.Random() if rng is None else rng
        self.args = args

        clause_file = args.trained_model_folder / args.learned_clause_file
        data = file_utils.load_clauses(clause_file)
        args.p_inv_counter = data["p_inv_counter"]
        args.clauses = [cs for acs in data["clauses"] for cs in acs]
        bk_preds = [bk.neural_predicate_2[bk_pred_name] for bk_pred_name in args.bk_pred_names.split(",")]
        neural_preds = file_utils.load_neural_preds(bk_preds, "bk_pred")
        args.neural_preds = [neural_pred for neural_pred in neural_preds]
        args.lark_path = config.path_nesy / "lark" / "exp.lark"

        if args.env == 'getout':
            args.action_names = config.action_name_getout
            args.game_info = config.game_info_getout
        elif args.env == 'Freeway':
            args.action_names = config.action_name_freeway
            args.game_info = config.game_info_freeway

        lang = ai_interface.get_pretrained_lang(args, data["inv_consts"], data["all_pi_clauses"],
                                                data["all_invented_preds"])
        self.actor = ai_interface.get_nsfr_model(args, lang)
        self.prednames = self.get_prednames()
        if self.args.m == 'threefish':
            self.critic = MLPThreefish(out_size=1, logic=True)
        elif self.args.m == 'getout':
            self.critic = MLPGetout(out_size=1, logic=True)
        elif self.args.m == 'loot':
            self.critic = MLPLoot(out_size=1, logic=True)
        elif self.args.m == 'atari':
            self.critic = MLPAtari(out_size=1, logic=True)
        self.num_actions = len(self.prednames)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device=self.args.device))
        self.upprior = Categorical(
            torch.tensor([0.9] + [0.1 / (self.num_actions - 1) for _ in range(self.num_actions - 1)],
                         device=self.args.device))

    def forward(self):
        raise NotImplementedError

    def act(self, logic_state, epsilon=0.0, norm_factor=None):
        if self.args.env == 'getout':
            logic_state[:, :, -2:] = logic_state[:, :, -2:] / 50
        elif self.args.env == "Freeway":
            # positive data
            logic_state[:, :, -2:] = logic_state[:, :, -2:] / norm_factor

            player_pos_data = logic_state[:, 0:1]
            cars_data = logic_state[:, 1:]
            # above of player
            mask_above = (logic_state[:, 1:, -1] < logic_state[:, 0:1, -1])
            pos_above_data = []
            for s_i in range(len(cars_data)):
                _, above_indices = (player_pos_data[s_i, 0, -1] - cars_data[s_i, mask_above[s_i], -1]).sort()
                data = cars_data[s_i][mask_above[s_i]][above_indices][:3]
                if data.shape[0] < 3:
                    data = torch.cat([torch.zeros(3 - data.shape[0], data.shape[1]), data], dim=0)
                pos_above_data.append(data.unsqueeze(0))
            pos_above_data = torch.cat(pos_above_data, dim=0)

            mask_below = logic_state[:, 1:, -1] > logic_state[:, 0:1, -1]
            pos_below_data = []
            for s_i in range(len(cars_data)):
                _, below_indices = (-player_pos_data[s_i, 0, -1] + cars_data[s_i, mask_below[s_i], -1]).sort()
                data = cars_data[s_i][mask_below[s_i]][below_indices][:1]
                if data.shape[0] < 1:
                    data = torch.cat([torch.zeros(1 - data.shape[0], data.shape[1]), data], dim=0)
                pos_below_data.append(data.unsqueeze(0))
            pos_below_data = torch.cat(pos_below_data, dim=0)
            logic_state = torch.cat((player_pos_data, pos_above_data, pos_below_data), dim=1)

        else:
            raise ValueError
        P_pos = torch.zeros(1, len(self.actor.atoms)).to(self.args.device) + 1e+20
        V_T, param = self.actor.eval_quick(logic_state, P_pos)
        # aa_atom_v = V_T.numpy().T
        # aa_atom = self.actor.atoms
        action_probs = self.actor.get_predictions(V_T, prednames=self.prednames).to(self.args.device)
        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            # get the best action
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(self.args.device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        V_T, param = self.actor.eval_quick(logic_state)
        action_probs = self.actor.get_predictions(V_T.squeeze(), prednames=self.prednames).to(self.args.device)

        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

        return action_logprobs, state_values, dist_entropy

    def get_prednames(self):
        return self.actor.get_prednames()


class LogicPPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.args = args
        if args.with_pi:
            self.policy = NSFR_PI_ActorCritic(self.args).to(self.args.device)
        else:
            self.policy = NSFR_ActorCritic(self.args).to(self.args.device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.get_params(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        if args.with_pi:
            self.policy_old = NSFR_PI_ActorCritic(self.args).to(self.args.device)
        else:
            self.policy_old = NSFR_ActorCritic(self.args).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.prednames = self.get_prednames()

    def select_action(self, state, epsilon=0.0):

        # extract state for different games
        # import ipdb; ipdb.set_trace()
        if self.args.m == 'getout':
            logic_state = extract_logic_state_getout(state, self.args)
            neural_state = extract_neural_state_getout(state, self.args)
            norm_factor = 50
        elif self.args.m == 'threefish':
            logic_state = extract_logic_state_threefish(state, self.args)
            neural_state = extract_neural_state_threefish(state, self.args)
        elif self.args.m == 'loot':
            logic_state = extract_logic_state_loot(state, self.args)
            neural_state = extract_neural_state_loot(state, self.args)
        elif self.args.m == 'atari':
            logic_state = extract_logic_state_atari(state.objects, self.args)
            neural_state = extract_neural_state_atari(state.objects, self.args)
            norm_factor = state.observation_space.shape[0]
        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            # import ipdb; ipdb.set_trace()
            action, action_logprob = self.policy_old.act(logic_state, epsilon=epsilon, norm_factor=norm_factor)

        self.buffer.neural_states.append(neural_state)
        self.buffer.logic_states.append(logic_state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        # different games use different action system, need to map it to the correct action.
        # action of logic game means a String, need to map string to the correct action,
        action = action.item()
        if self.args.m == 'getout':
            action = action_map_getout(action, self.args, self.prednames)
        elif self.args.m == 'threefish':
            action = action_map_threefish(action, self.args, self.prednames)
        elif self.args.m == 'loot':
            action = action_map_loot(action, self.args, self.prednames)
        elif self.args.m == 'atari':
            action = action_map_atari(action, self.args, self.prednames)

        return action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.args.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor

        old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0)).detach().to(self.args.device)
        old_logic_states = torch.squeeze(torch.stack(self.buffer.logic_states, dim=0)).detach().to(self.args.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.args.device)
        try:
            old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.args.device)
        except RuntimeError:
            raise RuntimeError()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_neural_states, old_logic_states,
                                                                        old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # training does not converge if the entropy term is added ...
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path, directory, step_list, reward_list, weight_list):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

        data = {"step_list": step_list,
                "reward_list": reward_list,
                "weight_list": weight_list}
        torch.save(data, directory / "data.pt")

    def load(self, directory):
        # only for recover form crash
        files = []
        if os.path.exists(directory) and os.path.isdir(directory):
            # Iterate through all files in the directory
            for file in os.listdir(directory):
                if file.endswith(".pth"):
                    # Join the folder path with the file name to get the full path
                    file_path = os.path.join(directory, file)
                    # Check if it's a file (not a directory)
                    if os.path.isfile(file_path):
                        files.append(file_path)
        file_epoch_nums = torch.tensor([int(file.split("_")[-1].split(".")[0]) for file in files])
        latest_file_idx = file_epoch_nums.argmax()
        model_name = files[latest_file_idx]

        model_file = os.path.join(directory, model_name)
        self.policy_old.load_state_dict(torch.load(model_file, map_location=torch.device(self.args.device)))
        self.policy.load_state_dict(torch.load(model_file, map_location=torch.device(self.args.device)))

        data = torch.load(directory / "data.pt", map_location=torch.device(self.args.device))
        step_list = data["step_list"]
        reward_list = data["reward_list"]
        weight_list = data["weight_list"]
        return step_list, reward_list, weight_list

    def get_predictions(self, state):
        self.prediction = state
        return self.prediction

    def get_weights(self):
        return self.policy.actor.get_params()

    def get_prednames(self):
        return self.policy.actor.get_prednames()


class LogicPlayer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.prednames = model.get_prednames()

    def act(self, state):
        if self.args.m == 'getout':
            action, explaining = self.getout_actor(state)
        elif self.args.m == 'threefish':
            action, explaining = self.threefish_actor(state)
        elif self.args.m == 'loot':
            action, explaining = self.loot_actor(state)
        elif self.args.m == 'atari':
            action, explaining = self.atari_actor(state)
        return action, explaining

    def get_probs(self):
        probs = self.model.get_probs()
        return probs

    def get_explaining(self):
        explaining = 0
        return explaining

    def get_state(self, state):
        if self.args.m == 'getout':
            logic_state = extract_logic_state_getout(state, self.args).squeeze(0)
        elif self.args.m == 'threefish':
            logic_state = extract_logic_state_threefish(state, self.args).squeeze(0)
        if self.args.m == 'loot':
            logic_state = extract_logic_state_loot(state, self.args).squeeze(0)
        if self.args.m == 'atari':
            logic_state = extract_logic_state_atari(state, self.args).squeeze(0)
        logic_state = logic_state.tolist()
        result = []
        for list in logic_state:
            obj_state = [round(num, 2) for num in list]
            result.append(obj_state)
        return result

    def getout_actor(self, getout):
        extracted_state = extract_logic_state_getout(getout, self.args)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        explaining = self.prednames[prediction]
        action = preds_to_action_getout(prediction, self.prednames)
        return action, explaining

    def atari_actor(self, atari_env):
        # import ipdb; ipdb.set_trace()
        extracted_state = extract_logic_state_atari(atari_env, self.args)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        explaining = self.prednames[prediction]
        action = preds_to_action_atari(prediction, self.prednames)
        return action, explaining

    def threefish_actor(self, state):
        state = extract_logic_state_threefish(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        explaining = self.prednames[action.item()]
        action = preds_to_action_threefish(action, self.prednames)
        return action, explaining

    def loot_actor(self, state):
        state = extract_logic_state_loot(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        explaining = self.prednames[action.item()]
        action = preds_to_action_loot(action, self.prednames)
        return action, explaining


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.neural_states = []
        self.logic_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.neural_states[:]
        del self.logic_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]
