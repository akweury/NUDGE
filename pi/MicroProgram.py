# Created by jing at 04.12.23
import os.path

from tqdm import tqdm
import torch
import torch.nn as nn
from pi.utils import file_utils

from pi.utils import smp_utils, Fact, beh_utils
from pi.utils.Behavior import Behavior
from pi import predicate
from pi import pi_lang
from src import config


class MicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args, all_predicates, p_satisfactions, sample_nums, passed_obj_combs, props, action_prob,
                 mask_name, action_mask, reward):
        super().__init__()
        self.args = args
        self.action_prob = action_prob
        self.mask = action_mask
        self.obj_type_existance = smp_utils.mask_name_to_tensor(mask_name, config.mask_splitter)
        self.mask_name = mask_name
        self.obj_type_combs = passed_obj_combs
        self.prop_codes = props
        self.preds = all_predicates
        self.pred_ids = torch.arange(0, len(all_predicates))
        self.sample_nums = sample_nums
        self.p_satisfication = p_satisfactions
        self.obj_type_num = len(passed_obj_combs)
        self.expected_reward = reward

        self.p_spaces = []
        for p_i, pred in enumerate(all_predicates):
            self.p_spaces.append(
                smp_utils.get_param_range(pred.p_bound['min'], pred.p_bound['max'], config.smp_param_unit))

        self.oppm_combs, self.oppm_keys, self.key_nums = smp_utils.arrange_mps(self.obj_type_combs, self.prop_codes,
                                                                               self.obj_type_existance,
                                                                               args.obj_type_indices,
                                                                               self.p_satisfication)

    def check_exists(self, x, obj_type_dict):
        if x.ndim != 3:
            raise ValueError
        state_num = x.size(0)
        mask_batches = self.mask.unsqueeze(0)
        obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
        for obj_type_name, obj_type_indices in obj_type_dict.items():
            if obj_type_name in self.obj_type_names:
                obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
                exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
                exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
                obj_type_exists[:, obj_indices] *= exist_type

        mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
        exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
        return exist_res.bool()

    def forward(self, x, obj_type_indices, avg_data=False):
        # game Getout: tensor with size batch_size * 4 * 6
        satisfies = torch.zeros(x.size(0), dtype=torch.bool)
        # if use_given_parameters:
        #     given_parameters = self.p_bound
        # else:
        #     given_parameters = None
        type_1_name = self.obj_type_names[self.type_codes[0]]
        type_1_obj_codes = obj_type_indices[type_1_name]
        type_2_name = self.obj_type_names[self.type_codes[1]]
        type_2_obj_codes = obj_type_indices[type_2_name]

        if len(type_2_obj_codes) > 1 or len(type_1_obj_codes) > 1:
            print("WARNING:")
        # check predicates satisfaction
        for obj_1 in type_1_obj_codes:
            for obj_2 in type_2_obj_codes:
                data_A = x[:, obj_1, self.prop_codes].reshape(-1)
                data_B = x[:, obj_2, self.prop_codes].reshape(-1)

                obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)
                p_spaces = []
                for p_i, pred in enumerate(self.preds):
                    p_space = self.p_spaces[p_i]
                    p_satisfied = self.p_satisfication[p_i]
                    if not p_satisfied:
                        func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(size=data_A.size())
                    else:
                        func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
                    p_spaces.append(p_values.unsqueeze(0))

                    # satisfy all
                    obj_comb_satisfies *= func_satisfy

                # satisfy any
                satisfies += obj_comb_satisfies

        # check mask satisfaction
        exist_satisfy = self.check_exists(x, obj_type_indices)

        # satisfy all
        satisfies *= exist_satisfy

        # return action probs
        action_probs = torch.zeros(x.size(0), len(self.action))
        action_probs[satisfies] += self.action
        action_probs[satisfies] = action_probs[satisfies] / (action_probs[satisfies] + 1e-20)

        return action_probs, p_spaces


class UngroundedMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args, all_predicates, p_satisfactions, sample_nums, passed_obj_combs, props, action_type,
                 mask_name, action_mask, reward):
        super().__init__()
        self.args = args
        self.action = action_type
        self.mask = action_mask
        self.obj_type_existance = smp_utils.mask_name_to_tensor(mask_name, config.mask_splitter)
        self.mask_name = mask_name
        self.obj_type_combs = passed_obj_combs
        self.prop_codes = props
        self.preds = all_predicates
        self.pred_ids = torch.arange(0, len(all_predicates))
        self.sample_nums = sample_nums

        self.p_satisfication = p_satisfactions
        self.obj_type_num = len(passed_obj_combs)
        self.expected_reward = reward

        self.p_spaces = []
        for p_i, pred in enumerate(all_predicates):
            self.p_spaces.append(
                smp_utils.get_param_range(pred.p_bound['min'], pred.p_bound['max'], config.smp_param_unit))

        self.oppm_combs, self.oppm_keys, self.key_nums = smp_utils.arrange_mps(self.obj_type_combs, self.prop_codes,
                                                                               self.obj_type_existance,
                                                                               args.obj_type_indices,
                                                                               self.p_satisfication)
        self.id = str(reward) + str(action_type) + str(mask_name) + str(props)

    def forward(self, x):
        # given game a state, predict an action
        # game Getout: tensor with size batch_size * 4 * 6

        satisfactions = []
        for oppm_comb in self.oppm_combs:
            satisfaction = smp_utils.oppm_eval(x, oppm_comb, self.oppm_keys, self.preds, self.p_spaces,
                                               self.args.obj_type_indices, self.args.obj_type_names)
            satisfactions.append(satisfaction)

        return satisfactions


class SymbolicRewardMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.action_num = len(args.action_names)
        self.ungrounded_behaviors = []
        self.buffer = None
        self.data_rewards = None
        self.data_actions = None
        self.data_combs = None
        self.obj_ungrounded_behavior_ids = []
        self.preds = None
        self.facts = None

    def load_buffer(self, buffer):
        print(f'- SMP new buffer, total states: {len(buffer.logic_states)}')
        self.buffer = buffer
        self.data_rewards = smp_utils.split_data_by_reward(self.buffer.logic_states, self.buffer.actions,
                                                           self.buffer.rewards, self.action_num)
        self.data_actions = smp_utils.split_data_by_action(self.buffer.logic_states, self.buffer.actions,
                                                           self.action_num)
        if len(self.buffer.rewards) > 0:
            self.data_combs = smp_utils.comb_buffers(self.buffer.logic_states, self.buffer.actions, self.buffer.rewards,
                                                     self.buffer.reason_source)

    def student_searching(self, agent):
        behaviors = agent.model.behaviors
        game_info = agent.model.game_info
        for state, action, reward, behavior_ids in self.data_combs:
            if behavior_ids is not None:
                # predict actions based on behaviors
                # reward is negative, thus this behavior should not be activated
                if reward < -0.1:
                    for b_id in range(len(behaviors)):
                        if b_id in behavior_ids:
                            behaviors[b_id].falsify_pred_params(self.preds, state, game_info)
                        else:
                            behaviors[b_id].validify_pred_params(self.preds, state, game_info)

    def update(self, args=None, preds=None, facts=None):
        if args is not None:
            self.args = args
        if preds is not None:
            self.preds = preds
        if facts is not None:
            self.facts = facts

    def forward_searching(self, relate_2_obj_types, relate_2_prop_types, obj_types):
        # ungrounded behaviors:
        # Existing at least one predicate is true,
        # but not know which part of the data is the reason.

        obj_grounded_behaviors = []
        for reward, reward_states in self.data_rewards.items():
            if reward == -0.1:
                continue
            for action_prob, action_states in reward_states.items():
                masks = smp_utils.all_exist_mask(action_states, obj_types)
                for mask_name, action_mask in masks.items():
                    if action_mask.sum() == 0:
                        continue
                    for props in relate_2_prop_types:
                        passed_obj_combs = []
                        sample_nums = []
                        p_satisfactions = []
                        all_predicates = predicate.get_preds(len(props))
                        for objs in relate_2_obj_types:
                            p_satisfaction, sample_num = smp_utils.check_pred_satisfaction(action_states,
                                                                                           all_predicates,
                                                                                           action_mask, objs, props)
                            if (sum(p_satisfaction)) > 0:
                                passed_obj_combs.append(objs)
                                sample_nums.append(sample_num)
                                p_satisfactions.append(p_satisfaction)
                        if len(passed_obj_combs) > 1:
                            # obj ungrounded behavior
                            behavior = UngroundedMicroProgram(self.args, all_predicates, p_satisfactions, sample_nums,
                                                              passed_obj_combs, props, action_prob, mask_name,
                                                              action_mask, reward)
                            if behavior.id not in self.obj_ungrounded_behavior_ids:
                                self.ungrounded_behaviors.append(behavior)
                                self.obj_ungrounded_behavior_ids.append(behavior.id)
                            else:
                                print("should not be called.")
                        elif len(passed_obj_combs) == 1:
                            behavior = MicroProgram(self.args, all_predicates, p_satisfactions, sample_nums,
                                                    passed_obj_combs, props, action_prob, mask_name, action_mask,
                                                    reward)
                            obj_grounded_behaviors.append(behavior)
                print(f"reward: {reward}, action: {action_prob}, states: {len(action_states)}")
        print(f'forward searched ungrounded behaviors: {len(self.ungrounded_behaviors)}')
        print(f'forward searched grounded behaviors: {len(obj_grounded_behaviors)}')
        return obj_grounded_behaviors

    def backward_searching(self):
        print(f"- backward searching ...")
        for reward, reward_states in self.data_rewards.items():
            if reward == -0.1:
                continue
            for action_prob, action_states in reward_states.items():
                for behavior in self.ungrounded_behaviors:

                    if behavior.expected_reward == reward and torch.equal(behavior.action, action_prob):
                        # check which explains can satisfy more states

                        satisfactions = behavior(action_states)
                        # filter out wrong groundings, until only one option left
                        # update behavior's oppm combs
                        behavior.oppm_combs = self.grounding(behavior.oppm_combs, satisfactions)
                        # top scored explains shall be kept
                        # how to remove them?
        new_ungrounded_behaviors = []
        for behavior in self.ungrounded_behaviors:
            if len(behavior.oppm_combs) > 0:
                new_ungrounded_behaviors.append(behavior)
        self.ungrounded_behaviors = new_ungrounded_behaviors

    def backward_searching2(self):
        print(f"- backward searching 2...")
        # consider batching evaluation
        for behavior in self.ungrounded_behaviors:
            satisfactions_behavior = []
            satisfactions_mask = []
            for data in self.data_combs:
                state, action, reward = data
                if behavior.expected_reward == reward:

                    satisfactions_data = []
                    same_action_id = torch.argmax(behavior.action).item() == action
                    same_mask = smp_utils.mask_name_from_state(state, self.args.obj_type_names,
                                                               config.mask_splitter) == behavior.mask_name
                    if same_action_id and same_mask:
                        satisfactions_mask.append(True)
                        for oppm_comb in behavior.oppm_combs:
                            comb_obj_id = [oppm_comb[idx] for idx in behavior.oppm_keys["obj"]]
                            comb_prop_id = [oppm_comb[idx] for idx in behavior.oppm_keys["prop"]]
                            comb_preds = [oppm_comb[idx] for idx in behavior.oppm_keys["pred"]]
                            state_preds, _ = smp_utils.check_pred_satisfaction(state, behavior.preds, None, comb_obj_id,
                                                                               comb_prop_id, behavior.p_spaces,
                                                                               mode="eval")
                            same_pred = state_preds == comb_preds
                            if same_pred:
                                satisfactions_data.append(True)
                            else:
                                satisfactions_data.append(False)
                    else:
                        satisfactions_mask.append(False)
                else:
                    satisfactions_data = [False] * len(behavior.oppm_combs)
                    satisfactions_mask.append(False)

                satisfactions_behavior.append(satisfactions_data)
            passed_states = [satisfactions_behavior[s_i] for s_i, state in enumerate(satisfactions_mask) if state]

            behavior.oppm_combs = self.grounding(behavior.oppm_combs, passed_states)

        new_ungrounded_behaviors = []
        for behavior in self.ungrounded_behaviors:
            if len(behavior.oppm_combs) > 0:
                new_ungrounded_behaviors.append(behavior)
        self.ungrounded_behaviors = new_ungrounded_behaviors

    def grounding(self, combs, satisfactions):

        satisfaction_count = torch.tensor(satisfactions).float()
        if len(satisfaction_count.size()) == 2:
            satisfaction_count = satisfaction_count.sum(0)
        if len(combs) != len(satisfaction_count):
            print("Warning:")
        satisfactions = satisfaction_count > 0
        grounded_combs = [combs[s_i] for s_i, satisfaction in enumerate(satisfactions) if satisfaction]
        print(f"behavior grounded reasons from {len(combs)} to {len(grounded_combs)}")
        return grounded_combs

    def programming(self, agent, game_info, prop_indices):
        relate_2_obj_types = smp_utils.get_all_2_combinations(game_info)
        relate_2_prop_types = smp_utils.get_all_subsets(prop_indices, empty_set=False)

        self.facts = smp_utils.get_smp_facts(game_info, relate_2_obj_types, relate_2_prop_types, self.preds)
        behaviors = self.student_searching(agent)

        # obj_grounded_behaviors = self.forward_searching(relate_2_obj_types, relate_2_prop_types, obj_types)
        # self.backward_searching()
        # self.backward_searching2()

        return behaviors


class SymbolicMicroProgram(nn.Module):
    """ generate one micro-program
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.action_num = len(args.action_names)
        self.ungrounded_behaviors = []
        self.buffer = None
        self.data_rewards = None
        self.data_actions = None
        self.actions = None
        self.rewards = None
        self.states = None
        self.data_combs = None
        self.obj_ungrounded_behavior_ids = []
        self.preds = None

    def load_buffer(self, buffer):
        print(f'- Loaded game history, win games : {len(buffer.logic_states)}, loss games : {len(buffer.lost_logic_states)}')
        self.buffer = buffer
        # self.data_rewards = smp_utils.split_data_by_reward(self.buffer.logic_states, self.buffer.actions,
        #                                                    self.buffer.rewards, self.action_num)
        # self.data_actions = smp_utils.split_data_by_action(self.buffer.logic_states, self.buffer.actions,
        #                                                    self.action_num)
        # if len(self.buffer.rewards) != len(self.buffer.logic_states):
        #     self.buffer.rewards = [0] * len(self.buffer.logic_states)
        self.data_combs = smp_utils.comb_buffers(self.buffer.logic_states, self.buffer.actions, self.buffer.rewards)

        self.actions = []
        self.lost_actions = []
        self.rewards = []
        self.lost_rewards = []
        self.states = []
        self.lost_states = []
        self.lost_game_ids = []
        for g_i in range(len(buffer.actions)):
            self.actions += buffer.actions[g_i]
            self.rewards += buffer.rewards[g_i]
            self.states += buffer.logic_states[g_i].tolist()

        for g_i in range(len(buffer.lost_actions)):
            self.lost_actions += buffer.lost_actions[g_i]
            self.lost_rewards += buffer.lost_rewards[g_i]
            self.lost_states += buffer.lost_logic_states[g_i].tolist()
            self.lost_game_ids += [g_i] * len(buffer.lost_logic_states[g_i].tolist())

        self.actions = torch.tensor(self.actions)
        self.lost_actions = torch.tensor(self.lost_actions)
        self.rewards = torch.tensor(self.rewards)
        self.lost_rewards = torch.tensor(self.lost_rewards)
        self.states = torch.tensor(self.states)
        self.lost_states = torch.tensor(self.lost_states)
        self.lost_game_ids = torch.tensor(self.lost_game_ids)

    def extract_behaviors(self, facts, fact_actions):
        beh_indices = fact_actions.sum(dim=-1) == 1
        beh_facts = facts[beh_indices]
        beh_actions = fact_actions[beh_indices].argmax(dim=-1)
        behaviors = []
        for beh_i in range(len(beh_facts)):
            behavior = Behavior(beh_facts[beh_i], beh_actions[beh_i], 0)
            behaviors.append(behavior)
        return behaviors

    def teacher_searching(self, game_info, prop_indices):
        # strategy: action, objects, mask, properties, if predicates are valid
        behaviors = []
        # states do the same action, but they can still have different reasons for doing this action
        for action, action_states in self.data_actions.items():
            # states with same object existence
            facts = smp_utils.get_all_facts(game_info, prop_indices)
            all_masks = smp_utils.all_exist_mask(action_states, game_info)
            for f_i, fact in enumerate(facts):
                if fact["mask"] == "exist_agent#exist_key#exist_door#exist_enemy" and fact["objs"] == [1, 2] and \
                        fact["props"] == [4] and torch.equal(fact["pred_tensors"],
                                                             torch.tensor([False, False, False, True])):
                    print("watch")

                satisfy = smp_utils.satisfy_fact(fact, action_states, all_masks, game_info)
                if satisfy:
                    if fact["pred_tensors"][3] == False and fact["preds"][3].dist is not None:
                        print("watch")
                    passed_state_num = sum(all_masks[fact["mask"]])
                    behavior = Behavior([fact], action, passed_state_num)
                    behavior.clause = pi_lang.behavior2clause(self.args, behavior)
                    behaviors.append(behavior)

        for b in behaviors:
            print(f"teacher beh: {b.clause}")
        print(f"teacher behaviors num: {len(behaviors)}")
        return behaviors

    def student_search(self, teacher_behaviors, game_info, prop_indices):

        student_behaviors = []
        un_checked_behaviors = teacher_behaviors
        for i in range(3):
            child_behaviors = []
            for t_beh in un_checked_behaviors:
                satisfied_data, conflict_data = smp_utils.search_behavior_conflict_states(t_beh, self.data_combs,
                                                                                          game_info, self.action_num)
                if len(satisfied_data) / len(self.data_combs) > 0.995 or conflict_data is None:
                    student_behaviors.append(t_beh)
                else:
                    child_behaviors.append({"beh": t_beh, "pos": satisfied_data, "neg": conflict_data})
            print(f"round {i}, child behaviors: {len(child_behaviors)}")
            un_checked_behaviors = []
            for t_i, c_beh in enumerate(child_behaviors):
                conflict_behaviors = self.teacher_search(self.args, c_beh["neg"], game_info, prop_indices)
                combined_behaviors = self.combine_behaviors(c_beh["beh"], conflict_behaviors)
                conflict_behaviors = smp_utils.back_check(c_beh["pos"], combined_behaviors, game_info, self.action_num)
                un_checked_behaviors += conflict_behaviors

            print(f"round {i}, unchecked behaviors: {len(un_checked_behaviors)}")

        return student_behaviors

    def combine_behaviors(self, parent_behavior, child_behaviors):
        combined_behaviors = []
        for conflict_behavior in child_behaviors:
            combined_behavior = self.combine_behavior(parent_behavior, conflict_behavior)
            if combined_behavior is not None:
                combined_behaviors.append(combined_behavior)
        for beh in combined_behaviors:
            print(f"combined beh: {beh.clause}")
        return combined_behaviors

    def combine_behavior(self, parent_behavior, child_behavior):
        parent_mask = parent_behavior.fact[0]["mask"]
        child_mask = child_behavior.fact[0]["mask"]
        assert parent_mask == child_mask

        child_repeat_facts = torch.zeros(len(child_behavior.fact), dtype=torch.bool)
        for cf_i, c_fact in enumerate(child_behavior.fact):
            repeat_fact = False
            for p_fact in parent_behavior.fact:
                if c_fact["mask"] == p_fact["mask"] and c_fact["objs"] == p_fact["objs"] and c_fact["props"] == \
                        p_fact["props"] and torch.equal(c_fact["pred_tensors"], p_fact["pred_tensors"]):
                    repeat_fact = True
            if repeat_fact:
                child_repeat_facts[cf_i] = True

        child_facts = [child_behavior.fact[rf_i] for rf_i in range(len(child_repeat_facts)) if
                       not child_repeat_facts[rf_i]]
        if len(child_facts) == 0:
            return None
        comb_fact = parent_behavior.fact + child_facts

        new_behavior = Behavior(comb_fact, child_behavior.action, 0)
        new_behavior.clause = pi_lang.behavior2clause(self.args, new_behavior)
        # print(f"new combined behavior : {new_behavior.clause}")

        return new_behavior

    def teacher_search(self, args, action_splitted_states, game_info, prop_indices, eval_mode=False):
        relate_2_obj_types = smp_utils.get_all_2_combinations(game_info)
        relate_2_prop_types = [[each] for each in prop_indices]
        facts = smp_utils.get_smp_facts(game_info, relate_2_obj_types, relate_2_prop_types)

        behaviors = []
        for action, action_states in action_splitted_states.items():
            # states with same object existence
            all_masks = smp_utils.all_exist_mask(action_states, game_info)
            for fact in facts:
                satisfy = smp_utils.satisfy_fact(fact, action_states, all_masks, game_info, eval_mode)
                if satisfy:
                    passed_state_num = sum(all_masks[fact["mask"]])
                    behavior = Behavior([fact], action, passed_state_num)
                    behavior.clause = pi_lang.behavior2clause(args, behavior)
                    behaviors.append(behavior)
        return behaviors

    def grounding(self, combs, satisfactions):

        satisfaction_count = torch.tensor(satisfactions).float()
        if len(satisfaction_count.size()) == 2:
            satisfaction_count = satisfaction_count.sum(0)
        if len(combs) != len(satisfaction_count):
            print("Warning:")
        satisfactions = satisfaction_count > 0
        grounded_combs = [combs[s_i] for s_i, satisfaction in enumerate(satisfactions) if satisfaction]
        print(f"behavior grounded reasons from {len(combs)} to {len(grounded_combs)}")
        return grounded_combs

    def programming(self, game_info, prop_indices):
        ######## learn from negative rewards
        neg_states_stat_file = self.args.check_point_path / "neg_stats.json"
        if os.path.exists(neg_states_stat_file):
            def_beh_data = file_utils.load_json(neg_states_stat_file)
        else:
            def_beh_data = smp_utils.stat_negative_rewards(self.lost_states, self.lost_actions, self.lost_rewards,
                                                           self.args.zero_reward, game_info)
            file_utils.save_json(neg_states_stat_file, def_beh_data)


        neg_beh_file = self.args.check_point_path / 'neg_beh.pkl'
        if os.path.exists(neg_beh_file):
            defense_behaviors = file_utils.load_pickle(neg_beh_file)
            for def_beh in defense_behaviors:
                print(f"# defense behavior: {def_beh.clause}")
        else:
            defense_behaviors = beh_utils.create_negative_behaviors(self.args, def_beh_data)
            file_utils.save_pkl(neg_beh_file, defense_behaviors)

        ############# learn from positive rewards
        pos_states_stat_file = self.args.check_point_path / "pos_states.json"
        if os.path.exists(pos_states_stat_file):
            pos_beh_data = file_utils.load_json(pos_states_stat_file)
        else:
            pos_beh_data = smp_utils.stat_pos_data(self.states, self.actions, self.rewards, game_info, prop_indices,
                                                   self.args.top_kp,
                                                   self.args.pass_th, self.args.failed_th)
            file_utils.save_json(pos_states_stat_file, pos_beh_data)

        pos_behavior_data = smp_utils.best_pos_data_comb(pos_beh_data)
        path_behaviors = beh_utils.create_positive_behaviors(self.args, pos_behavior_data)

        behaviors = defense_behaviors + path_behaviors
        return behaviors

    def merge_truth_table_celles(self, facts, fact_truth_table, actions):

        head_types, fact_head_ids, fact_bodies, fact_heads = smp_utils.fact_grouping_by_head(facts)
        for h_i, head_type in enumerate(head_types):
            head_type_i = [f_i for f_i in range(len(facts)) if fact_head_ids[f_i] == h_i]
            if len(head_type_i) == 0:
                continue

            fact_table_i = fact_truth_table[:, head_type_i]

            for f_i in range(len(head_type_i) - 1):
                action_types = actions[fact_table_i[:, f_i]].unique()
                if len(action_types) == 1:
                    min_value = facts[head_type_i[f_i]]["preds"][1].p_bound["min"]
                    max_value = facts[head_type_i[f_i]]["preds"][2].p_bound["max"]

                    # right_min = facts[head_type_i[f_i + 1]]["preds"][1].p_bound["min"]
                    # right_shift = 1
                    # # right_max = facts[head_type_i[f_i + 1]]["preds"][2].p_bound["max"]
                    # right_border_index = f_i

                    smaller_f_i = f_i
                    for r_f_i in range(len(head_type_i)):
                        right_action_types = actions[fact_table_i[:, r_f_i]].unique()
                        right_min = facts[head_type_i[r_f_i]]["preds"][1].p_bound["min"]
                        right_max = facts[head_type_i[r_f_i]]["preds"][2].p_bound["max"]
                        if len(right_action_types) == 1 and right_action_types == action_types and right_min == min_value and right_max >= max_value:
                            fact_table_i[:, smaller_f_i] = False
                            smaller_f_i = r_f_i
                            max_value = right_max

                    #
                    # while (right_max > max_value and right_min == min_value):
                    #     right_action_types = actions[fact_table_i[:, f_i + right_shift]].unique()
                    #     if len(right_action_types) == 1 and right_action_types == action_types:
                    #         fact_table_i[:, f_i + right_shift - 1] = False  # merge from right side
                    #         right_shift += 1
                    #         right_border_index += 1
                    #         try:
                    #             right_min = facts[head_type_i[f_i + right_shift]]["preds"][1].p_bound["min"]
                    #             right_max = facts[head_type_i[f_i + right_shift]]["preds"][2].p_bound["max"]
                    #         except IndexError:
                    #             print("")
                    #     else:
                    #         break
                    #

                    for l_f_i in range(len(head_type_i)):
                        if l_f_i != smaller_f_i:
                            lfi_action_types = actions[fact_table_i[:, l_f_i]].unique()
                            left_min = facts[head_type_i[l_f_i]]["preds"][1].p_bound["min"]
                            left_max = facts[head_type_i[l_f_i]]["preds"][2].p_bound["max"]
                            if len(lfi_action_types) == 1 and lfi_action_types == action_types and left_min >= min_value and left_max == max_value:
                                fact_table_i[:, l_f_i] = False  # merge from left side

            fact_truth_table[:, head_type_i] = fact_table_i
        return fact_truth_table

    def get_new_behavior(self, data, action):

        beh_facts = []
        for fact in data["facts"]:
            beh_facts.append(
                Fact.ProbFact([predicate.GT()], fact["mask"], fact["objs"], fact["props"][0], fact["delta"]))
        reward = data["expected_reward"]
        passed_state_num = data["passed_state_num"]
        test_passed_state_num = data["test_passed_state_num"]
        failed_state_num = data["failed_state_num"]
        test_failed_state_num = data["test_failed_state_num"]
        neg_beh = False
        behavior = Behavior(neg_beh, beh_facts, action, reward, passed_state_num, test_passed_state_num,
                            failed_state_num,
                            test_failed_state_num)
        behavior.clause = pi_lang.behavior2clause(self.args, behavior)
        return behavior

