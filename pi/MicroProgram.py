# Created by jing at 04.12.23

from tqdm import tqdm
import torch
import torch.nn as nn

from pi.utils import smp_utils
from pi import predicate
from pi import pi_lang
from pi.Behavior import Behavior
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
        self.data_combs = None
        self.obj_ungrounded_behavior_ids = []
        self.preds = None

    def load_buffer(self, buffer):
        print(f'- SMP new buffer, total states: {len(buffer.logic_states)}')
        self.buffer = buffer
        self.data_rewards = smp_utils.split_data_by_reward(self.buffer.logic_states, self.buffer.actions,
                                                           self.buffer.rewards, self.action_num)
        self.data_actions = smp_utils.split_data_by_action(self.buffer.logic_states, self.buffer.actions,
                                                           self.action_num)
        if len(self.buffer.rewards) != len(self.buffer.logic_states):
            self.buffer.rewards = [0] * len(self.buffer.logic_states)
        self.data_combs = smp_utils.comb_buffers(self.buffer.logic_states, self.buffer.actions, self.buffer.rewards,
                                                 self.buffer.reason_source)

    def extract_behaviors(self, facts, fact_actions):
        beh_indices = fact_actions.sum(dim=-1) == 1
        beh_facts = facts[beh_indices]
        beh_actions = fact_actions[beh_indices].argmax(dim=-1)
        behaviors = []
        for beh_i in range(len(beh_facts)):
            behavior = Behavior(beh_facts[beh_i], beh_actions[beh_i], 0)
            behaviors.append(behavior)
        return behaviors

    # def extend_facts(self, game_info, prop_indices):
    #     behaviors = []
    #     facts = smp_utils.get_all_facts(game_info, prop_indices)
    #     base_fact_actions = smp_utils.check_corresponding_actions(facts, self.data_combs, game_info)
    #     base_facts = smp_utils.remove_trivial_facts(facts, base_fact_actions)
    #     new_behs = self.extract_behaviors(facts, base_fact_actions)
    #     behaviors.append(new_behs)
    #
    #     facts = []
    #     fact_actions = []
    #     for i in range(3):
    #         facts, fact_actions = smp_utils.extend_one_fact_to_fact(facts, fact_actions, base_facts, base_fact_actions)
    #         new_behs = self.extract_behaviors(facts, fact_actions)
    #         behaviors.append(new_behs)

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

    def student_searching(self, game_info, teacher_behaviors):
        # strategy: action, objects, mask, properties, if predicates are valid
        student_behaviors = []

        for i in range(len(teacher_behaviors)):
            beh_match_result = []
            teacher_behavior = teacher_behaviors[i]
            for data_comb in self.data_combs:
                state, action, reward, reason_resource = data_comb

                # all facts have to be matched
                fact_match = teacher_behavior.eval_behavior(state, game_info)
                if not fact_match:
                    beh_match_result.append(True)
                if fact_match:
                    action_match = teacher_behavior.action.argmax() == action
                    state_match = fact_match * action_match
                    beh_match_result.append(state_match)

            beh_not_match_result = ~torch.tensor(beh_match_result)
            print(f"\n- behavior {i} failed states: {sum(beh_not_match_result)} / {len(self.data_combs)}")
            print(f"parent: {teacher_behavior.clause}")
            beh_not_match_states = [self.data_combs[i][0].tolist() for i in range(len(beh_not_match_result)) if
                                    beh_not_match_result[i]]
            beh_not_match_state_actions = [self.data_combs[i][1].tolist() for i in range(len(beh_not_match_result)) if
                                           beh_not_match_result[i]]
            beh_not_match_states = torch.tensor(beh_not_match_states).squeeze()
            beh_not_match_state_actions = torch.tensor(beh_not_match_state_actions)

            data_actions = smp_utils.split_data_by_action(beh_not_match_states, beh_not_match_state_actions,
                                                          self.action_num)
            variation_behaviors = self.teacher_search(self.args, self.facts, data_actions, game_info)
            combined_behaviors = self.combine_behaviors(teacher_behavior, variation_behaviors)
            print(f"child behaviors: {len(variation_behaviors)}")
            for behavior in combined_behaviors:
                print(f"child: {behavior.clause}")

            student_behaviors += combined_behaviors
            print(f"studend behaviors num: {len(student_behaviors)}")
            print("")
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

    def programming(self, game_info, prop_indices):
        # for loop the teacher searching, different iteration use different obj combs, or maybe different facts
        facts, fact_truth_table, actions = self.calc_truth_table(game_info, prop_indices)
        merged_fact_truth_table = self.merge_truth_table_celles(facts, fact_truth_table)
        behaviors = self.brute_search(facts, merged_fact_truth_table, actions)
        for beh in behaviors:
            print(f"{beh.clause}")
        return behaviors

    def merge_truth_table_celles(self, facts, fact_truth_table, actions):

        head_types, fact_head_ids, fact_bodies, fact_heads = smp_utils.fact_grouping_by_head(facts)
        for h_i, head_type in enumerate(head_types):
            facts_i = [f_i for f_i in range(len(facts)) if fact_head_ids[f_i] == h_i]
            fact_table_i = fact_truth_table[:, facts_i]
            for f_i in range(len(facts_i) - 1):
                assert fact_heads[facts_i[f_i]] == fact_heads[facts_i[f_i + 1]]
                if fact_bodies[facts_i[f_i]]["max"] == fact_bodies[facts_i[f_i+1]]["min"]:
                    print(f"{fact_bodies[facts_i[f_i]]}, {fact_bodies[facts_i[f_i + 1]]}")
                else:
                    print(f"")
                both_true = fact_table_i[:, f_i] * fact_table_i[:, f_i + 1]

                if both_true.sum() > 0:
                    print("")
                fact_table_i[both_true, f_i] = False
            fact_truth_table[:, facts_i] = fact_table_i
        return fact_truth_table

    def get_new_behavior(self, facts, action, passed_state_num):
        if not isinstance(facts, list):
            facts = [facts]
        behavior = Behavior(facts, action, passed_state_num)
        behavior.clause = pi_lang.behavior2clause(self.args, behavior)
        return behavior

    def calc_truth_table(self, game_info, prop_indices):
        facts = smp_utils.get_all_facts(game_info, prop_indices)
        actions = torch.tensor([d[1].tolist() for d in self.data_combs]).squeeze()

        truth_table = smp_utils.check_fact_truth(facts, self.data_combs, game_info)
        valid_indices = truth_table.sum(dim=0) > 0
        valid_truth_table = truth_table[:, valid_indices]
        valid_facts = [facts[i] for i in range(len(facts)) if valid_indices[i]]

        return valid_facts, valid_truth_table, actions

    def brute_search(self, facts, fact_truth_table, actions):
        # searching behaviors
        behaviors = []
        for at_i, action_type in enumerate(actions.unique()):
            fact_table = fact_truth_table[actions == at_i, :]
            fact_anti_table = fact_truth_table[actions != at_i, :]
            for repeat_i in range(2):
                ci_combs = smp_utils.get_fact_combs(repeat_i + 1, fact_table.size(1))
                for ci_i in tqdm(range(len(ci_combs)), ascii=True, desc=f"{repeat_i + 1} fact behavior search"):
                    ci_comb = ci_combs[ci_i]
                    fact_comb_state_indices = fact_table[:, ci_comb].prod(dim=-1).bool()
                    fact_comb_neg_state_indices = fact_anti_table[:, ci_comb].prod(dim=-1).bool()
                    passed_state_num = fact_comb_state_indices.sum()
                    passed_neg_state_num = fact_comb_neg_state_indices.sum()
                    if passed_state_num > 10 and passed_neg_state_num == 0:
                        beh_facts = [facts[i] for i in ci_comb]
                        behavior = self.get_new_behavior(beh_facts, action_type, passed_state_num)
                        behaviors.append(behavior)

        return behaviors

    # def behaviors_from_actions(self, relate_2_obj_types, relate_2_prop_types, obj_types):
    #     behaviors = []
    #     for action, states in self.data_actions.items():
    #         masks = smp_utils.all_exist_mask(states, obj_types)
    #         for mask_name, mask in masks.items():
    #             for obj_types in relate_2_obj_types:
    #                 obj_1_type = obj_types[obj_types[0]]
    #                 obj_2_type = obj_types[obj_types[1]]
    #                 obj_1_indices = obj_types[obj_1_type]
    #                 obj_2_indices = obj_types[obj_2_type]
    #                 for prop_types in relate_2_prop_types:
    #                     # as long as any two objs satisfied
    #                     all_preds = predicate.get_preds()
    #                     p_satisfication = torch.zeros(len(all_preds), dtype=torch.bool)
    #                     for obj_1_index in obj_1_indices:
    #                         for obj_2_index in obj_2_indices:
    #                             # select data
    #                             if obj_2_index >= 4:
    #                                 raise ValueError
    #                             data_A = states[mask, obj_1_index]
    #                             data_B = states[mask, obj_2_index]
    #                             if len(data_A) == 0:
    #                                 continue
    #                             data_A = data_A[:, prop_types]
    #                             data_B = data_B[:, prop_types]
    #                             # distinguish predicates
    #                             for p_i, pred in enumerate(all_preds):
    #                                 p_satisfication[p_i] += pred.fit(data_A, data_B, obj_types)
    #
    #                     if (p_satisfication.sum()) > 0:
    #                         print(f'new pred, grounded_objs:{obj_types}, action:{action}')
    #                         behavior = {'preds': all_preds,
    #                                     'p_satisfication': p_satisfication,
    #                                     'is_grounded': True,
    #                                     'grounded_types': obj_types,
    #                                     'grounded_prop': prop_types,
    #                                     'action': action,
    #                                     'mask': mask_name}
    #                         behaviors.append(behavior)
    #
    #     return behaviors
    #
    # def check_exists(self, x, obj_type_dict):
    #     if x.ndim != 3:
    #         raise ValueError
    #     state_num = x.size(0)
    #     mask_batches = self.mask.unsqueeze(0)
    #     obj_type_exists = torch.ones(size=(state_num, self.obj_type_num), dtype=torch.bool)
    #     for obj_type_name, obj_type_indices in obj_type_dict.items():
    #         if obj_type_name in self.obj_type_names:
    #             obj_indices = [n_i for n_i, name in enumerate(self.obj_type_names) if name == obj_type_name]
    #             exist_objs = (x[:, obj_type_indices, obj_indices] > 0.8)
    #             exist_type = exist_objs.prod(dim=-1, keepdims=True).bool()
    #             obj_type_exists[:, obj_indices] *= exist_type
    #
    #     mask_batches = torch.repeat_interleave(mask_batches, x.size(0), dim=0)
    #     exist_res = torch.prod(mask_batches == obj_type_exists, dim=1)
    #     return exist_res.bool()
    #
    # def forward(self, x, obj_type_indices, avg_data=False):
    #     # game Getout: tensor with size batch_size * 4 * 6
    #     action_probs = torch.zeros(len(self.type_codes), x.size(0), len(self.action))
    #     p_spaces = []
    #     for t_i, type_code in enumerate(self.type_codes):
    #         t_i_p_spaces = []
    #         satisfies = torch.zeros(x.size(0), dtype=torch.bool)
    #         type_1_obj_codes = obj_type_indices[self.obj_type_names[type_code[0]]]
    #         type_2_obj_codes = obj_type_indices[self.obj_type_names[type_code[1]]]
    #         # check predicates satisfaction
    #         for obj_1 in type_1_obj_codes:
    #             for obj_2 in type_2_obj_codes:
    #                 for prop_code in self.prop_codes:
    #                     data_A = x[:, obj_1, prop_code].reshape(-1)
    #                     data_B = x[:, obj_2, prop_code].reshape(-1)
    #
    #                     obj_comb_satisfies = torch.ones(x.size(0), dtype=torch.bool)
    #
    #                     for p_i, pred in enumerate(self.preds):
    #                         p_space = self.p_spaces[t_i][p_i]
    #                         p_satisfied = self.p_satisfication[p_i]
    #                         if not p_satisfied:
    #                             func_satisfy, p_values = torch.ones(data_A.size()).bool(), torch.zeros(
    #                                 size=data_A.size())
    #                         else:
    #                             func_satisfy, p_values = pred.eval(data_A, data_B, p_space)
    #                         t_i_p_spaces.append(p_values.unsqueeze(0))
    #
    #                         # satisfy all
    #                         obj_comb_satisfies *= func_satisfy
    #
    #                     # satisfy any
    #                     satisfies += obj_comb_satisfies
    #
    #         # check mask satisfaction
    #         exist_satisfy = self.check_exists(x, obj_type_indices)
    #
    #         # satisfy all
    #         satisfies *= exist_satisfy
    #
    #         # return action probs
    #         action_probs[t_i, satisfies] += self.action
    #         action_probs[t_i, satisfies] = action_probs[t_i, satisfies] / (action_probs[t_i, satisfies] + 1e-20)
    #         p_spaces.append(t_i_p_spaces)
    #
    #     return action_probs, p_spaces
