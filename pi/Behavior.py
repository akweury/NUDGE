# Created by shaji at 21/12/2023


class Behavior():
    def __init__(self, counter_action, neural_action, state, explains):

        self.neural_action = neural_action
        self.state = state
        self.counter_action = counter_action
        self.explains = explains


class Explain():
    def __init__(self,mask_name, behavior_pred, obj_idx, prop_idx, parameter):
        self.pred = behavior_pred
        self.parameters = parameter
        self.grounded_objs = obj_idx
        self.grounded_prop = prop_idx
        self.mask = mask_name