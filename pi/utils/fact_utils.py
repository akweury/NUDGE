# Created by jing at 31.01.24
import torch


def delta_from_states(states, game_ids, obj_a, obj_b, prop):
    deltas = torch.zeros(len(states), len(prop), dtype=torch.bool)

    for s_i in range(1, len(states)):
        # if previous state and current state have the same game id, then check their delta
        if game_ids[s_i] == game_ids[s_i-1]:
            for p_i in range(len(prop)):
                last_dist = torch.abs(states[s_i - 1, obj_a, prop[p_i]] - states[s_i - 1, obj_b, prop[p_i]])
                dist = torch.abs(states[s_i, obj_a, prop[p_i]] - states[s_i, obj_b, prop[p_i]])
                deltas[s_i, p_i] = last_dist >= dist
            # the delta of the first state equal to the delta of the second state in a game
            if game_ids[s_i-1] != game_ids[s_i-2]:
                deltas[s_i-1] = deltas[s_i]
    return deltas
