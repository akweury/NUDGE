# Created by jing at 31.01.24
import torch


def delta_from_states(states, game_ids, obj_a, obj_b, prop):
    deltas = []
    ids = game_ids.unique()

    for id in ids:
        id_states = states[game_ids == id]
        id_deltas = [False]

        for s_i in range(1, len(id_states)):
            last_dist = torch.abs(id_states[s_i - 1, obj_a, prop] - id_states[s_i - 1, obj_b, prop])
            dist = torch.abs(id_states[s_i, obj_a, prop] - id_states[s_i, obj_b, prop])
            if last_dist <= dist:
                delta = False
            else:
                delta = True
            id_deltas.append(delta)

        if len(id_deltas) > 1:
            id_deltas[0] = id_deltas[1]
        deltas += id_deltas

    return torch.tensor(deltas)
