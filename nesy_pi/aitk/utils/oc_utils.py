# Created by jing at 18.01.24
import torch
import numpy as np


def extract_logic_state_assault(objects, args, noise=False):
    extracted_states = {'Player': {'name': 'Player', 'exist': False, 'x': [], 'y': []},
                        'PlayerMissileVertical': {'name': 'PlayerMissileVertical', 'exist': False, 'x': [], 'y': []},
                        'PlayerMissileHorizontal': {'name': 'PlayerMissileHorizontal', 'exist': False, 'x': [],
                                                    'y': []},
                        'EnemyMissile': {'name': 'EnemyMissile', 'exist': False, 'x': [], 'y': []},
                        'Enemy': {'name': 'Enemy', 'exist': False, 'x': [], 'y': []}
                        }
    # import ipdb; ipdb.set_trace()
    for object in objects:
        if object.category == 'Player':
            extracted_states['Player']['exist'] = True
            extracted_states['Player']['x'].append(object.x)
            extracted_states['Player']['y'].append(object.y)
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif object.category == 'PlayerMissileVertical':
            extracted_states['PlayerMissileVertical']['exist'] = True
            extracted_states['PlayerMissileVertical']['x'].append(object.x)
            extracted_states['PlayerMissileVertical']['y'].append(object.y)
        elif object.category == 'PlayerMissileHorizontal':
            extracted_states['PlayerMissileHorizontal']['exist'] = True
            extracted_states['PlayerMissileHorizontal']['x'].append(object.x)
            extracted_states['PlayerMissileHorizontal']['y'].append(object.y)
        elif object.category == 'Enemy':
            extracted_states['Enemy']['exist'] = True
            extracted_states['Enemy']['x'].append(object.x)
            extracted_states['Enemy']['y'].append(object.y)
        elif object.category == 'EnemyMissile':
            extracted_states['EnemyMissile']['exist'] = True
            extracted_states['EnemyMissile']['x'].append(object.x)
            extracted_states['EnemyMissile']['y'].append(object.y)
        elif object.category == "MotherShip":
            pass
        elif object.category == "PlayerScore":
            pass
        elif object.category == "Health":
            pass
        elif object.category == "Lives":
            pass
        else:
            raise ValueError
    player_id = 0
    player_missile_vertical_id = 1
    player_missile_horizontal_id = 3
    enemy_id = 5
    enemy_missile_id = 10

    player_exist_id = 0
    player_missile_vertical_exist_id = 1
    player_missile_horizontal_exist_id = 2
    enemy_exist_id = 3
    enemy_missile_exist_id = 4
    x_idx = 5
    y_idx = 6

    states = torch.zeros((12, 7))
    if extracted_states['Player']['exist']:
        states[player_id, player_exist_id] = 1
        assert len(extracted_states['Player']['x']) == 1
        states[player_id, x_idx] = extracted_states['Player']['x'][0]
        states[player_id, y_idx] = extracted_states['Player']['y'][0]

    if extracted_states['PlayerMissileVertical']['exist']:
        for i in range(len(extracted_states['PlayerMissileVertical']['x'])):
            states[player_missile_vertical_id + i, player_missile_vertical_exist_id] = 1
            states[player_missile_vertical_id + i, x_idx] = extracted_states['PlayerMissileVertical']['x'][i]
            states[player_missile_vertical_id + i, y_idx] = extracted_states['PlayerMissileVertical']['y'][i]
            if i > 1:
                raise ValueError
    if extracted_states['PlayerMissileHorizontal']['exist']:
        for i in range(len(extracted_states['PlayerMissileHorizontal']['x'])):
            states[player_missile_horizontal_id + i, player_missile_horizontal_exist_id] = 1
            states[player_missile_horizontal_id + i, x_idx] = extracted_states['PlayerMissileHorizontal']['x'][i]
            states[player_missile_horizontal_id + i, y_idx] = extracted_states['PlayerMissileHorizontal']['y'][i]
            if i > 1:
                raise ValueError

    if extracted_states['Enemy']['exist']:
        for i in range(len(extracted_states['Enemy']['x'])):
            states[enemy_id + i, enemy_exist_id] = 1
            states[enemy_id + i, x_idx] = extracted_states['Enemy']['x'][i]
            states[enemy_id + i, y_idx] = extracted_states['Enemy']['y'][i]
            if i > 5:
                raise ValueError
    if extracted_states['EnemyMissile']['exist']:
        for i in range(len(extracted_states['EnemyMissile']['x'])):
            states[enemy_missile_id + i, enemy_missile_exist_id] = 1
            states[enemy_missile_id + i, x_idx] = extracted_states['EnemyMissile']['x'][i]
            states[enemy_missile_id + i, y_idx] = extracted_states['EnemyMissile']['y'][i]
            if i > 1:
                raise ValueError

    return states


def extract_obj_state_boxing(obj, obj_id, objt_len, prop_info, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    arm_max_length = 72
    obj_state[prop_info['left_arm_length']] = obj.left_arm_length / arm_max_length
    obj_state[prop_info['right_arm_length']] = obj.right_arm_length / arm_max_length

    obj_state[prop_info['axis_x_col']] = obj.center[0] / norm_factor
    obj_state[prop_info['axis_y_col']] = obj.center[1] / norm_factor
    return obj_state


def extract_obj_state_pong(obj, obj_id, objt_len, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[-6] = obj.center[0] / norm_factor - 0.5 * obj.wh[0] / norm_factor  # x1
    obj_state[-5] = obj.center[1] / norm_factor + 0.5 * obj.wh[1] / norm_factor  # y1
    obj_state[-4] = obj.center[0] / norm_factor + 0.5 * obj.wh[0] / norm_factor  # x2
    obj_state[-3] = obj.center[1] / norm_factor - 0.5 * obj.wh[1] / norm_factor  # y2
    obj_state[-2] = obj.center[0] / norm_factor
    obj_state[-1] = obj.center[1] / norm_factor
    return obj_state


def extract_obj_state_asterix(obj, obj_id, objt_len, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[-6] = obj.center[0] / norm_factor - 0.5 * obj.wh[0] / norm_factor  # x1
    obj_state[-5] = obj.center[1] / norm_factor + 0.5 * obj.wh[1] / norm_factor  # y1
    obj_state[-4] = obj.center[0] / norm_factor + 0.5 * obj.wh[0] / norm_factor  # x2
    obj_state[-3] = obj.center[1] / norm_factor - 0.5 * obj.wh[1] / norm_factor  # y2
    obj_state[-2] = obj.center[0] / norm_factor
    obj_state[-1] = obj.center[1] / norm_factor
    return obj_state
def extract_obj_state_freeway(obj, obj_id, objt_len, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[-6] = obj.center[0] / norm_factor - 0.5 * obj.wh[0] / norm_factor  # x1
    obj_state[-5] = obj.center[1] / norm_factor + 0.5 * obj.wh[1] / norm_factor  # y1
    obj_state[-4] = obj.center[0] / norm_factor + 0.5 * obj.wh[0] / norm_factor  # x2
    obj_state[-3] = obj.center[1] / norm_factor - 0.5 * obj.wh[1] / norm_factor  # y2
    obj_state[-2] = obj.center[0] / norm_factor
    obj_state[-1] = obj.center[1] / norm_factor
    return obj_state

def extract_obj_state_kangaroo(obj, obj_id, objt_len, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[-6] = obj.center[0] / norm_factor - 0.5 * obj.wh[0] / norm_factor  # x1
    obj_state[-5] = obj.center[1] / norm_factor + 0.5 * obj.wh[1] / norm_factor  # y1
    obj_state[-4] = obj.center[0] / norm_factor + 0.5 * obj.wh[0] / norm_factor  # x2
    obj_state[-3] = obj.center[1] / norm_factor - 0.5 * obj.wh[1] / norm_factor  # y2
    obj_state[-2] = obj.center[0] / norm_factor
    obj_state[-1] = obj.center[1] / norm_factor
    return obj_state


def extract_obj_state_frostbite(obj, obj_id, objt_len, prop_info, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[prop_info['axis_x_col']] = obj.center[0] / norm_factor
    obj_state[prop_info['axis_y_col']] = obj.center[1] / norm_factor
    return obj_state


def extract_obj_state_montezuma_revenge(obj, obj_id, objt_len, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[-2] = obj.center[0] / norm_factor
    obj_state[-1] = obj.center[1] / norm_factor
    return obj_state


def extract_obj_state_fishing_derby(obj, obj_id, objt_len, norm_factor):
    obj_state = torch.zeros(objt_len)
    obj_state[obj_id] = 1
    obj_state[-4] = obj.wh[0] / norm_factor
    obj_state[-3] = obj.wh[1] / norm_factor
    obj_state[-2] = obj.center[0] / norm_factor
    obj_state[-1] = obj.center[1] / norm_factor
    return obj_state


def extract_logic_state_atari(args, objects, game_info, norm_factor, noise=False):
    # print('Extracting logic states...')
    states = torch.zeros((game_info["state_row_num"], game_info["state_col_num"]))
    state_score = 0
    row_start = 0
    args.row_names = []

    # print(objects)
    for o_i, (obj_name, obj_num) in enumerate(game_info["obj_info"]):
        obj_count = 0
        for obj in objects:
            if obj.category == obj_name:
                if obj_count >= obj_num:
                    continue
                if game_info['name'] == 'Boxing':
                    states[row_start + obj_count] = extract_obj_state_boxing(obj, o_i, game_info["state_col_num"],
                                                                             game_info['prop_info'], norm_factor)
                elif game_info['name'] == 'Pong':
                    states[row_start + obj_count] = extract_obj_state_pong(obj, o_i, game_info["state_col_num"],
                                                                           norm_factor)
                elif game_info['name'] == 'Asterix':
                    states[row_start + obj_count] = extract_obj_state_asterix(obj, o_i, game_info["state_col_num"],
                                                                              norm_factor)
                elif game_info['name'] == 'Freeway':
                    states[row_start + obj_count] = extract_obj_state_freeway(obj, o_i, game_info["state_col_num"],
                                                                              norm_factor)
                elif game_info['name'] == 'Kangaroo':
                    states[row_start + obj_count] = extract_obj_state_kangaroo(obj, o_i, game_info["state_col_num"],
                                                                               norm_factor)
                elif game_info['name'] == 'Frostbite':
                    states[row_start + obj_count] = extract_obj_state_frostbite(obj, o_i, game_info["state_col_num"],
                                                                                game_info['prop_info'], norm_factor)
                elif game_info['name'] == 'montezuma_revenge':
                    states[row_start + obj_count] = extract_obj_state_montezuma_revenge(obj, o_i,
                                                                                        game_info["state_col_num"],
                                                                                        norm_factor)
                elif game_info['name'] == 'fishing_derby':
                    states[row_start + obj_count] = extract_obj_state_fishing_derby(obj, o_i,
                                                                                    game_info["state_col_num"],
                                                                                    norm_factor)
                else:
                    raise ValueError
                obj_count += 1
            # elif obj.category == "Score":
            #     state_score = obj.value
        row_start += obj_num
        args.row_names += [obj_name] * obj_num
    # draw_utils.plot_heat_map(states, args.output_folder, game_info['name'], row_names=row_names)
    return states.tolist(), state_score


def extract_logic_state_getout(coin_jump, args, noise=False):
    if args.hardness == 1:
        num_of_feature = 6
        num_of_object = 8
        representation = coin_jump.level.get_representation()
        # import ipdb; ipdb.set_trace()
        extracted_states = np.zeros((num_of_object, num_of_feature))
        for entity in representation["entities"]:
            if entity[0].name == 'PLAYER':
                extracted_states[0][0] = 1
                extracted_states[0][-2:] = entity[1:3]
                # 27 is the width of map, this is normalization
                # extracted_states[0][-2:] /= 27
            elif entity[0].name == 'KEY':
                extracted_states[1][1] = 1
                extracted_states[1][-2:] = entity[1:3]
                # extracted_states[1][-2:] /= 27
            elif entity[0].name == 'DOOR':
                extracted_states[2][2] = 1
                extracted_states[2][-2:] = entity[1:3]
                # extracted_states[2][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY':
                extracted_states[3][3] = 1
                extracted_states[3][-2:] = entity[1:3]
                # extracted_states[3][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY2':
                extracted_states[4][3] = 1
                extracted_states[4][-2:] = entity[1:3]
            elif entity[0].name == 'GROUND_ENEMY3':
                extracted_states[5][3] = 1
                extracted_states[5][-2:] = entity[1:3]
            elif entity[0].name == 'BUZZSAW1':
                extracted_states[6][3] = 1
                extracted_states[6][-2:] = entity[1:3]
            elif entity[0].name == 'BUZZSAW2':
                extracted_states[7][3] = 1
                extracted_states[7][-2:] = entity[1:3]
    else:
        """
        extract state to metric
        input: coin_jump instance
        output: extracted_state to be explained
        set noise to True to add noise

        x:  agent, key, door, enemy, position_X, position_Y
        y:  obj1(agent), obj2(key), obj3(door)，obj4(enemy)

        To be changed when using object-detection tech
        """
        num_of_feature = 6
        num_of_object = 4
        representation = coin_jump.level.get_representation()
        extracted_states = np.zeros((num_of_object, num_of_feature))
        for entity in representation["entities"]:
            if entity[0].name == 'PLAYER':
                extracted_states[0][0] = 1
                extracted_states[0][-2:] = entity[1:3]
                # 27 is the width of map, this is normalization
                # extracted_states[0][-2:] /= 27
            elif entity[0].name == 'KEY':
                extracted_states[1][1] = 1
                extracted_states[1][-2:] = entity[1:3]
                # extracted_states[1][-2:] /= 27
            elif entity[0].name == 'DOOR':
                extracted_states[2][2] = 1
                extracted_states[2][-2:] = entity[1:3]
                # extracted_states[2][-2:] /= 27
            elif entity[0].name == 'GROUND_ENEMY':
                extracted_states[3][3] = 1
                extracted_states[3][-2:] = entity[1:3]
                # extracted_states[3][-2:] /= 27

    if sum(extracted_states[:, 1]) == 0:
        key_picked = True
    else:
        key_picked = False

    def simulate_prob(extracted_states, num_of_objs, key_picked):
        for i, obj in enumerate(extracted_states):
            obj = add_noise(obj, i, num_of_objs)
            extracted_states[i] = obj
        if key_picked:
            extracted_states[:, 1] = 0
        return extracted_states

    def add_noise(obj, index_obj, num_of_objs):
        mean = torch.tensor(0.2)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    if noise:
        extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)
    states = torch.tensor(np.array(extracted_states), dtype=torch.float32, device="cpu").unsqueeze(0)
    return states
