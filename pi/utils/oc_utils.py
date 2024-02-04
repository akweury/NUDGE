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


def extract_logic_state_asterix(objects, args, noise=False):
    # print('Extracting logic states...')
    return
    extracted_states = {'Player': {'exist': False, 'x': [], 'y': []},
                        'Cauldron': {'exist': False, 'x': [], 'y': []},
                        'Enemy': {'exist': False, 'x': [], 'y': []}
                        }
    # import ipdb; ipdb.set_trace()
    for object in objects:
        if object.category == 'Player':
            extracted_states['Player']['exist'] = True
            extracted_states['Player']['x'].append(object.x)
            extracted_states['Player']['y'].append(object.y)
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif object.category == 'Cauldron':
            extracted_states['Cauldron']['exist'] = True
            extracted_states['Cauldron']['x'].append(object.x)
            extracted_states['Cauldron']['y'].append(object.y)
        elif object.category == 'Enemy':
            extracted_states['Enemy']['exist'] = True
            extracted_states['Enemy']['x'].append(object.x)
            extracted_states['Enemy']['y'].append(object.y)
        elif object.category == 'Reward50':
            pass
        elif object.category == "Score":
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


def extract_logic_state_getout(coin_jump, args, noise=False):
    if args.m == 'getoutplus':
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
                # extracted_states[3][-2:] /= 27
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

