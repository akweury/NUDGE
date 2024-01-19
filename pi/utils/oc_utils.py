# Created by jing at 18.01.24
import torch


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
            states[player_missile_horizontal_id+i, player_missile_horizontal_exist_id] = 1
            states[player_missile_horizontal_id+i, x_idx] = extracted_states['PlayerMissileHorizontal']['x'][i]
            states[player_missile_horizontal_id+i, y_idx] = extracted_states['PlayerMissileHorizontal']['y'][i]
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
            states[enemy_missile_id+i, enemy_missile_exist_id] = 1
            states[enemy_missile_id+i, x_idx] = extracted_states['EnemyMissile']['x'][i]
            states[enemy_missile_id+i, y_idx] = extracted_states['EnemyMissile']['y'][i]
            if i > 1:
                raise ValueError

    return states
