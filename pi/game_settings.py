# Created by jing at 29.11.23
from src import config


def get_idx(args):
    idx_list = []
    if args.m == "threefish":
        idx_x = config.state_idx_threefish_x
        idx_y = config.state_idx_threefish_y
        idx_radius = config.state_idx_threefish_radius
        idx_list.append([idx_x])
        idx_list.append([idx_y])
        idx_list.append([idx_radius])

    elif args.m == "getout" or args.m == "getoutplus":
        idx_x = config.state_idx_getout_x
        idx_y = config.state_idx_getout_y
        idx_list.append(idx_x)
        idx_list.append(idx_y)
    elif args.m == "Assault":
        idx_x = config.state_idx_assault_x
        idx_y = config.state_idx_assault_y
        idx_list.append(idx_x)
        idx_list.append(idx_y)
    elif args.m in ["Asterix", "Boxing", "Breakout", "Freeway", "Kangaroo"]:
        idx_x = args.game_info["axis_x_col"]
        idx_y = args.game_info["axis_y_col"]
        idx_list.append(idx_x)
        idx_list.append(idx_y)
    else:
        raise ValueError

    return idx_list


def get_game_info(args):
    if args.m == "threefish":
        obj_data = config.obj_name_threefish
    elif args.m == "getout":
        obj_data = config.obj_info_getout
    elif args.m == 'getoutplus':
        obj_data = config.obj_data_getoutplus
    elif args.m == 'Assault':
        obj_data = config.obj_info_assault
    elif args.m == 'Asterix':
        obj_data = config.obj_info_asterix
    else:
        raise ValueError

    return obj_data


def atari_obj_info(obj_info):
    obj_counter = 0
    info = []
    for o_i, (obj_name, obj_num) in enumerate(obj_info):
        info.append({"name": obj_name,
                     "indices": list(range(obj_counter, obj_counter + obj_num))})
        obj_counter += obj_num
    return info


def switch_hardness(args):
    if args.m == "getout" and args.hardness == 0:
        args.game_info = config.game_info_getout
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = atari_obj_info(args.obj_info)
    if args.m == "getout" and args.hardness == 1:
        args.game_info = config.game_info_getoutplus
        args.obj_info = args.game_info["obj_info"]
        args.obj_info = atari_obj_info(args.obj_info)
    return args
