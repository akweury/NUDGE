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
    else:
        raise ValueError

    return idx_list


def get_game_info(args):

    if args.m == "threefish":
        obj_data = config.obj_name_threefish
    elif args.m == "getout":
        obj_data = config.obj_info_getout
    elif args.m == 'getoutplus':
        obj_data= config.obj_data_getoutplus
    elif args.m == 'Assault':
        obj_data = config.obj_info_assault
    else:
        raise ValueError

    return obj_data
