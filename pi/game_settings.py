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

    elif args.m == "getout":
        idx_x = config.state_idx_getout_x
        idx_list.append([idx_x])
    else:
        raise ValueError

    return idx_list


def get_game_info(args):

    if args.m == "threefish":
        obj_types = config.obj_name_threefish
    elif args.m == "getout":
        obj_types = config.obj_type_name_getout
    else:
        raise ValueError

    if args.env == 'getout':
        obj_names = config.obj_type_indices_getout
    elif args.env == 'getoutplus':
        obj_names = config.obj_type_indices_getout_plus
    else:
        raise ValueError

    return obj_types, obj_names
