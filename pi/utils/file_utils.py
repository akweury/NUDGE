# Created by jing at 31.01.24
import json
import torch
import pickle
import os

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
    # print('- Saved checkpoint file: {}'.format(filename))


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    # print('- Loaded checkpoint file: {}'.format(filename))
    return data




def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pkl(pkl_file, obj):
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f)


def all_file_in_folder(path_frames):
    paths = [os.path.join(path_frames, filename) for filename in os.listdir(path_frames) if
     os.path.isfile(os.path.join(path_frames, filename))]
    sorted_paths = sorted(paths, key=lambda f: os.stat(f).st_ctime)
    return sorted_paths


def save_agent(save_path, agent, env_args):
    save_dict = {"state_dict": agent.policy_net.state_dict(),
                 "learn_performance": agent.learn_performance,
                 "avg_score": torch.mean(env_args.win_rate)}
    torch.save(save_dict, save_path)