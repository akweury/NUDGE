# Created by jing at 31.01.24
import json
import torch
import pickle


def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
    print('- Saved checkpoint file: {}'.format(filename))


def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    print('- Loaded checkpoint file: {}'.format(filename))
    return data


def save_pt(data, filename):
    torch.save(data, filename)


def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def save_pkl(pkl_file, obj):
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f)
