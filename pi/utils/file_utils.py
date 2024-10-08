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
import datetime
def create_log_file(exp_output_path, name):
    date_now = datetime.datetime.today().date()
    time_now = datetime.datetime.now().strftime("%H_%M_%S")

    file_name = str(exp_output_path / f"log_{name}.txt")
    with open(file_name, "w") as f:
        f.write(f"Log from {date_now}, {time_now}")

    return str(exp_output_path / file_name)

def add_lines(line_str, log_file):
    print(line_str)
    with open(log_file, "a") as f:
        f.write(str(line_str) + "\n")
