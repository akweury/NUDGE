import csv
import time
import gym3
import numpy as np
import ast
import pandas as pd
import sys
import io

from ocatari.core import OCAtari
from PIL import Image, ImageDraw

from src.environments.procgen.procgen import ProcgenGym3Env
from src.environments.getout.getout.imageviewer import ImageViewer
from src.environments.getout.getout.getout.paramLevelGenerator import ParameterizedLevelGenerator
from src.environments.getout.getout.getout.getout import Getout
from src.environments.getout.getout.getout.actions import GetoutActions

# font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
# font = ImageFont.truetype(font_path, size=40)
disp_text = None
repeated = 10


def hexify(la):
    return hex(int("".join([str(l) for l in la])))


def run(env, nb_games=20):
    """
    Display a window to the user and loop until the window is closed
    by the user.
    """
    prev_time = env._renderer.get_time()
    env._renderer.start()
    env._draw()
    env._renderer.finish()

    old_stdout = sys.stdout  # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()

    # whatWasPrinted = buffer.getvalue()  # Return a str containing the entire contents of the buffer.
    while buffer.getvalue().count("final info") < nb_games:
        now = env._renderer.get_time()
        dt = now - prev_time
        prev_time = now
        if dt < env._sec_per_timestep:
            sleep_time = env._sec_per_timestep - dt
            time.sleep(sleep_time)

        keys_clicked, keys_pressed = env._renderer.start()
        if "O" in keys_clicked:
            env._overlay_enabled = not env._overlay_enabled
        env._update(dt, keys_clicked, keys_pressed)
        env._draw()
        env._renderer.finish()
        if not env._renderer.is_open:
            break
    sys.stdout = old_stdout  # Put the old stream back in place
    all_summaries = [line for line in buffer.getvalue().split("\n") if line.startswith("final")]
    return all_summaries


def get_values(summaries, key_str, stype=float):
    all_values = []
    for line in summaries:
        dico = ast.literal_eval(line[11:])
        all_values.append(stype(dico[key_str]))
    return all_values


def render_getout(agent, args):
    # envname = "getout"
    KEY_SPACE = 32
    # KEY_SPACE = 32
    KEY_w = 119
    KEY_a = 97
    KEY_s = 115
    KEY_d = 100
    KEY_r = 114
    KEY_LEFT = 65361
    KEY_RIGHT = 65363
    KEY_UP = 65362

    def setup_image_viewer(getout):
        print(getout.camera.height, getout.camera.width)
        viewer = ImageViewer(
            "getout",
            getout.camera.height,
            getout.camera.width,
            monitor_keyboard=True,
        )
        return viewer

    def create_getout_instance(args, seed=None):
        if args.m == 'getoutplus':
            enemies = True
        else:
            enemies = False
        # level_generator = DummyGenerator()
        getout = Getout()
        level_generator = ParameterizedLevelGenerator(enemies=enemies)
        level_generator.generate(getout, seed=seed)
        getout.render()

        return getout

    # seed = random.randint(0, 100000000)
    # print(seed)
    getout = create_getout_instance(args)
    viewer = setup_image_viewer(getout)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    num_epi = 1
    max_epi = 100
    win_count = 0
    game_count = 0
    total_reward = 0
    epi_reward = 0
    step = 0
    game_states = []
    last_explaining = None
    scores = []
    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.agent == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
        elif args.agent == 'ppo' or args.agent == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
        elif args.agent == 'human':
            head = ['episode', 'reward']
        writer.writerow(head)

    decision_history = []
    max_game_frames = 200
    game_frame_counter = 0
    while num_epi <= max_epi:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time  # save frame start time for next iteration
        # step game
        step += 1
        action = []
        explaining = None
        # predict action
        if not getout.level.terminated:
            if game_frame_counter > max_game_frames:
                game_count += 1
                agent.revise_timeout(decision_history)
                getout = create_getout_instance(args)
                decision_history = []
                # print("epi_reward: ", round(epi_reward, 2))
                # print("--------------------------     next game    --------------------------")
                print(f"Episode {num_epi} Win: {win_count}/{game_count}")
                print(f"==========")
                if args.agent_type == 'human':
                    data = [(num_epi, round(epi_reward, 2))]
                    # writer.writerows(data)
                total_reward += epi_reward
                epi_reward = 0
                action = 0
                # average_reward = round(total_reward / num_epi, 2)
                num_epi += 1
                step = 0
                game_states = []

            if args.agent_type in ['logic', "smp"]:
                action, explaining = agent.act(getout)

            elif args.agent_type == 'ppo':
                action = agent.act(getout)
            elif args.agent_type == 'human':
                if KEY_a in viewer.pressed_keys or KEY_LEFT in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_LEFT)
                if KEY_d in viewer.pressed_keys or KEY_RIGHT in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_RIGHT)
                if (KEY_SPACE in viewer.pressed_keys) or (
                        KEY_w in viewer.pressed_keys) or KEY_UP in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_UP)
                if KEY_s in viewer.pressed_keys:
                    action.append(GetoutActions.MOVE_DOWN)
            elif args.agent_type == 'random':
                action = agent.act(getout)

            reward = getout.step(action)
            explaining["reward"].append(reward)
            decision_history.append(explaining)
            for beh_i in explaining['behavior_index']:
                print(f"f: {game_frame_counter}, rw: {reward}, behavior: {agent.model.explains[beh_i]}")
            game_states.append(explaining['state'])
            # print(reward)
            epi_reward += reward

        else:
            game_count += 1
            if epi_reward > 1:
                win_count += 1
                agent.revise_win(decision_history, args.obj_info)
            else:
                lost_game_data = agent.revise_loss(decision_history, args.obj_info)
                agent.update_lost_buffer(lost_game_data)
                def_behaviors = agent.reasoning_def_behaviors()
                agent.update_behaviors(None, def_behaviors, args)
                print("- revise loss finished.")
            getout = create_getout_instance(args)
            decision_history = []
            game_frame_counter = 0
            # print("epi_reward: ", round(epi_reward, 2))
            # print("--------------------------     next game    --------------------------")
            print(f"Episode {num_epi} Win: {win_count}/{game_count}")
            print(f"===============================================")
            if args.agent_type == 'human':
                data = [(num_epi, round(epi_reward, 2))]
                # writer.writerows(data)
            total_reward += epi_reward
            epi_reward = 0
            action = 0
            # average_reward = round(total_reward / num_epi, 2)
            num_epi += 1
            step = 0

        game_frame_counter += 1

        if args.render:
            screen = getout.camera.screen
            ImageDraw.Draw(screen).text((40, 60), "", (120, 20, 20))
            np_img = np.asarray(screen)
            viewer.show(np_img[:, :, :3])
            if args.record:
                screen.save(f"renderings/{step:03}.png")

        # terminated = getout.level.terminated
        # if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

        if getout.level.terminated:
            if args.record:
                exit()
            step = 0
            scores.append(epi_reward)
        if num_epi > 100:
            break

    # df = pd.DataFrame({'reward': scores})
    # df.to_csv(f"logs/{envname}/random_{envname}_log_{args.seed}.csv", index=False)
    # df.to_csv(f"logs/{envname}/{args.agent}_{envname}_log_{args.seed}.csv", index=False)
    # print(f"saved in logs/{envname}/{args.agent}_{envname}_log_{args.seed}.csv")


def render_threefish(agent, args):
    envname = args.env
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", rand_seed=args.seed, start_level=args.seed)

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.agent == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
            writer.writerow(head)
        elif args.agent == 'ppo' or args.agent == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
            writer.writerow(head)

    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        all_summaries = run(ia, 10)

        df_scores = get_values(all_summaries, "episode_return")
        data = {'reward': df_scores}
        # convert list to df_scores
        # pd.to_csv(df_scores, f"{player_name}_scores.csv")
        df = pd.DataFrame(data)
        df.to_csv(args.logfile, index=False)

    else:
        if args.render:
            env = gym3.ViewerWrapper(env, info_key="rgb", height=600, width=900)
        reward, obs, done = env.observe()
        scores = []
        last_explaining = None
        for epi in range(20):
            print(f"Episode {epi}")
            print(f"==========")
            total_r = 0
            step = 0
            while True:
                step += 1
                if args.agent == 'logic':
                    action, explaining = agent.act(obs)
                else:
                    action = agent.act(obs)
                env.act(action)
                rew, obs, done = env.observe()
                total_r += rew[0]
                if args.agent == 'logic':
                    if last_explaining is None or (explaining != last_explaining and repeated > 4):
                        print(explaining)
                        last_explaining = explaining
                        disp_text = explaining
                        repeated = 0
                    repeated += 1
                if done:
                    step = 0
                    print("episode: ", epi)
                    print("return: ", total_r)
                    scores.append(total_r)
                    if args.record:
                        exit()
                    break
                if epi > 100:
                    break
                if args.record:
                    screen = Image.fromarray(env._get_image())
                    ImageDraw.Draw(screen).text((40, 60), disp_text, (120, 20, 20))
                    screen.save(f"renderings/{step:03}.png")

            df = pd.DataFrame({'reward': scores})
            df.to_csv(f"logs/{envname}/{args.agent}_{envname}_log_{args.seed}.csv", index=False)


def render_loot(agent, args):
    envname = args.env
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", rand_seed=args.seed, start_level=args.seed)

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)
        if args.agent == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
            writer.writerow(head)
        elif args.agent == 'ppo' or args.agent == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
            writer.writerow(head)

    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768 * 2, width=768 * 2)
        all_summaries = run(ia, 20)

        scores = get_values(all_summaries, "episode_return")
        df = pd.DataFrame({'reward': scores})
        df.to_csv(args.logfile, index=False)
    else:
        if args.render:
            env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        scores = []
        last_explaining = None
        nsteps = 0
        for epi in range(20):
            print(f"Episode {epi}")
            print(f"==========")
            total_r = 0
            step = 0
            while True:
                step += 1
                nsteps += 1
                if args.agent == 'logic':
                    # print(obs['positions'])
                    action, explaining = agent.act(obs)
                else:
                    action = agent.act(obs)
                env.act(action)
                rew, obs, done = env.observe()
                total_r += rew[0]
                if args.agent == 'logic':
                    if last_explaining is None or (explaining != last_explaining and repeated > 2):
                        # print(explaining)
                        last_explaining = explaining
                        disp_text = explaining
                        repeated = 0
                    repeated += 1

                if args.record:
                    screen = Image.fromarray(env._get_image())
                    ImageDraw.Draw(screen).text((40, 60), disp_text, (20, 170, 20))
                    screen.save(f"renderings/{nsteps:03}.png")

                # if args.log:
                #     if args.agent == 'logic':
                #         probs = agent.get_probs()
                #         logic_state = agent.get_state(obs)
                #         data = [(epi, step, rew[0], average_r, logic_state, probs)]
                #         writer.writerows(data)
                #     else:
                #         data = [(epi, step, rew[0], average_r)]
                #         writer.writerows(data)

                if done:
                    # step = 0
                    print("episode: ", epi)
                    print("return: ", total_r)
                    scores.append(total_r)
                    break

                if epi > 100:
                    break

            # import ipdb; ipdb.set_trace()
            df = pd.DataFrame({'reward': scores})
            df.to_csv(f"logs/{envname}/{args.agent}_{envname}_log_{args.seed}.csv", index=False)


def render_ecoinrun(agent, args):
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array", seed=args.seed)
    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        ia.run()
    else:
        env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        i = 0
        while True:
            action = agent.act(obs)
            env.act(action)
            rew, obs, done = env.observe()
            # if i % 40 == 0:
            #     print("\n" * 50)
            #     print(obs["positions"])
            i += 1


def render_atari(agent, args):
    # gamename = 
    rdr_mode = "human" if args.render else "rgb_array"
    env = OCAtari(env_name=args.env.capitalize(), render_mode=rdr_mode, mode="revised")
    obs = env.reset()
    # from pprint import pprint
    try:
        agent.nb_actions = env.nb_actions
    except:
        pass
    scores = []
    nb_epi = 20
    for epi in range(nb_epi):
        total_r = 0
        step = 0
        print(f"Episode {epi}")
        print(f"==========")
        while True:
            # action = random.randint(0, env.nb_actions-1)
            if args.agent == 'logic':
                action, explaining = agent.act(env.objects)
                print(action, explaining)
            elif args.agent == 'random':
                action = np.random.randint(env.nb_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            step += 1
            # if step % 10 == 0:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(env._get_obs())
            #     plt.show()
            if terminated:
                print("episode: ", epi)
                print("return: ", total_r)
                scores.append(total_r)
                env.reset()
                step = 0
                break
    print(np.average(scores))
