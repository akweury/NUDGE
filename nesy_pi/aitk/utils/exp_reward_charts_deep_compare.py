# Created by shaji at 13/08/2024


from pathlib import Path
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import pandas as pd

from src import config
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
marks = ['o', "*", "x"]
colors = ['skyblue', 'salmon', 'lightgreen', "orchid", "gold"]


def thousands(x, pos):
    return '%1.0fK' % (x * 1e-3)


def smooth_tensor(window_size, tensor):
    # Compute the cumulative sum

    # Initialize the smoothed tensor
    smoothed_tensor = torch.zeros_like(tensor)

    # Compute the rolling average
    for i in range(len(tensor)):
        if i < window_size - 1:
            smoothed_tensor[i] = tensor[i] / (i + 1)
        else:
            smoothed_tensor[i] = (tensor[i - window_size: i]).sum() / window_size
    return smoothed_tensor


def conf_interval(data, color):
    std_dev = torch.std(data)
    # Plot confidence intervals
    lower_bound = data - 1.96 * std_dev
    upper_bound = data + 1.96 * std_dev
    return lower_bound, upper_bound


window_size = 20

getout_reward_file = Path(config.path_nesy / "wandb" / "wandb_reward_getout.csv")
loot_reward_file = Path(config.path_nesy / "wandb" / "wandb_reward_loot.csv")
threefish_reward_file = Path(config.path_nesy / "wandb" / "wandb_reward_threefish.csv")

#
# df = pd.read_csv(getout_reward_file)
# ppo_df = pd.read_csv(getout_ppo_file)
# getout_steps = torch.tensor(df[df.columns[0]].array)
# getout_phi8 = torch.tensor(df[df.columns[1]].array)
# getout_phi1 = torch.tensor(df[df.columns[4]].array)
# getout_phi90 = torch.tensor(df[df.columns[10]].array)
# getout_phi180 = torch.tensor(df[df.columns[7]].array)
# getout_ppo = torch.tensor(ppo_df[ppo_df.columns[1]].array)[:500]
# getout_ppo_scale_factor = (3.8 - getout_ppo.min()) / (getout_ppo.max() - getout_ppo.min())
# getout_ppo = getout_ppo_scale_factor * getout_ppo + (getout_ppo.min() - getout_ppo_scale_factor * getout_ppo.min())


axis = 0
df = pd.read_csv(getout_reward_file)
steps = torch.tensor(df[df.columns[0]].array)
phi8 = torch.tensor(df[df.columns[1]].array)
phi1 = torch.tensor(df[df.columns[4]].array)
phi10 = torch.tensor(df[df.columns[7]].array)



axs[axis].text(steps[300], 10, 'phi90-rho100', va='center', ha='left', color='black', fontsize=12)
axs[axis].text(steps[350], -10, 'phi8-rho10', va='center', ha='left', color='black', fontsize=12)
axs[axis].text(steps[400], -30, 'phi1-rho1', va='center', ha='left', color='black', fontsize=12)

# axs[axis].axhline(5.7, color='green', linestyle='--')  # Add horizontal dashed line at y=1
# axs[axis].text(steps[5], 5.5, 'human', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[50], 3.5, 'EXPIL', va='center', ha='left', color='black', fontsize=12)
# axs[axis].axhline(y=0.6, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
# axs[axis].text(steps[70], 0.4, 'random', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[170], 0.7, 'PPO', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[150], -0.6, 'NUDGE', va='center', ha='left', color='black', fontsize=12)

phi1 = smooth_tensor(window_size, phi1)
phi8 = smooth_tensor(window_size, phi8)
phi10 = smooth_tensor(window_size, phi10)


# loot_ppo = smooth_tensor(window_size, loot_ppo)
axs[axis].plot(steps[:500], phi10[:500], color=colors[0])
axs[axis].plot(steps[:500], phi8[:500], color=colors[1])
axs[axis].plot(steps[:500], phi1[:500], color=colors[2])

lb_phi1, ub_phi1 = conf_interval(phi1[:500], colors[2])
lb_phi8, ub_phi8 = conf_interval(phi8[:500], colors[1])
lb_phi10, ub_phi10 = conf_interval(phi10[:500], colors[0])

axs[axis].fill_between(steps[:500], lb_phi1, ub_phi1, color=colors[2], alpha=0.3)
axs[axis].fill_between(steps[:500], lb_phi8, ub_phi8, color=colors[1], alpha=0.3)
axs[axis].fill_between(steps[:500], lb_phi10, ub_phi10, color=colors[0], alpha=0.3)

# axs[1].fill_between(loot_steps[:200], lb_loot_ppo, ub_loot_ppo, color=colors[0], alpha=0.3)

axs[axis].set_title('Getout')

formatter = ticker.FuncFormatter(thousands)
plt.gca().xaxis.set_major_formatter(formatter)



loot_df = pd.read_csv(loot_reward_file)
loot_steps = torch.tensor(loot_df[loot_df.columns[0]].array)
loot_rho8 = torch.tensor(loot_df[loot_df.columns[1]].array)
loot_phi8 = torch.tensor(loot_df[loot_df.columns[4]].array)
loot_phirho8 = torch.tensor(loot_df[loot_df.columns[7]].array)
loot_phi10 = torch.tensor(loot_df[loot_df.columns[10]].array)
# loot_phi1 = torch.tensor(loot_df[loot_df.columns[16]].array)

threefish_df = pd.read_csv(threefish_reward_file)
df = threefish_df
threefish_steps = torch.tensor(df[df.columns[0]].array)
threefish_rho8 = torch.tensor(df[df.columns[1]].array)
threefish_phi8 = torch.tensor(df[df.columns[4]].array)
threefish_phirho8 = torch.tensor(df[df.columns[7]].array)
threefish_phi10 = torch.tensor(df[df.columns[10]].array)
# threefish_phi1 = torch.tensor(df[df.columns[16]].array)

#
# threefish_root = Path("E:\\projects\\storage\\check_point\\threefish\\trained_models")
# threefish_reward_file = threefish_root / "training_reward.csv"
# threefish_ppo_file = threefish_root / "threefish_ppo.csv"
# threefish_df = pd.read_csv(threefish_reward_file)
# threefish_ppo_df = pd.read_csv(threefish_ppo_file)
# threefish_steps = torch.tensor(threefish_df[threefish_df.columns[0]].array)
# threefish_phi8 = torch.tensor(threefish_df[threefish_df.columns[4]].array)
# threefish_phi1 = torch.tensor(threefish_df[threefish_df.columns[1]].array)
# threefish_ppo = torch.tensor(threefish_ppo_df[threefish_ppo_df.columns[1]].array)[:500]
# threefish_ppo_scale_factor = (-0.4 - threefish_ppo.min()) / (threefish_ppo.max() - threefish_ppo.min())
# threefish_ppo = threefish_ppo_scale_factor * threefish_ppo + (
#             threefish_ppo.min() - threefish_ppo_scale_factor * threefish_ppo.min())
# Create subplots with 1 row and 4 columns

# getout_phi1 = smooth_tensor(window_size, getout_phi1)
# getout_phi8 = smooth_tensor(window_size, getout_phi8)
# getout_phi90 = smooth_tensor(window_size, getout_phi90)
# getout_ppo = smooth_tensor(window_size, getout_ppo)

# axs[0].plot(getout_steps, getout_phi1, color=colors[2])
# lb_getout_phi1, ub_getout_phi1 = conf_interval(getout_phi1, colors[2])
# axs[0].fill_between(getout_steps, lb_getout_phi1, ub_getout_phi1, color=colors[2], alpha=0.3)

# axs[0].plot(getout_steps, getout_phi90, color=colors[1])
# lb_getout_phi90, ub_getout_phi90 = conf_interval(getout_phi90, colors[1])
# lb_getout_phi90 = smooth_tensor(window_size, lb_getout_phi90)
# ub_getout_phi90 = smooth_tensor(window_size, ub_getout_phi90)

# axs[0].fill_between(getout_steps, lb_getout_phi90, ub_getout_phi90, color=colors[1], alpha=0.3)

# axs[0].plot(getout_steps, getout_ppo, color=colors[0])
# lb_getout_ppo, ub_getout_ppo = conf_interval(getout_ppo, colors[0])
# lb_getout_ppo = smooth_tensor(window_size, lb_getout_ppo)
# ub_getout_ppo = smooth_tensor(window_size,ub_getout_ppo)
# axs[0].fill_between(getout_steps, lb_getout_ppo, ub_getout_ppo, color=colors[0], alpha=0.3)
# axs[0].plot(getout_phi8, color=colors[0])

# axs[0].axhline(y=13.5, color='green', linestyle='--')  # Add horizontal dashed line at y=1
# axs[0].text(getout_steps[1], 15, 'human', va='center', ha='left', color='black', fontsize=12)
# axs[0].text(getout_steps[-20], 10, 'EXPIL', va='center', ha='left', color='black', fontsize=12)
# axs[0].text(getout_steps[-25], 0, 'PPO', va='center', ha='left', color='black', fontsize=12)
# axs[0].text(getout_steps[-25], -30, 'NUDGE', va='center', ha='left', color='black', fontsize=12)

# axs[0].axhline(y=-22.5, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
# axs[0].text(getout_steps[-20], -21, 'random', va='center', ha='left', color='black', fontsize=12)

# axs[0].set_title('Getout')
# axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer

# formatter = ticker.FuncFormatter(thousands)
# plt.gca().xaxis.set_major_formatter(formatter)

axis = 1
axs[axis].text(steps[50], 4.5, 'phi8', va='center', ha='left', color='black', fontsize=12)
axs[axis].text(steps[50], 3.6, 'phi10', va='center', ha='left', color='black', fontsize=12)
axs[axis].text(steps[120], 3, 'phi8-rho8', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[100], -0.6, 'phi1-rho1', va='center', ha='left', color='black', fontsize=12)


# loot_phi1 = smooth_tensor(window_size, loot_phi1)
loot_phi8 = smooth_tensor(window_size, loot_phi8)
loot_phi10 = smooth_tensor(window_size, loot_phi10)
loot_rho8 = smooth_tensor(window_size, loot_rho8)
loot_phirho8 = smooth_tensor(window_size, loot_phirho8)

# loot_ppo = smooth_tensor(window_size, loot_ppo)

axs[axis].plot(loot_steps[:600], loot_phi10[:600], color=colors[0])
axs[axis].plot(loot_steps[:600], loot_phi8[:600], color=colors[1])
# axs[axis].plot(loot_steps[:500], loot_phi1[:500], color=colors[2])
# axs[axis].plot(loot_steps[:600], loot_rho8[:600], color=colors[3])
axs[axis].plot(loot_steps[:600], loot_phirho8[:600], color=colors[4])

# lb_loot_phi1, ub_loot_phi1 = conf_interval(loot_phi1[:500], colors[2])
lb_loot_phi8, ub_loot_phi8 = conf_interval(loot_phi8[:600], colors[1])
lb_loot_phi10, ub_loot_phi10 = conf_interval(loot_phi10[:600], colors[0])
# lb_loot_rho8, ub_loot_rho8 = conf_interval(loot_rho8[:600], colors[3])
lb_loot_phirho8, ub_loot_phirho8 = conf_interval(loot_phirho8[:600], colors[4])

# axs[axis].fill_between(loot_steps[:500], lb_loot_phi1, ub_loot_phi1, color=colors[2], alpha=0.3)
axs[axis].fill_between(loot_steps[:600], lb_loot_phi8, ub_loot_phi8, color=colors[1], alpha=0.3)
axs[axis].fill_between(loot_steps[:600], lb_loot_phi10, ub_loot_phi10, color=colors[0], alpha=0.3)
# axs[axis].fill_between(loot_steps[:600], lb_loot_rho8, ub_loot_rho8, color=colors[3], alpha=0.3)
axs[axis].fill_between(loot_steps[:600], lb_loot_phirho8, ub_loot_phirho8, color=colors[4], alpha=0.3)

# axs[1].fill_between(loot_steps[:200], lb_loot_ppo, ub_loot_ppo, color=colors[0], alpha=0.3)

axs[axis].set_title('Loot')

formatter = ticker.FuncFormatter(thousands)
plt.gca().xaxis.set_major_formatter(formatter)

axis = 2
df = pd.read_csv(threefish_reward_file)
steps = torch.tensor(df[df.columns[0]].array)
rho8 = torch.tensor(df[df.columns[4]].array)
phirho8 = torch.tensor(df[df.columns[10]].array)
phi10 = torch.tensor(df[df.columns[13]].array)
# phi1 = torch.tensor(df[df.columns[13]].array)
phi8 = torch.tensor(df[df.columns[19]].array)

# axs[axis].text(steps[400], 0.3, 'phi8', va='center', ha='left', color='black', fontsize=12)
axs[axis].text(steps[450], 0.0, 'phi10', va='center', ha='left', color='black', fontsize=12)
axs[axis].text(steps[300], -0.5, 'phi10-rho10', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[300], -0.8, 'phi1-rho1', va='center', ha='left', color='black', fontsize=12)

# axs[axis].axhline(5.7, color='green', linestyle='--')  # Add horizontal dashed line at y=1
# axs[axis].text(steps[5], 5.5, 'human', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[50], 3.5, 'EXPIL', va='center', ha='left', color='black', fontsize=12)
# axs[axis].axhline(y=0.6, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
# axs[axis].text(steps[70], 0.4, 'random', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[170], 0.7, 'PPO', va='center', ha='left', color='black', fontsize=12)
# axs[axis].text(steps[150], -0.6, 'NUDGE', va='center', ha='left', color='black', fontsize=12)

# phi1 = smooth_tensor(window_size, phi1)
phi8 = smooth_tensor(window_size, phi8)
phi10 = smooth_tensor(window_size, phi10)
rho8 = smooth_tensor(window_size, rho8)
phirho8 = smooth_tensor(window_size, phirho8)

# loot_ppo = smooth_tensor(window_size, loot_ppo)
axs[axis].plot(steps[:1600], phi10[:1600], color=colors[0])
# axs[axis].plot(steps[:], phi8[:], color=colors[1])
# axs[axis].plot(steps[:500], phi1[:500], color=colors[2])
# axs[axis].plot(steps[:500], rho8[:500], color=colors[3])
axs[axis].plot(steps[:], phirho8[:], color=colors[4])

# lb_phi1, ub_phi1 = conf_interval(phi1[:500], colors[2])
lb_phi8, ub_phi8 = conf_interval(phi8[:500], colors[1])
lb_phi10, ub_phi10 = conf_interval(phi10[:1600], colors[0])
# lb_rho8, ub_rho8 = conf_interval(rho8[:500], colors[3])
lb_phirho8, ub_phirho8 = conf_interval(phirho8[:1000], colors[4])

# axs[axis].fill_between(steps[:500], lb_phi1, ub_phi1, color=colors[2], alpha=0.3)
# axs[axis].fill_between(steps[:500], lb_phi8, ub_phi8, color=colors[1], alpha=0.3)
axs[axis].fill_between(steps[:1600], lb_phi10, ub_phi10, color=colors[0], alpha=0.3)
# axs[axis].fill_between(steps[:500], lb_rho8, ub_rho8, color=colors[3], alpha=0.3)
axs[axis].fill_between(steps[:1000], lb_phirho8, ub_phirho8, color=colors[4], alpha=0.3)

# axs[1].fill_between(loot_steps[:200], lb_loot_ppo, ub_loot_ppo, color=colors[0], alpha=0.3)

axs[axis].set_title('Threefish')

formatter = ticker.FuncFormatter(thousands)
plt.gca().xaxis.set_major_formatter(formatter)
# threefish_phi1 = smooth_tensor(window_size, threefish_phi1)
# threefish_phi8 = smooth_tensor(window_size, threefish_phi8)
# threefish_ppo = smooth_tensor(window_size, threefish_ppo)
# axs[2].axhline(0.8, color='green', linestyle='--')  # Add horizontal dashed line at y=1
# axs[2].text(threefish_steps[0], 0.7, '2.5 ↑ human', va='center', ha='left', color='black', fontsize=12)
# axs[2].text(threefish_steps[-120], 0.1, 'EXPIL', va='center', ha='left', color='black', fontsize=12)
#
# axs[2].axhline(y=-0.7, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
# axs[2].text(threefish_steps[-80], -0.7, 'random', va='center', ha='left', color='black', fontsize=12)
# axs[2].text(threefish_steps[-150], -0.6, 'PPO', va='center', ha='left', color='black', fontsize=12)
# axs[2].text(threefish_steps[-350], -0.6, 'NUDGE', va='center', ha='left', color='black', fontsize=12)
#
# lines = []
# p, = axs[2].plot(threefish_steps[:400], threefish_phi1[:400], color=colors[2])
# lines.append(p)
#
#
#
# p, = axs[2].plot(threefish_steps[:400], threefish_phi8[:400], color=colors[1])
# lines.append(p)
# p, = axs[2].plot(threefish_steps[:400], threefish_ppo[:400], color=colors[0])
# lines.append(p)
#
# lb_tf_phi1, ub_tf_phi1 = conf_interval(threefish_phi1[:400], colors[2])
# lb_tf_phi8, ub_tf_phi8 = conf_interval(threefish_phi8[:400], colors[1])
# lb_tf_ppo, ub_tf_ppo = conf_interval(threefish_ppo[:400], colors[0])
# axs[2].fill_between(threefish_steps[:400], lb_tf_phi1, ub_tf_phi1, color=colors[2], alpha=0.3)
# axs[2].fill_between(threefish_steps[:400], lb_tf_phi8, ub_tf_phi8, color=colors[1], alpha=0.3)
# axs[2].fill_between(threefish_steps[:400], lb_tf_ppo, ub_tf_ppo, color=colors[0], alpha=0.3)
#
#
#
# axs[2].set_title('Threefish')
#
# formatter = ticker.FuncFormatter(thousands)
# plt.gca().xaxis.set_major_formatter(formatter)

# Format the x-axis labels to show units in thousands
for ax in axs:
    ax.set_xlabel('Steps (in thousands)')
    ax.set_ylabel('Rewards')
    ax.xaxis.set_major_formatter(formatter)
    ax.grid(True)

# labels = [h.get_label() for h in lines]
# lines = []
# labels = ["NUDGE", "EXPIL"]
# plt.subplots_adjust(left=0.05, right=0.98, bottom=0.25)
# leg.set_in_layout(False)
# fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0), ncol=3)
# axs[2].legend(lines, ["Suff", "Ness", "Sum"], loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)

plt.tight_layout()
plt.savefig(config.path_nesy / "wandb" / f"model_reward_compare.pdf")
plt.cla()
