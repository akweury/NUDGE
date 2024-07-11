# Created by shaji at 11/05/2024


from pathlib import Path
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import pandas as pd

import matplotlib.pylab as pylab
import matplotlib.ticker as ticker


params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

root = Path("E:\\projects\\storage\\check_point\\getout\\trained_models")
getout_reward_file = root / "training_reward.csv"
df = pd.read_csv(getout_reward_file)
getout_steps = torch.tensor(df[df.columns[0]].array)
getout_phi8 = torch.tensor(df[df.columns[1]].array)
getout_phi1 = torch.tensor(df[df.columns[4]].array)
getout_phi90 = torch.tensor(df[df.columns[10]].array)
getout_phi180 = torch.tensor(df[df.columns[7]].array)

loot_root = Path("E:\\projects\\storage\\check_point\\loot\\trained_models")
loot_reward_file = loot_root / "training_reward.csv"
loot_df = pd.read_csv(loot_reward_file)

loot_steps = torch.tensor(loot_df[loot_df.columns[0]].array)
loot_phi8 = torch.tensor(loot_df[loot_df.columns[1]].array)
loot_phi1 = torch.tensor(loot_df[loot_df.columns[4]].array)

threefish_root = Path("E:\\projects\\storage\\check_point\\threefish\\trained_models")
threefish_reward_file = threefish_root / "training_reward.csv"
threefish_df = pd.read_csv(threefish_reward_file)
threefish_steps = torch.tensor(threefish_df[threefish_df.columns[0]].array)
threefish_phi8 = torch.tensor(threefish_df[threefish_df.columns[4]].array)
threefish_phi1 = torch.tensor(threefish_df[threefish_df.columns[1]].array)

# Create subplots with 1 row and 4 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
marks = ['o', "*", "x"]
colors = ['skyblue', 'salmon', 'lightgreen']

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


window_size = 20
getout_phi1 = smooth_tensor(window_size, getout_phi1)
getout_phi8 = smooth_tensor(window_size, getout_phi8)
getout_phi90 = smooth_tensor(window_size, getout_phi90)

axs[0].plot(getout_steps, getout_phi1, color=colors[2])
axs[0].plot(getout_steps, getout_phi90, color=colors[1])
# axs[0].plot(getout_phi8, color=colors[0])

axs[0].axhline(y=13.5, color='green', linestyle='--')  # Add horizontal dashed line at y=1
axs[0].text(getout_steps[1], 15, 'human', va='center', ha='left', color='black', fontsize=12)
axs[0].text(getout_steps[-20], 10, 'EXPIL', va='center', ha='left', color='black', fontsize=12)
axs[0].text(getout_steps[-25], -30, 'NUDGE', va='center', ha='left', color='black', fontsize=12)

axs[0].axhline(y=-22.5, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
axs[0].text(getout_steps[-20], -21, 'random', va='center', ha='left', color='black', fontsize=12)

axs[0].set_title('Getout')
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer

formatter = ticker.FuncFormatter(thousands)
plt.gca().xaxis.set_major_formatter(formatter)

axs[1].axhline(5.7, color='green', linestyle='--')  # Add horizontal dashed line at y=1
axs[1].text(loot_steps[5], 5.5, 'human', va='center', ha='left', color='black', fontsize=12)
axs[1].text(loot_steps[50], 3.5, 'EXPIL', va='center', ha='left', color='black', fontsize=12)
axs[1].axhline(y=0.6, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
axs[1].text(loot_steps[100], 0.4, 'random', va='center', ha='left', color='black', fontsize=12)
axs[1].text(loot_steps[150], 0, 'NUDGE', va='center', ha='left', color='black', fontsize=12)

loot_phi1 = smooth_tensor(window_size, loot_phi1)
loot_phi8 = smooth_tensor(window_size, loot_phi8)
axs[1].plot(loot_steps[:200], loot_phi1[:200], color=colors[2])
axs[1].plot(loot_steps[:200], loot_phi8[:200], color=colors[1])
axs[1].set_title('Loot')

formatter = ticker.FuncFormatter(thousands)
plt.gca().xaxis.set_major_formatter(formatter)

threefish_phi1 = smooth_tensor(window_size, threefish_phi1)
threefish_phi8 = smooth_tensor(window_size, threefish_phi8)
axs[2].axhline(2.5, color='green', linestyle='--')  # Add horizontal dashed line at y=1
axs[2].text(threefish_steps[-20], 2.4, 'human', va='center', ha='left', color='black', fontsize=12)
axs[2].text(threefish_steps[-20], 0, 'EXPIL', va='center', ha='left', color='black', fontsize=12)

axs[2].axhline(y=-0.7, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
axs[2].text(threefish_steps[-20], -0.8, 'random', va='center', ha='left', color='black', fontsize=12)
axs[2].text(threefish_steps[-25], -0.5, 'NUDGE', va='center', ha='left', color='black', fontsize=12)

lines = []
p, = axs[2].plot(threefish_steps, threefish_phi1, color=colors[2])
lines.append(p)
p, = axs[2].plot(threefish_steps, threefish_phi8, color=colors[1])
lines.append(p)
axs[2].set_title('Threefish')

formatter = ticker.FuncFormatter(thousands)
plt.gca().xaxis.set_major_formatter(formatter)

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
plt.savefig(root / f"training_reward.pdf")
plt.cla()
