# Created by shaji at 11/05/2024


from pathlib import Path
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator

import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

root = Path("E:\\projects\\storage\\check_point\\getout\\trained_models")
suff_0 = root / "suff_2-0.pt"
suff_1 = root / "suff_2-1.pt"
suff_2 = root / "suff_2-2.pt"
suff_3 = root / "suff_2-3.pt"
suff_4 = root / "suff_2-4.pt"
suff_5 = root / "suff_2-5.pt"

# Sample data for four bar charts
data_0 = torch.load(suff_0).to("cpu")
data_1 = torch.load(suff_1).to("cpu")
data_2 = torch.load(suff_2).to("cpu")
data_3 = torch.load(suff_3).to("cpu")
data_4 = torch.load(suff_4).to("cpu")
data_5 = torch.load(suff_5).to("cpu")

th_dash_line = 0.92

# Create subplots with 1 row and 4 columns
fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
marks = ['o', "*", "x"]
colors = ['skyblue', 'salmon', 'lightgreen']
for i in range(3):
    axs[0].plot(data_0[i], marker=marks[i], color=colors[i])
    axs[0].axhline(y=th_dash_line, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
    axs[0].set_title('SuffPred0')
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer

for i in range(3):
    axs[1].plot(data_1[i], marker=marks[i], color=colors[i])
    axs[1].axhline(y=th_dash_line, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
    axs[1].set_title('SuffPred1')
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer

for i in range(3):
    axs[2].plot(data_2[i], marker=marks[i], color=colors[i])
    axs[2].axhline(y=th_dash_line, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer
    axs[2].set_title('SuffPred2')

# for i in range(3):
#     axs[1,1].plot(data_3[i], marker=marks[i], color=colors[i])
#     axs[1,1].axhline(y=1, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
#     axs[1,1].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer
#     axs[1,1].set_title('SuffPred3')
# for i in range(3):
#     axs[4].plot(data_4[i], marker=marks[i], color=colors[i])
#     axs[4].axhline(y=1, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
#     axs[4].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer
#     axs[4].set_title('SuffPred4')
lines = []
for i in range(3):
    p, = axs[3].plot(data_5[i], marker=marks[i], color=colors[i])
    lines.append(p)
    axs[3].axhline(y=th_dash_line, color='gray', linestyle='--')  # Add horizontal dashed line at y=1
    axs[3].xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis scale to integer
    axs[3].set_title('SuffPred3')
#
# for ax in axs:
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# Adjust layout to prevent overlap

# plt.subplots_adjust(bottom=0.2)
# labels = [h.get_label() for h in lines]
labels = ["Ness", "Suff", "Sum"]
plt.subplots_adjust(left=0.03, right=0.98, bottom=0.17, top=0.95)
# leg.set_in_layout(False)
fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3)
# axs[2].legend(lines, ["Suff", "Ness", "Sum"], loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)

# plt.tight_layout()
plt.savefig(root / f"suff_line_chart.pdf")
plt.cla()
