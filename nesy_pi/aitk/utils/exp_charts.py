# Created by shaji at 11/05/2024


from pathlib import Path
import matplotlib.pyplot as plt
import torch

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

root = Path( "E:\\projects\\storage\\check_point\\getout\\trained_models")
left_8 = root / "left_8.pt"
left_90 = root / "left_90.pt"
right_90 = root / "right_90.pt"
jump_8 = root / "jump_8.pt"
jump_90 = root / "jump_90.pt"

# Sample data for four bar charts
left_8_ness = torch.load(left_8)[0].to("cpu")
left_8_suff = torch.load(left_8)[1].to("cpu")

left_90_ness = torch.load(left_90)[0][:50].to("cpu")
left_90_suff = torch.load(left_90)[1][:50].to("cpu")

jump_8_ness = torch.load(jump_8)[0].to("cpu")
jump_8_suff = torch.load(jump_8)[1].to("cpu")


right_90_ness = torch.load(right_90).permute(1,0)[0][:50].to("cpu")
right_90_suff = torch.load(right_90).permute(1,0)[1][:50].to("cpu")

jump_90_ness = torch.load(jump_90)[0][:50].to("cpu")
jump_90_suff = torch.load(jump_90)[1][:50].to("cpu")

# Create subplots with 1 row and 4 columns
fig, axs = plt.subplots(6, 1, figsize=(10,12))

#
# # Plotting the first bar chart
# axs[0,0].bar(torch.arange(len(left_8_ness)).tolist(),left_8_ness, color='skyblue')
# axs[0,0].set_title('Left_phi_8_Ness', fontsize="24")
# axs[1,0].bar(torch.arange(len(left_8_suff)).tolist(),left_8_suff, color='skyblue')
# axs[1,0].set_title('Left_phi_8_Suff', fontsize="24")
#
#
# # Plotting the third bar chart
# axs[2,0].bar(torch.arange(len(jump_8_ness)), jump_8_ness, color='lightgreen')
# axs[2,0].set_title('Jump_phi_8_Ness', fontsize="24")
# axs[3,0].bar(torch.arange(len(jump_8_suff)), jump_8_suff, color='lightgreen')
# axs[3,0].set_title('Jump_phi_8_Suff', fontsize="24")


# Plotting the second bar chart
p0 = axs[0].bar(torch.arange(len(left_90_ness)), left_90_ness, color='skyblue')
axs[0].set_title('Left_Ness', fontsize="24")
# axs[0].set_yscale('log')  # Set y-axis scale to logarithmic
p1 = axs[1].bar(torch.arange(len(left_90_suff)), left_90_suff, color='skyblue')
axs[1].set_title('Left_Suff', fontsize="24")
# axs[1].set_yscale('log')  # Set y-axis scale to logarithmic

# Plotting the fourth bar chart
p2 = axs[2].bar(torch.arange(len(jump_90_ness)), jump_90_ness, color='lightgreen')
axs[2].set_title('Jump_Ness', fontsize="24")
# axs[2].set_yscale('log')  # Set y-axis scale to logarithmic

p3 = axs[3].bar(torch.arange(len(jump_90_suff)), jump_90_suff, color='lightgreen')
axs[3].set_title('Jump_Suff', fontsize="24")
# axs[3].set_yscale('log')  # Set y-axis scale to logarithmic

# Plotting the fourth bar chart
p4 = axs[4].bar(torch.arange(len(right_90_ness)), right_90_ness, color='gold')
axs[4].set_title('Right_Ness', fontsize="24")
# axs[4].set_yscale('log')  # Set y-axis scale to logarithmic

p5 = axs[5].bar(torch.arange(len(right_90_suff)), right_90_suff, color='gold')
axs[5].set_title('Right_Suff', fontsize="24")
# axs[5].set_yscale('log')  # Set y-axis scale to logarithmic

for row_ax in axs:
    row_ax.spines['top'].set_visible(False)
    row_ax.spines['right'].set_visible(False)



# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(root / f"histogram.pdf")
plt.cla()
