# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

k_values = np.array([0, 1, 5, 10, 20, 30, 50, 100, 150])
sst2 = [0.9576, 0.9167, 0.8967, 0.8782, 0.8586, 0.7600, 0.7799, 0.7792, 0.6162][::-1]

k_positions = {
    0: 0.00,
    1: 0.05,
    3: 0.10,
    5: 0.14,
    6: 0.18,
    10: 0.28,
    20: 0.33,
    30: 0.41,
    # 40: 0.35,
    50: 0.515,
    100: 0.75,
    150: 1.00
}

k_ranens_values = np.array([0,1,3,5,6,10])
sst2_ranens = [0.6162, 0.9109, 0.9380, 0.9790, 0.9566, 0.9583]
x_ranens_scaled = [k_positions[k] for k in k_ranens_values]

x_scaled = [k_positions[k] for k in k_values]
fig, ax = plt.subplots(figsize=(6, 5))

# Plot the lines (only SST2 and CR per user code)
# ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='forestgreen', label=r'$\mathrm{Top}$' + "-" + r'$\mathrm{k}$', ls='--', lw=5, markersize=15)
# set marker size
ax.plot(x_ranens_scaled, sst2_ranens, marker='o', color='royalblue', label=r'$\mathrm{Ours}$', markersize=15, lw=5)


# Set x-ticks and x-tick labels
selected_ticks = [0 , 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
rm_selected_ticks = [r'$\mathrm{0}$' , r'$\mathrm{10}$', r'$\mathrm{50}$', r'$\mathrm{100}$', r'$\mathrm{150}$']
ax.set_xticklabels(rm_selected_ticks, fontsize=18)

# Axis labels and limits
ax.set_xlabel(r'$k$', fontsize=40)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=40)
ax.set_ylim(0.6, 1.05)
ax.axhline(y=0.9583, linestyle='--', color='blue', linewidth=3)
# Tick font size
ax.tick_params(axis='y', labelsize=40)
ax.tick_params(axis='x', labelsize=40)

# Legend and grid
ax.legend(fontsize=24)
# plt.title(r'$\mathrm{SST}$' + "-" + r'$\mathrm{2}$', fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
plt.savefig("long_context_7B_sst2.pdf")



# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['text.usetex'] = True  # not really needed

k_values = np.array([0, 1, 5, 10, 20, 30, 50, 100, 150])
sst2 = [0.4770, 0.6055, 0.5034, 0.4706, 0.4376, 0.4483, 0.3604, 0.3484, 0.3418][::-1]
# , 0.7866131527
# , 0.6779388084
# , 0.9177419355
# Custom transformation: manually assign positions to ensure 1~50 occupy ~1/3 of space
# First third (0 to 0.33): k in 1~50
# Remaining two-thirds (0.33 to 1): k in 100 and 150
k_positions = {
    0: 0.00,
    1: 0.05,
    3: 0.10,
    5: 0.14,
    6: 0.18,
    7: 0.215,
    8: 0.25,
    10: 0.28,
    20: 0.33,
    30: 0.41,
    # 40: 0.35,
    50: 0.515,
    100: 0.75,
    150: 1.00
}
k_ranens_values = np.array([0,1,3,6, 7,8])
sst2_ranens = [0.3418, 0.4100, 0.6723, 0.7281, 0.7776, 0.7780]
x_ranens_scaled = [k_positions[k] for k in k_ranens_values]

x_scaled = [k_positions[k] for k in k_values]
fig, ax = plt.subplots(figsize=(6, 5))

# Plot the lines (only SST2 and CR per user code)
# ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='forestgreen', label=r'$\mathrm{Top-k}$', ls='--', lw=5, markersize=12)
ax.plot(x_ranens_scaled, sst2_ranens, marker='o', color='royalblue', label=r'$\mathrm{Ours}$', markersize=15, lw=5)

# Set x-ticks and x-tick labels
selected_ticks = [0 , 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
rm_selected_ticks = [r'$\mathrm{0}$' , r'$\mathrm{10}$', r'$\mathrm{50}$', r'$\mathrm{100}$', r'$\mathrm{150}$']
ax.set_xticklabels(rm_selected_ticks, fontsize=40)

# Axis labels and limits
ax.set_xlabel(r'$k$', fontsize=40)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=40)
plt.yticks(np.arange(0.2, 0.9, 0.2))
plt.ylim(0.3, 0.85)

ax.axhline(y=0.7780, linestyle='--', color='blue', linewidth=3)
# Tick font size
ax.tick_params(axis='y', labelsize=40)
ax.tick_params(axis='x', labelsize=40)

# Legend and grid
# ax.legend(fontsize=24)
# plt.title(r'$\mathrm{Coin}$~$\mathrm{Flip}$', fontsize=40)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
plt.savefig("long_context_7B_coin_flip.pdf")