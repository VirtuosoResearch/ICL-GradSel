# %%
# Improve the non-linear x-axis scaling to make k=1~50 occupy only ~1/3 of the horizontal space

# Original k values and data
import numpy as np
import matplotlib.pyplot as plt
k_values = np.array([1, 5, 10, 20, 30, 50, 100, 150])
coin_flip = [0.4481674526, 0.3398425812, 0.3871933546, 0.5192239859, 0.6873563218, 0.7518958855, 0.7973722184, 0.7911603205]
sst2 =      [0.5974235105, 0.8782467532, 0.8193496588, 0.7996794872, 0.7398959584, 0.6124031008, 0.6347402597, 0.5416666667]
cr =        [0.5671296296, 0.9141414141, 0.8980407837, 0.8980407837, 0.8962148962, 0.8380952381, 0.7638888889, 0.7638888889]
# , 0.7866131527
# , 0.6779388084
# , 0.9177419355
# Custom transformation: manually assign positions to ensure 1~50 occupy ~1/3 of space
# First third (0 to 0.33): k in 1~50
# Remaining two-thirds (0.33 to 1): k in 100 and 150
k_positions = {
    1: 0.00,
    5: 0.09,
    10: 0.25,
    20: 0.33,
    30: 0.41,
    # 40: 0.35,
    50: 0.50,
    100: 0.75,
    150: 1.00
}
x_scaled = [k_positions[k] for k in k_values]
fig, ax = plt.subplots(figsize=(4, 3.3))

# Plot the lines (only SST2 and CR per user code)
ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='blue', label='SST2')
ax.plot(x_scaled, cr, marker='D', color='orange', label='CR')

# Set x-ticks and x-tick labels
selected_ticks = [1, 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
ax.set_xticklabels(selected_ticks, fontsize=18)

# Axis labels and limits
ax.set_xlabel(r'$\mathrm{\#~Samples}$', fontsize=20)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=20)
ax.set_ylim(0.3, 1.1)

# Tick font size
ax.tick_params(axis='y', labelsize=15)

# Legend and grid
# ax.legend(fontsize=10)
plt.title("Model size: 8B", fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
plt.savefig("long_context_8b.pdf")

# %%
import numpy as np
import matplotlib.pyplot as plt
k_values = np.array([1, 5, 10, 20, 30, 50, 100, 150])
coin_flip =     [0.3333, 0.3333, 0.3428, 0.3750, 0.5386, 0.6398, 0.3614, 0.3884][::-1]
sst2 =          [0.6702, 0.7619, 0.7429, 0.6783, 0.7772, 0.7878, 0.8264, 0.7029][::-1]
cr =            [0.3929, 0.4502, 0.3929, 0.4502, 0.5943, 0.6351, 0.7915, 0.6003][::-1]
# , 0.3627
# , 0.6528
# , 0.5024
# Custom transformation: manually assign positions to ensure 1~50 occupy ~1/3 of space
# First third (0 to 0.33): k in 1~50
# Remaining two-thirds (0.33 to 1): k in 100 and 150
k_positions = {
    1: 0.00,
    5: 0.07,
    10: 0.14,
    20: 0.21,
    30: 0.28,
    # 40: 0.35,
    50: 0.42,
    100: 0.71,
    150: 1.00
}
x_scaled = [k_positions[k] for k in k_values]
fig, ax = plt.subplots(figsize=(4, 3.3))

# Plot the lines (only SST2 and CR per user code)
ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='blue', label='SST2')
ax.plot(x_scaled, cr, marker='D', color='orange', label='CR')

# Set x-ticks and x-tick labels
selected_ticks = [1, 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
ax.set_xticklabels(selected_ticks, fontsize=18)

# Axis labels and limits
ax.set_xlabel(r'$\mathrm{\#~Samples}$', fontsize=20)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=20)
ax.set_ylim(0.3, 1.1)

# Tick font size
ax.tick_params(axis='y', labelsize=15)

# Legend and grid
ax.legend(fontsize=13,loc='upper right')
plt.title("Model size: 3B", fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.subplots_adjust(right=0.8) 
# plt.show()
plt.savefig("long_context_3b.pdf")


# %%
# Redraw the plot without SST2 and CR, move legend to the top of the chart
import numpy as np
import matplotlib.pyplot as plt

k_values = np.array([1, 5, 10, 20, 30, 40, 50, 100, 150])
k_positions = {
    1: 0.00,
    5: 0.05,
    10: 0.10,
    20: 0.18,
    30: 0.26,
    40: 0.30,
    50: 0.33,
    100: 0.67,
    150: 1.00
}
x_scaled = [k_positions[k] for k in k_values]

llama_8b = [0.5417, 0.6347, 0.6124, 0.6779, 0.7399, 0.7997, 0.8193, 0.8782, 0.5974][::-1]
deepseek_7b = [0.9576, 0.9167, 0.8967, 0.8980, 0.8782, 0.8586, 0.7600, 0.7799, 0.7792][::-1]
llama_3b = [0.6702, 0.7619, 0.7429, 0.6528, 0.6783, 0.7772, 0.7878, 0.8264, 0.7029][::-1]
opt_1_3b = [ 0, 0, 0.4358, 0.3761, 0.3442, 0.3107, 0.3442, 0.3442, 0.5308][::-1]
llama_1b = [0.5246, 0.4907, 0.4637, 0.4907, 0.5166, 0.4907, 0.4637, 0.6347, 0.5758][::-1]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(x_scaled, llama_8b, marker='o', label='LLaMA-8B')
ax.plot(x_scaled, deepseek_7b, marker='o', label='Deepseek-7B')
ax.plot(x_scaled, llama_3b, marker='o', label='LLaMA-3B')
ax.plot(x_scaled, opt_1_3b, marker='o', label='OPT-1.3B')
ax.plot(x_scaled, llama_1b, marker='o', label='LLaMA-1B')

selected_ticks = [1, 10, 20, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
ax.set_xticklabels(selected_ticks, fontsize=11)

ax.set_xlabel('k', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_ylim(0.3, 1.0)
ax.tick_params(axis='y', labelsize=11)

ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# %%
# Improve the non-linear x-axis scaling to make k=1~50 occupy only ~1/3 of the horizontal space

# Original k values and data
import numpy as np
import matplotlib.pyplot as plt
k_values = np.array([0, 1, 5, 10, 20, 30, 50, 100, 150])
sst2 = [0.9576, 0.9167, 0.8967, 0.8782, 0.8586, 0.7600, 0.7799, 0.7792, 0.6162][::-1]
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
    10: 0.28,
    20: 0.33,
    30: 0.41,
    # 40: 0.35,
    50: 0.50,
    100: 0.75,
    150: 1.00
}

k_ranens_values = np.array([0,1,3,5,6,10])
sst2_ranens = [0.6162, 0.9109, 0.9380, 0.9790, 0.9566, 0.9583]
x_ranens_scaled = [k_positions[k] for k in k_ranens_values]

x_scaled = [k_positions[k] for k in k_values]
fig, ax = plt.subplots(figsize=(4, 3.3))

# Plot the lines (only SST2 and CR per user code)
# ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='green', label=r'Top-$K$')
ax.plot(x_ranens_scaled, sst2_ranens, marker='D', color='blue', label='Ours')

# Set x-ticks and x-tick labels
selected_ticks = [0 , 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
ax.set_xticklabels(selected_ticks, fontsize=18)

# Axis labels and limits
ax.set_xlabel(r'$\mathrm{\#~Samples}$', fontsize=20)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=20)
ax.set_ylim(0.6, 1.0)
ax.axhline(y=0.9583, linestyle='--', color='blue', linewidth=1)
# Tick font size
ax.tick_params(axis='y', labelsize=15)

# Legend and grid
ax.legend(fontsize=10)
plt.title("GLUE-SST2", fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
plt.savefig("long_context_7B_sst2.pdf")


# %%
# Improve the non-linear x-axis scaling to make k=1~50 occupy only ~1/3 of the horizontal space

# Original k values and data
import numpy as np
import matplotlib.pyplot as plt
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
fig, ax = plt.subplots(figsize=(4, 3.3))

# Plot the lines (only SST2 and CR per user code)
# ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='green', label=r'Top-$K$')
ax.plot(x_ranens_scaled, sst2_ranens, marker='D', color='blue', label='Ours')

# Set x-ticks and x-tick labels
selected_ticks = [0 , 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
ax.set_xticklabels(selected_ticks, fontsize=18)

# Axis labels and limits
ax.set_xlabel(r'$\mathrm{\#~Samples}$', fontsize=20)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=20)
ax.set_ylim(0.6, 1.0)
ax.axhline(y=0.9583, linestyle='--', color='blue', linewidth=1)
# Tick font size
ax.tick_params(axis='y', labelsize=15)

# Legend and grid
ax.legend(fontsize=16)
plt.title("GLUE-SST2", fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
plt.savefig("long_context_7B_sst2.pdf")


# %%

# Original k values and data
# Re-import necessary libraries after kernel reset
import matplotlib.pyplot as plt

# Data
k_vals = [1,3,5,7,9]
llama1b = [0.2754, 0.2754, 0.5308, 0.7374, 0.7600]
llama3b = [0.4876, 0.6961, 0.6533, 0.6881, 0.7814]
llama8b = [0.2754, 0.5974, 0.9369, 0.9109, 0.9785]
dpsk7b  = [0.9108, 0.9379, 0.9789, 0.8980, 0.9369]
opt1b   = [0.3827, 0.3827, 0.5192, 0.5338, 0.5600]
# Plot
fig, ax = plt.subplots(figsize=(7, 3.3))
ax.plot(k_vals, llama1b, marker='D', color='purple', label='LLaMA-1B')
ax.plot(k_vals, opt1b, marker='D', color='olive', label='OPT-1.3B')
ax.plot(k_vals, llama3b, marker='D', color='blue', label='LLaMA-3B')
ax.plot(k_vals, dpsk7b, marker='D', color='violet', label='DeepSeek7B')
ax.plot(k_vals, llama8b, marker='D', color='green', label='LLaMA-8B')

# Axis settings
ax.set_xlabel(r'$\mathrm{\#~Samples}$', fontsize=18)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=18)
xticks = [1,3,5,7,9]
ax.set_xticks(xticks)
ax.tick_params(axis='both', labelsize=17)
ax.set_title("Model probing", fontsize=18)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_ylim(0.2, 1.05)
ax.legend(fontsize=10)

plt.tight_layout()
# plt.show()

plt.savefig("model_probing.pdf")

# %%
# Improve the non-linear x-axis scaling to make k=1~50 occupy only ~1/3 of the horizontal space

# Original k values and data
import numpy as np
import matplotlib.pyplot as plt
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
    8: 0.25,
    10: 0.28,
    20: 0.33,
    30: 0.41,
    # 40: 0.35,
    50: 0.515,
    100: 0.75,
    150: 1.00
}
k_ranens_values = np.array([0,1,3,5,8])
sst2_ranens = [0.3418, 0.4100, 0.3672, 0.6133, 0.7780]
x_ranens_scaled = [k_positions[k] for k in k_ranens_values]

x_scaled = [k_positions[k] for k in k_values]
fig, ax = plt.subplots(figsize=(4, 3.3))

# Plot the lines (only SST2 and CR per user code)
# ax.plot(x_scaled, coin_flip, marker='D', color='green', label='Coin-Flip')
ax.plot(x_scaled, sst2, marker='D', color='green', label=r'Top-$K$')
ax.plot(x_ranens_scaled, sst2_ranens, marker='D', color='blue', label='Ours')

# Set x-ticks and x-tick labels
selected_ticks = [0 , 10, 50, 100, 150]
selected_x_scaled = [k_positions[k] for k in selected_ticks]
ax.set_xticks(selected_x_scaled)
ax.set_xticklabels(selected_ticks, fontsize=18)

# Axis labels and limits
ax.set_xlabel(r'$\mathrm{\#~Samples}$', fontsize=20)
ax.set_ylabel(r'$\mathrm{F_1~Score}$', fontsize=20)
ax.set_ylim(0.17, 0.9)
ax.axhline(y=0.7780, linestyle='--', color='blue', linewidth=1)
# Tick font size
ax.tick_params(axis='y', labelsize=15)

# Legend and grid
ax.legend(fontsize=16)
plt.title("Coin-Flip", fontsize=18)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.show()
plt.savefig("long_context_7B_coin_flip.pdf")
