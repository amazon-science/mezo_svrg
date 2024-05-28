import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ensure that fonts are not converted to outlines in PDF
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

# Sample data for the new parameters
parameters = ['1.6B', '2.7B', '6.7B']
# memory_me_zo_adapter = [0.9, 2, 9.4]  # MeZO (Adapter)
# memory_zo_svrg_adapter = [0.94, 2.3, 11.9]  # ZO-SVRG (Adapter)
# memory_fo_sgd_adapter = [1.9, 7.4, 35.6]  # FO-SGD (Adapter)
memory_me_zo_full = [9, 14, 34]  # MeZO (Full), 1, 2.5, 
memory_zo_svrg_full = [18, 31, 74]  # ZO-SVRG (Full)1.1, 4.6,
memory_fo_sgd_full = [33, 64, 131]  # FO-SGD (Full), 1.4, 6,

# Colors for the bars
# colors_adapter = ['#add8e6', 'lightgreen', 'lightcoral']  # Lighter shades for 'Adapter'
colors_full = ['#0000ff', 'darkgreen', 'darkred']  # Standard color codes for 'Full'

# Creating the bar plot
x = np.arange(len(parameters))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
# ax.bar(x - width, memory_me_zo_adapter, width, label='MeZO (Adapter)', color=colors_adapter[0])
# ax.bar(x, memory_zo_svrg_adapter, width, label='ZO-SVRG (Adapter)', color=colors_adapter[1])
# ax.bar(x + width, memory_fo_sgd_adapter, width, label='FO-SGD (Adapter)', color=colors_adapter[2])
ax.bar(x - width, memory_me_zo_full, width, label='MeZO', color=colors_full[0])
ax.bar(x, memory_zo_svrg_full, width, label='MeZO-SVRG', color=colors_full[1])
ax.bar(x + width, memory_fo_sgd_full, width, label='FO-SGD', color=colors_full[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_yscale('log')
ax.set_xlabel('# Parameters', fontsize=12)
ax.set_ylabel('GPU Memory (GB)', fontsize=12)
ax.set_title('Memory Usage', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(parameters, fontsize=10)
ax.legend(fontsize=10)
ax.grid(True)

fig.tight_layout()

# Save the plot as a PDF file
plt.savefig('Memory_Usage.pdf')

plt.show()