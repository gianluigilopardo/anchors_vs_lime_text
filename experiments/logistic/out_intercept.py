import pickle
import os

import pandas as pd


from matplotlib import pyplot as plt
import seaborn as sns

# specific parameters
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.set_theme(style='darkgrid')
lw = 3  # linewidth
ds = 100  # dot size
fs = 25  # fontsize
sns.set(font_scale=2)  # / 0.5 * 0.35)
pd.set_option("display.max_rows", None, "display.max_columns", None)

MODEL = 'intercept_lambda'

info = pickle.load(open(os.path.join(os.getcwd().replace('experiments', 'results'), str(MODEL)) + '.p', 'rb'))

for k in info.keys():
    print('{}: {}'.format(k, info[k]))

# Figure
plt.tight_layout()
sns.scatterplot(data=info['Anchors'], x='Intercept', y='Anchor')
plt.legend()
plt.xlabel('- Intercept')
plt.ylabel('Occurrences')

filename = os.path.join(str(MODEL))
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)

# log
