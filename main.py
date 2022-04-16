"""
    Interpretability Comparison for Text Data: Anchors vs LIME on simple models for sentiment analysis.
"""

# This script only generates figures and logs for experiments.

import os
import pickle
import logging
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

# specific parameters
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.set_theme(style='darkgrid')
lw = 3  # linewidth
ds = 15  # dot size
fs = 25  # fontsize
sns.set(font_scale=2)


# DECISION TREES
res = os.path.join(os.getcwd(), 'results', 'decision_trees')

# indicator
print('Saving indicator...')
info_indicator = pickle.load(
    open(os.path.join('results', 'decision_trees', str('indicator' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
fig.tight_layout()
# LIME
sorted_index = info_indicator['LIME'].median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=info_indicator['LIME'][sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')
# Anchors
info_indicator['Anchors'].sort()
sns.countplot(info_indicator['Anchors'], ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
filename = os.path.join(res, 'indicator')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'indicator.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_indicator.keys():
    logging.info(str(key) + ': ' + str(info_indicator[key]))

# dtree
print('Saving dtree...')
info_dtree = pickle.load(open(os.path.join('results', 'decision_trees', str('dtree' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
fig.tight_layout()
# LIME
sorted_index = info_dtree['LIME'].median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=info_dtree['LIME'][sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')
# Anchors
info_dtree['Anchors'].sort()
sns.countplot(info_dtree['Anchors'], ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
filename = os.path.join(res, 'dtree')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'dtree.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_dtree.keys():
    logging.info(str(key) + ': ' + str(info_dtree[key]))

# product_limit
print('Saving product_limit...')
info_product_limit = pickle.load(
    open(os.path.join('results', 'decision_trees', str('product_limit' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [2.5, 1.5]})
fig.tight_layout()
fig.suptitle(info_product_limit['Example'], fontsize=fs)
# LIME
sorted_index = info_product_limit['LIME'].median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=info_product_limit['LIME'][sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')
# Anchors
info_product_limit['Anchors'].sort()
sns.countplot(info_product_limit['Anchors'], ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
filename = os.path.join(res, 'product_limit')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'product_limit.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_product_limit.keys():
    logging.info(str(key) + ': ' + str(info_product_limit[key]))

# product_breakpoint
print('Saving product_breakpoint...')
info_product_breakpoint = pickle.load(
    open(os.path.join('results', 'decision_trees', str('product_breakpoint' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [2.5, 1.5]})
fig.tight_layout()
fig.suptitle(info_product_breakpoint['Example'], fontsize=fs)
# LIME
sorted_index = info_product_breakpoint['LIME'].median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=info_product_breakpoint['LIME'][sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')
# Anchors
info_product_breakpoint['Anchors'].sort()
sns.countplot(info_product_breakpoint['Anchors'], ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
filename = os.path.join(res, 'product_breakpoint')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'product_breakpoint.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_product_breakpoint.keys():
    logging.info(str(key) + ': ' + str(info_product_breakpoint[key]))

# subsets
for i in [1, 2, 4, 5]:
    print('Saving subsets_m' + str(i) + '...')
    info_subsets = pickle.load(
        open(os.path.join('results', 'decision_trees', str('subsets_m' + str(i) + '.p')), 'rb'))
    # Figure
    fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [2.5, 1.5]})
    fig.tight_layout()
    fig.suptitle('m_{very}=' + str(info_subsets['Multiplicities']['very']), fontsize=fs)
    # LIME
    sorted_index = info_subsets['LIME'].median().abs().sort_values(ascending=False).index[:10]
    sns.boxplot(data=info_subsets['LIME'][sorted_index], orient='h', ax=axes[0])
    axes[0].set_title('LIME')
    # Anchors
    info_subsets['Anchors'].sort()
    sns.countplot(info_subsets['Anchors'], ax=axes[1])
    axes[1].set_title('Anchors')
    axes[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    filename = os.path.join(res,  'subsets_m' + str(i))
    plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
    plt.clf()
    # log
    filename = os.path.join(res, 'subsets_m' + str(i) + '.log')
    with open(filename, 'w'):
        pass
    logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
    logging.info('Comparing Anchors and LIME on text data. \n')
    for key in info_subsets.keys():
        logging.info(str(key) + ': ' + str(info_subsets[key]))


# LOGISTIC
res = os.path.join(os.getcwd(), 'results', 'logistic')

# sparse
print('Saving sparse...')
info_sparse = pickle.load(
    open(os.path.join('results', 'logistic', str('sparse' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
fig.tight_layout()
# LIME
sorted_index = info_sparse['LIME'].median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=info_sparse['LIME'][sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')
# Anchors
info_sparse['Anchors'].sort()
sns.countplot(info_sparse['Anchors'], ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
filename = os.path.join(res, 'sparse')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'sparse.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_sparse.keys():
    logging.info(str(key) + ': ' + str(info_sparse[key]))


# arbitrary
print('Saving arbitrary...')
info_arbitrary = pickle.load(
    open(os.path.join('results', 'logistic', str('arbitrary' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [5, 1]})
fig.tight_layout()
# LIME
sorted_index = info_arbitrary['LIME'].median().abs().sort_values(ascending=False).index[:10]
sns.boxplot(data=info_arbitrary['LIME'][sorted_index], orient='h', ax=axes[0])
axes[0].set_title('LIME')
# Anchors
info_arbitrary['Anchors'].sort()
sns.countplot(info_arbitrary['Anchors'], ax=axes[1])
axes[1].set_title('Anchors')
axes[1].tick_params(axis='x', rotation=30)
plt.tight_layout()
filename = os.path.join(res, 'arbitrary')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'arbitrary.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_arbitrary.keys():
    logging.info(str(key) + ': ' + str(info_arbitrary[key]))


# intercept
w_1 = 'haircut'
w_2 = 'daughter'
print('Saving intercept...')
info_intercept = pickle.load(
    open(os.path.join('results', 'logistic', str('intercept' + '.p')), 'rb'))
# Figure
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
# LIME
info_intercept['LIME'].Intercept = - info_intercept['LIME'].Intercept
sns.lineplot(data=info_intercept['LIME'], x='Intercept', y=w_1, label=w_1, linewidth=lw, ci='sd')
sns.lineplot(data=info_intercept['LIME'], x='Intercept', y=w_2, label=w_2, linewidth=lw, ci='sd')
plt.legend()
axes[0].set_xlabel('- Intercept')
axes[0].set_ylabel('Occurrences')
axes[0].set_title('LIME')
# Anchors
runs, anchors, a_1, a_2, intercept = [], [], [], [], []
for i, row in info_intercept['Anchors'].iterrows():
    a_1.append(row.Anchor.count(w_1))
    a_2.append(row.Anchor.count(w_2))
    runs.append(row.Run)
    intercept.append(row.Intercept)
anchors_out = pd.DataFrame(columns=['Run', 'Intercept', 'Anchor', w_1, w_2])
anchors_out.Anchor = anchors
anchors_out.Run = runs
anchors_out[w_1] = a_1
anchors_out[w_2] = a_2
anchors_out.Intercept = intercept
sns.lineplot(data=anchors_out, x='Intercept', y=a_1, label=w_1, drawstyle='steps-post', linewidth=lw, marker='o', markersize=ds)
sns.lineplot(data=anchors_out, x='Intercept', y=a_2, label=w_2, drawstyle='steps-post', linewidth=lw, marker='o', markersize=ds)
plt.legend()
axes[1].set_xlabel('- Intercept')
axes[1].set_ylabel('Occurrences')
axes[1].set_title('Anchors')
filename = os.path.join(res, 'intercept')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
plt.clf()
# log
filename = os.path.join(res, 'intercept.log')
with open(filename, 'w'):
    pass
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n')
for key in info_intercept.keys():
    logging.info(str(key) + ': ' + str(info_intercept[key]))

