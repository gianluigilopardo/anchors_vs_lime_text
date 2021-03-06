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
import warnings
warnings.filterwarnings("ignore")

# specific parameters
pd.set_option("display.max_rows", None, "display.max_columns", None)
sns.set_theme(style='darkgrid')
lw = 3  # linewidth
ds = 3  # dot size
fs = 25  # fontsize
sns.set(font_scale=2)

# log
filename = os.path.join(os.path.join(os.getcwd(),'results'), 'logs.log')
f = open(filename, 'w')
logging.basicConfig(format='%(message)s', filename=filename, level=logging.INFO)
logging.info('Comparing Anchors and LIME on text data. \n ')

# DECISION TREES
res = os.path.join(os.getcwd(), 'results', 'decision_trees')
logging.info('DECISION TREES \n')

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
logging.info('-Model: Indicator')
for key in info_indicator.keys():
    logging.info(str(key) + ': ' + str(info_indicator[key]))
f.close()

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
logging.info('\n-Model: dtree')
for key in info_dtree.keys():
    logging.info(str(key) + ': ' + str(info_dtree[key]))
f.close()

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
logging.info('\n-Model: product_limit')
for key in info_product_limit.keys():
    logging.info(str(key) + ': ' + str(info_product_limit[key]))
f.close()

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
logging.info('\n-Model: product_breakpoint')
for key in info_product_breakpoint.keys():
    logging.info(str(key) + ': ' + str(info_product_breakpoint[key]))
f.close()

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
    logging.info('\n-Model: subsets')
    for key in info_subsets.keys():
        logging.info(str(key) + ': ' + str(info_subsets[key]))
    f.close()

# LOGISTIC
logging.info('LOGISTIC \n')
res = os.path.join(os.getcwd(), 'results', 'logistic')

# sparse
print('Saving sparse...')
info_sparse = pickle.load(
    open(os.path.join(res, str('sparse.p')), 'rb'))
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
logging.info('\n-Model: sparse')
for key in info_sparse.keys():
    logging.info(str(key) + ': ' + str(info_sparse[key]))
f.close()

# arbitrary
print('Saving arbitrary...')
info_arbitrary = pickle.load(
    open(os.path.join(res, str('arbitrary' + '.p')), 'rb'))
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
logging.info('\n-Model: arbitrary')
for key in info_arbitrary.keys():
    logging.info(str(key) + ': ' + str(info_arbitrary[key]))
f.close()

# intercept
print('Saving intercept...')
info_intercept = pickle.load(
    open(os.path.join(res, str('intercept' + '.p')), 'rb'))
n_words = 3
words = list(info_intercept['Coefficients'].keys())[:n_words]
Anchors = info_intercept['Anchors'][1000:17484]
a = [[] for i in range(n_words)]
for i, row in Anchors.iterrows():
    for j in range(n_words):
        a[j].append(row.Anchor.count(words[j]))
# Figure
plt.tight_layout()
for j in range(n_words):
    sns.lineplot(data=Anchors, x='Shift', y=a[j], label=words[j], drawstyle='steps-post', linewidth=lw, marker='o',
                 markersize=ds)
plt.xlabel('Shift')
plt.ylabel('Occurrences')
filename = os.path.join(res, 'intercept')
plt.savefig(fname=str(filename + '.pdf'), bbox_inches='tight', pad_inches=0)
# log
logging.info('\n-Model: intercept')
for key in info_intercept.keys():
    logging.info(str(key) + ': ' + str(info_intercept[key]) + '\n')

