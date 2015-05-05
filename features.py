import pandas as pd
import numpy as np
import cPickle as pickle
from copy import copy
import itertools

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

'''
grouped = bids.groupby(['auction'])
bb = pd.concat([(group['time']-group['time'].min())/(group['time'].max()-group['time'].min()) for _, group in grouped], axis=0)
'''

bids = pd.read_csv('bids.csv', index_col=0)
bb = pd.read_csv('bidding_timing.csv', index_col=0)
bids = bids.join(bb)

gb = bids.groupby('bidder_id')
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
bidders_train = [line.strip() for line in open('bidders_train.csv')]
bidders_test = [line.strip() for line in open('bidders_test.csv')]
train_label = train.loc[bidders_train,'outcome']

def find_uniquness(g):
  return g.nunique()/float(g.size)

def avg_value_counts(g):
  return g.value_counts().mean()

def max_value_counts(g):
  return g.value_counts().max()

def bidding_timing(g):
  first_bid = g.min()
  last_bid = g.max()
  duration = last_bid - first_bid
  return pd.Series({'to_first_bid':g-first_bid, 'to_last_bid':last_bid-g, 'bid_timing': (g-first_bid+1)/float(duration+1)})
  #return (g-g.min())*1.0/(g.max()-g.min())

gb_size = gb.size()
cols = ['auction', 'merchandise', 'device', 'country', 'ip', 'url']
f_list = [pd.Series.nunique, find_uniquness]
apply_dict = dict((col, copy(f_list)) for col in cols)
apply_dict['auction'].append(max_value_counts)
apply_dict['auction'].append(avg_value_counts)
aa = gb.agg(apply_dict)
features = pd.concat([aa, gb_size], axis=1)
train_features = features.loc[bidders_train,:]
test_features = features.loc[bidders_test,:]

X = train_features.as_matrix()
y = train_label.as_matrix()

clf = GradientBoostingClassifier(n_estimators=20)
clf.fit(X, y)
X_test = test_features.as_matrix()
y_pred_proba = clf.predict_proba(X_test)
fout = open('results_test.csv', 'w')
for bidder, pred in itertools.izip(bidders_test, y_pred_proba[:,1]):
  fout.write(bidder+','+str(pred)+'\n')
fout.close()


