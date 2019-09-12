# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# # Compare Synthetic and Original Taxi Data
# This notebook gives a look at the first version of synthetic
# data generation. This version is designed to be fast and simple.
# Simple because the analyst just pushes one button and the whole
# dataset is synthesized. In other words, we try to create a replica
# of the original dataset.
#
# This notebook is only looking at descriptive analytics (which,
# by the way, isn't the primary intent of the synthetic data, but
# never-the-less is interesting).
# 
# It is fast (relatively) in that it
# minimizes the number of queries. It does this in two ways. First,
# it doesn't try for too much granularity and it doesn't query
# more than two columns at a time. The intent here is to find out how
# bad the results are as a baseline, and figure out where we need
# to improve.
#
# Note that this synthetic data only tries to synthesize individual
# rides, and not the behavior of individual drivers over time.

#%%
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'notebooks\\taxi'))
    print(os.getcwd())
except:
    pass
from IPython import get_ipython
#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
import matplotlib.pyplot as plt #visualization
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns #visualization
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
import random
#%%
origPath = "taxi_orig.csv"
orig = pd.read_csv(origPath)
orig = orig.drop(columns=['uid'])
cols = orig.columns.tolist()
cols.sort()
orig = orig[cols]
#orig['dropoff_datetime'] = pd.to_datetime(orig['dropoff_datetime'],unit='s')
#orig['pickup_datetime'] = pd.to_datetime(orig['pickup_datetime'],unit='s')
synPath = "taxi_syn.csv"
syn = pd.read_csv(synPath)
cols = syn.columns.tolist()
cols.sort()
syn = syn[cols]
syn['dropoff_datetime'] = syn['dropoff_datetime'].astype(int)
#syn['dropoff_datetime'] = pd.to_datetime(syn['dropoff_datetime'],unit='s')
syn['pickup_datetime'] = syn['pickup_datetime'].astype(int)
#syn['pickup_datetime'] = pd.to_datetime(syn['pickup_datetime'],unit='s')

#%% [markdown]
# ### Original Taxi Data, first 5 rows
#%%
orig.head()
#%% [markdown]
# ### Synthetic Taxi Data, first 5 rows
# Note that the "look-and-feel" of the synthetic matches that of
# the original data pretty well. This is because of an extra step
# we take that looks at each character position of every column
# and tries to mimic the original. For numeric columns, this allows
# us to get approximately the right number of decimal places, and
# for text columns, this allows us to for instance get a decent
# copy of formatted strings (for instance the 'med'column).
#
#%%
syn.head()
#%% [markdown]
# ## Compare data summaries
# The following compares statistical summaries of the original and synthetic data.
# While the per-column statistics of the synthetic data follow the original data
# pretty closely, the min and max and be far off because Aircloak
# hides extreme values when it anonymizes.
# ### Original data summary
#%%
print ("Rows     : " ,orig.shape[0])
orig.describe()
#%% [markdown]
# ### Synthetic data summary
#%%
print ("Rows     : " ,syn.shape[0])
syn.describe()
#%% [markdown]
# ## Compare graphs
# ### Trip Time
# The following graph compares trip time in seconds displayed in
# order of shortest to longest trip. The synthetic data tracks the
# original data very well, with the exception that the synthetic data
# misses a small number of very long trips (over roughly 50 minutes).
#%%
fig=plt.figure()
ax=plt.axes()
col = 'trip_time_in_secs'
yorig = list(orig[col])
yorig.sort()
ysyn = list(syn[col])
ysyn.sort()
ax.plot(range(len(yorig)),yorig,'r.',label='Original')
ax.plot(range(len(ysyn)),ysyn,'b.',label='Synthetic')
plt.ylabel(col)
ax.legend()
#%% [markdown]
# ### Pickup Locations
# From the two graphs below, we can see that the synthetic data
# for pickup location is "in the right ballpark", but leaves a lot
# to be desired. We know from Sebastian's synthetic data as well as
# my earlier stuff published on the CNIL blog that we can do a lot
# better than this.
#
# There are some strange artifacts in the synthetic data (the
# data points for instance that fall on the longitude -74.00 axis)
# that I need to look into.
#%%
fig=plt.figure()
ax=plt.axes()
xcol = 'pickup_longitude'
ycol = 'pickup_latitude'
yorig = list(orig[ycol])
xorig = list(orig[xcol])
plt.xlim(-74.091,-73.729)
plt.ylim(40.607,40.897)
ax.plot(xorig,yorig,'r.',label='Original',markersize=2)
plt.ylabel(ycol)
plt.xlabel(xcol)
ax.legend()
#%%
fig=plt.figure()
ax=plt.axes()
xcol = 'pickup_longitude'
ycol = 'pickup_latitude'
ysyn = list(syn[ycol])
xsyn = list(syn[xcol])
plt.xlim(-74.091,-73.729)
plt.ylim(40.607,40.897)
ax.plot(xsyn,ysyn,'b.',label='Synthetic',markersize=2)
plt.ylabel(ycol)
plt.xlabel(xcol)
ax.legend()
#%% [markdown]
# ### Trip time to La Guardia
# The following two graphs show the trip time to La Guardia
# airport for the original and synthetic data. The original
# data shows some reasonable color bands for trip distance.
#
# The synthetic data here is really unacceptable. The color
# bands are gone. We know we can do better, but I think that the
# analyst would have to specify in advance that they are interested
# in the correlations between location and trip time.

#%%
# 40.778259, -73.889729 (upper left)
# 40.764829, -73.853478 (lower right)
# Filter out rides whose dropoff location is around La Guardia
syn_lg = syn[(syn.dropoff_longitude >= -73.8897) &
    (syn.dropoff_longitude <= -73.8535) &
    (syn.dropoff_latitude >= 40.7468) &
    (syn.dropoff_latitude <= 40.7782)]
orig_lg = orig[(orig.dropoff_longitude >= -73.8897) &
    (orig.dropoff_longitude <= -73.8535) &
    (orig.dropoff_latitude >= 40.7468) &
    (orig.dropoff_latitude <= 40.7782)]
print(f"Original data has {len(orig_lg)} rides to La Guardia")
print(f"Synthetic data has {len(syn_lg)} rides to La Guardia")
# Filter out rides by trip time. The following flag determines
# whether we compute this using the trip_time_in_secs column or
# the difference between dropoff and pickup time. Either way,
# the synthetic data is poor here.
if False:
    syn_lg10 = syn_lg[syn.trip_time_in_secs < 600]
    syn_lg20 = syn_lg[(syn.trip_time_in_secs >= 600) &
        (syn.trip_time_in_secs < 1200)]
    syn_lg30 = syn_lg[(syn.trip_time_in_secs >= 1200) &
        (syn.trip_time_in_secs < 1800)]
    syn_lgmore = syn_lg[syn.trip_time_in_secs >= 1800]
    orig_lg10 = orig_lg[orig.trip_time_in_secs < 600]
    orig_lg20 = orig_lg[(orig.trip_time_in_secs >= 600) &
        (orig.trip_time_in_secs < 1200)]
    orig_lg30 = orig_lg[(orig.trip_time_in_secs >= 1200) &
        (orig.trip_time_in_secs < 1800)]
    orig_lgmore = orig_lg[orig.trip_time_in_secs >= 1800]
else:
    syn_lg10 = syn_lg[(syn.dropoff_datetime - syn.pickup_datetime) < 600]
    syn_lg20 = syn_lg[((syn.dropoff_datetime - syn.pickup_datetime) >= 600) &
        ((syn.dropoff_datetime - syn.pickup_datetime) < 1200)]
    syn_lg30 = syn_lg[((syn.dropoff_datetime - syn.pickup_datetime) >= 1200) &
        ((syn.dropoff_datetime - syn.pickup_datetime) < 1800)]
    syn_lgmore = syn_lg[(syn.dropoff_datetime - syn.pickup_datetime) >= 1800]
    orig_lg10 = orig_lg[(orig.dropoff_datetime - orig.pickup_datetime) < 600]
    orig_lg20 = orig_lg[((orig.dropoff_datetime - orig.pickup_datetime) >= 600) &
        ((orig.dropoff_datetime - orig.pickup_datetime) < 1200)]
    orig_lg30 = orig_lg[((orig.dropoff_datetime - orig.pickup_datetime) >= 1200) &
        ((orig.dropoff_datetime - orig.pickup_datetime) < 1800)]
    orig_lgmore = orig_lg[(orig.dropoff_datetime - orig.pickup_datetime) >= 1800]
print(len(syn_lg10), len(syn_lg20), len(syn_lg30), len(syn_lgmore))
print(len(orig_lg10), len(orig_lg20), len(orig_lg30), len(orig_lgmore))
# make graph for original data
fig=plt.figure()
ax=plt.axes()
xcol = 'pickup_longitude'
ycol = 'pickup_latitude'
plt.xlim(-74.091,-73.729)
plt.ylim(40.607,40.897)
yorig_lg10 = list(orig_lg10[ycol])
xorig_lg10 = list(orig_lg10[xcol])
ax.plot(xorig_lg10,yorig_lg10,'go',label='Less than 10 min',markersize=2)
yorig_lg20 = list(orig_lg20[ycol])
xorig_lg20 = list(orig_lg20[xcol])
ax.plot(xorig_lg20,yorig_lg20,'bo',label='10 to 20 min',markersize=2)
yorig_lg30 = list(orig_lg30[ycol])
xorig_lg30 = list(orig_lg30[xcol])
ax.plot(xorig_lg30,yorig_lg30,'yo',label='20 to 30 min',markersize=2)
yorig_lgmore = list(orig_lgmore[ycol])
xorig_lgmore = list(orig_lgmore[xcol])
ax.plot(xorig_lgmore,yorig_lgmore,'ro',label='More than 30 min',markersize=2)
plt.ylabel(ycol)
plt.xlabel(xcol + ': Original Data')
ax.legend()
# make graph for synthetic data
fig=plt.figure()
ax=plt.axes()
xcol = 'pickup_longitude'
ycol = 'pickup_latitude'
plt.xlim(-74.091,-73.729)
plt.ylim(40.607,40.897)
ysyn_lg10 = list(syn_lg10[ycol])
xsyn_lg10 = list(syn_lg10[xcol])
ax.plot(xsyn_lg10,ysyn_lg10,'go',label='Less than 10 min',markersize=2)
ysyn_lg20 = list(syn_lg20[ycol])
xsyn_lg20 = list(syn_lg20[xcol])
ax.plot(xsyn_lg20,ysyn_lg20,'bo',label='10 to 20 min',markersize=2)
ysyn_lg30 = list(syn_lg30[ycol])
xsyn_lg30 = list(syn_lg30[xcol])
ax.plot(xsyn_lg30,ysyn_lg30,'yo',label='20 to 30 min',markersize=2)
ysyn_lgmore = list(syn_lgmore[ycol])
xsyn_lgmore = list(syn_lgmore[xcol])
ax.plot(xsyn_lgmore,ysyn_lgmore,'ro',label='More than 30 min',markersize=2)
plt.ylabel(ycol)
plt.xlabel(xcol + ': Synthetic Data')
ax.legend()


#%%
