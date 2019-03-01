#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:49:03 2019

@author: gabi
"""

# =============================================================================
#               Growth and Development Economics: PS3
#
#                   Consumption Insurance Tests
# =============================================================================

import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as sm
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns


pd.options.display.float_format = '{:,.2f}'.format

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS3')

# Data panel provided by Professor
panel = pd.read_stata("dataUGA.dta")

#%% Getting the data

#   Here I follow TA Albert's code to get a balanced panel with 
# HHs variables and characteristics


#Balanced panel
panelbal = panel.loc[panel["counthh"]==4,]

counthh = panelbal.groupby(by="hh")[["hh"]].count()

control = panelbal.groupby(by="hh")[["lnc","lny",]].mean()
control.reset_index(inplace=True)
control['crich'] = pd.qcut(control["lnc"], 2, labels=False)
control['c_quin'] = pd.qcut(control["lnc"], 5, labels=False)
control['yrich'] = pd.qcut(control["lny"], 2, labels=False)
control['y_quin'] = pd.qcut(control["lny"], 5, labels=False)


#Create HH characteristics controls
dummies = pd.get_dummies(control["c_quin"])
dummies.drop([0.0], axis=1, inplace=True)
dummies.columns=["c2","c3","c4","c5"]

control = control.join(dummies)

dummiesy = pd.get_dummies(control["y_quin"])
dummiesy.drop([0.0], axis=1, inplace=True)
dummiesy.columns = ["y2","y3","y4","y5"]
control = control.join(dummiesy)


control13 = panelbal.loc[panelbal.wave=="2013-2014",["hh","sex","region","urban"]]
control13["female"] = (control13.sex==2)*1
control = control.merge(control13, on="hh",how="inner")

dummiesr = pd.get_dummies(control["region"])
dummiesr.drop([1.0], axis=1, inplace=True)
dummiesr.columns = ["region2","region3","region4"]
control = control.join(dummiesr)
control.drop(["lny","lnc","sex"],axis=1, inplace=True)


#panelbal.drop(["female","urban","region","region2","region3","region4"],axis=1, inplace=True)
panelbal.sort_values(["hh","wave"])
panelbal.set_index(["hh","wave"],inplace=True)

panelbaldiff = panelbal.groupby(level=0)['d_shock','d_aggregate','d_idiosyn','d_climate','d_prices','d_health','d_job','d_pests','lnc','lny','avgc','lnctotal_gift', "lnc_nogift"].diff()
panelbaldiff.columns = ["Δshock","Δaggregate","Δidiosyn","Δclimate","Δprices","Δhealth","Δjob","Δpests",'Δc','Δy','Δavgc','Δc_gift', 'Δc_nogift']
panelbaldiff.reset_index(inplace=True)
panelbaldiff = panelbaldiff[panelbaldiff.wave != "2009-2010"]

### Balanced panel with hh controls
panelbal.reset_index(inplace=True)
panelcontrol = panelbal.merge(control, on="hh", how='left')
panelcontrol.set_index(["hh","wave"],inplace=True)

panelcontroldiff = panelbaldiff.merge(control, on="hh", how='left')
panelcontroldiff.set_index(["hh","wave"],inplace=True)

del control, control13, counthh, dummies, dummiesr, dummiesy

#%%
# =============================================================================
# Q1: Individual insurance
# =============================================================================

#   To do the insurance test per HH, I am going to use as depedend variable 
# the residual of level regression of HH's consumption on controls;
# and as regressor the residual of level regression of HH's income on controls 

#%% LEVEL REGRESSION
# age, dummies for education, regions, sex, ethnicity, and household size, rural, year

### Dummies for ethnicity
panelbal_et = panel.loc[panel["counthh"]==4,]
panelbal_et=panelbal_et.sort_values(["hh","wave"])
#panelbal_et= panelbal_et[panelbal_et.wave == "2013-2014"]
dummies_et = pd.get_dummies(panelbal_et["ethnic"])
dummies_et.reset_index(inplace=True)
dummies_et=dummies_et.drop(['index'], axis=1)
hh_id=panelbal['hh']
hh_id=pd.DataFrame(hh_id)
hh_id.columns=["hhid"]
dummies_et = pd.concat([dummies_et,hh_id],axis=1)

panelcontrol=panelcontrol.reset_index(inplace=False)
panelcontrol=panelcontrol.sort_values(["hh","wave"])
panelcontrol=pd.concat([panelcontrol,dummies_et],axis=1)

del dummies_et, panelbal_et

### Year fixed effect
panelbal_y = panel.loc[panel["counthh"]==4,]
panelbal_y=panelbal_y.sort_values(["hh","wave"])

dummies_year = pd.get_dummies(panelbal_y["year"])
dummies_year.reset_index(inplace=True)
dummies_year=dummies_year.drop(['index'], axis=1)
dummies_year = pd.concat([dummies_year,hh_id],axis=1)

panelcontrol=pd.concat([panelcontrol,dummies_year],axis=1)

del dummies_year, panelbal_y, hh_id

variables = list(panelcontrol.columns)
#[index for index, value in enumerate(variables)]
variables.remove('hh')
variables.remove('hh')
variables.remove('hh')

# dummies for year and ethnicity
#ols_dum=variables[99:150]
#ols_year=ols_dum[46:51]

panelcontrol.rename(columns={2009.0:'y09',2010.0:'y10', 2011.0:'y11', 2012.0:'y12', 2013.0:'y13', 2014.0:'y14'}, inplace=True) 
panelcontrol=panelcontrol.drop(['hhid'], axis=1)

olsc = sm.ols(formula="lnc ~ age +age_sq + region2_y  + region3_y  +region4_y  +urban_y +female_y +familysize+y10+y11+y12+y13+y14 ", data=panelcontrol).fit()
print(olsc.summary())

panelcontrol['error_c'] = olsc.resid

olsy = sm.ols(formula="lny ~ age +age_sq + region2_y  + region3_y  +region4_y  +urban_y +female_y +familysize+y10+y11+y12+y13+y14  ", data=panelcontrol).fit()
print(olsy.summary())

panelcontrol['error_y'] = olsy.resid

result_level = summary_col([olsc,  olsy],stars=True)
print(result_level)

### AVERAGE across HH and Waves
panelcontrol['agg_c']=panelcontrol.loc[:,"lnc"].mean()
panelcontrol['agg_y']=panelcontrol.loc[:,"lny"].mean()


panelcontrol=panelcontrol.sort_values(["hh","wave"])

panelcontrol.reset_index(inplace=True)
panelcontrol.set_index(["hh"],inplace=True)

ps_panel=panelcontrol

#%% TOWNSEND TEST

ps_panel_diff = ps_panel.groupby(level=0)['error_c','error_y'].diff()
ps_panel_diff.columns = ['Δc','Δy']
ps_panel_diff.reset_index(inplace=True)
ps_panel_diff.set_index(["hh"],inplace=True)

agg_ct=panelcontrol.groupby('wave')['lnc'].mean()
agg_ct=pd.DataFrame(agg_ct)
agg_ct.columns=['aggct']

#agg_ct=agg_ct.tail(3)
#agg_ct.reset_index(inplace=True)
#agg_ct=agg_ct['aggct']

ca=[]
for row in ps_panel_diff['wave']:
    if row == "2011-2012":
        ca.append(agg_ct.iloc[2,0])
    elif row == "2010-2011":
        ca.append(agg_ct.iloc[1,0])
    elif row == "2013-2014":
         ca.append(agg_ct.iloc[3,0])
        
ps_panel_diff['aggctt']=ca    

#ps_panel_diff['aggc']=ps_panel['agg_c']

#ps_panel_diff['aggy']=ps_panel['agg_y']

ps_panel_diff['wave']=ps_panel['wave']

ps_panel_diff = ps_panel_diff[ps_panel_diff.wave != "2009-2010"]
ps_panel_diff=ps_panel_diff.reset_index(inplace=False)


b1 = np.full(1490, np.nan) # coefficient of income
b2 = np.full(1490, np.nan) # coefficient of average consumption

ps_panel_diff=ps_panel_diff.sort_values(["hh","wave"])

for i in range (0,4469,3):
    h=ps_panel_diff.iloc[i:i+3]
    h=h.fillna(0)
    test=sm.ols(formula="Δc ~ Δy + aggct -1 ", data=h).fit()
    t=test.params
    t1=t[0]
    t2=t[1]
    c=int(i/3)
    b1[c] =t1
    b2[c] = t2
    del h, test, t,t2,t1, c

#%% RESULTS

hh_id=panelbal['hh']
hh_id=hh_id.drop_duplicates()
hh_id=pd.DataFrame(hh_id)
 
### Income Coefficient: coef=0 -> full insurance
b1f= pd.DataFrame(b1)
b1f.columns =['b_inc']
b1f.describe()
b1f.hist()

# It seems there are outliers, so I am going to trim income coef.

# get rid off top and bottom 1%
b1trim = b1f[ (b1f>=float(b1f.quantile(0.001))) & (b1f<=float(b1f.quantile(0.999))) ]
b1trim.describe()
b1trim.isna().sum()
b1trim.hist()

#b1trim=b1trim.dropna()

# get rid off top and bottom 5%
b1trim2 = b1f[ (b1f>=float(b1f.quantile(0.005))) & (b1f<=float(b1f.quantile(0.995))) ]
b1trim2.describe()
b1trim2.isna().sum()
b1trim2.hist()


### Aggregate Consumption Coefficient: coef=1 -> full insurance
b2f= pd.DataFrame(b2)
b2f.columns =['b_cons']
b2f.describe()
b2f.astype(bool).sum(axis=0)
b2f.hist()


#%%
# =============================================================================
# Q2-A: Relationship between insurance and income - INCOME RANKING
# =============================================================================

### Preparing income data set (for simplicity)
panel_c_1011 = ps_panel[ps_panel.wave == "2010-2011"]
panel_c_1011['b1']=b1
panel_c_1011['b2']=b2

panel_c_1112 = ps_panel[ps_panel.wave == "2011-2012"]
panel_c_1112['b1']=b1
panel_c_1112['b2']=b2

panel_c_0910 = ps_panel[ps_panel.wave == "2009-2010"]
panel_c_0910['b1']=b1
panel_c_0910['b2']=b2

panel_c_1314 = ps_panel[ps_panel.wave == "2013-2014"]
panel_c_1314['b1']=b1
panel_c_1314['b2']=b2

ps_panel_b=pd.concat([panel_c_0910,panel_c_1011])
ps_panel_b=pd.concat([ps_panel_b,panel_c_1112])
ps_panel_b=pd.concat([ps_panel_b,panel_c_1314])

del panel_c_0910,panel_c_1011,panel_c_1112,panel_c_1314

ps_panel_b=ps_panel_b.sort_values(["hh","wave"])

ps_panel_b['cbar']=ps_panel_b.groupby(by="hh")[["lnc"]].mean()
ps_panel_b['ybar']=ps_panel_b.groupby(by="hh")[["lny"]].mean()

panel_inc=ps_panel_b[["ybar","cbar","b1","b2","urban_y"]]
panel_inc=panel_inc.drop_duplicates()

#panel_inc=pd.merge(b1_out,panel_inc, how='left', on='hh')
#panel_inc=pd.merge(b1_out_2,panel_inc, how='left', on='hh')

#%% INCOME RANKING

panel_inc=panel_inc.sort_values(["ybar"])
panel_inc.describe()
#panel_inc.corr()

panel_inc.plot.scatter(x='b1',y='ybar')


# Quantiles
y_q1=panel_inc['ybar'].quantile(0.2)
y_q2=panel_inc['ybar'].quantile(0.4)
y_q3=panel_inc['ybar'].quantile(0.6)
y_q4=panel_inc['ybar'].quantile(0.8)

#
panel_q1=panel_inc.loc[panel_inc['ybar'] <= y_q1 ]
panel_q1.describe()

sns.distplot(panel_q1['b1'], kde=True).set_title("Beta Distribution - 1st income group")
#sns.distplot(panel_q1['ybar'], kde=True).set_title("Income Distribution - 1st income group")
#panel_q1.hist()

panel_q2=panel_inc.loc[ (y_q1 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q2) ]
panel_q2.describe()

sns.distplot(panel_q2['b1'], kde=True).set_title("Beta Distribution - 2nd income group")
#sns.distplot(panel_q2['ybar'], kde=True).set_title("Income Distribution - 2nd income group")

#
panel_q3=panel_inc.loc[ (y_q2 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q3) ]
panel_q3.describe()

sns.distplot(panel_q3['b1'], kde=True).set_title("Beta Distribution - 3rd income group")
#sns.distplot(panel_q3['ybar'], kde=True).set_title("Income Distribution - 3rd income group")

#
panel_q4=panel_inc.loc[ (y_q3 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q4) ]
panel_q4.describe()

sns.distplot(panel_q4['b1'], kde=True).set_title("Beta Distribution - 4th income group")
#sns.distplot(panel_q4['ybar'], kde=True).set_title("Income Distribution - 4th income group")

#
panel_q5=panel_inc.loc[panel_inc['ybar'] > y_q4 ]
panel_q5.describe()

sns.distplot(panel_q5['b1'], kde=True).set_title("Beta Distribution - 5th income group")
#sns.distplot(panel_q5['ybar'], kde=True).set_title("Income Distribution - 5th income group")

panel_income=panel_inc

del panel_q1,panel_q2,panel_q3,panel_q4,panel_q5

#%%
# =============================================================================
# Q2-B: Relationship between insurance and income - BETA RANKING
# =============================================================================

#%% BETA RANKING
panel_inc=panel_inc.sort_values(["b1"])

b_q1=panel_inc['b1'].quantile(0.2)
b_q2=panel_inc['b1'].quantile(0.4)
b_q3=panel_inc['b1'].quantile(0.6)
b_q4=panel_inc['b1'].quantile(0.8)

#
panel_q1b=panel_inc.loc[panel_inc['b1'] <= b_q1 ]
panel_q1b.describe()

#sns.distplot(panel_q1b['b1'], kde=True).set_title("Beta Distribution - 1st beta group")
sns.distplot(panel_q1b['ybar'], kde=True).set_title("Income Distribution - 1st beta group")

#
panel_q2b=panel_inc.loc[ (b_q1 < panel_inc['b1']) & (panel_inc['b1'] <= b_q2) ]
panel_q2b.describe()

#sns.distplot(panel_q2b['b1'], kde=True).set_title("Beta Distribution - 2nd beta group")
sns.distplot(panel_q2b['ybar'], kde=True).set_title("Income Distribution - 2nd beta group")

#
panel_q3b=panel_inc.loc[ (b_q2 < panel_inc['b1']) & (panel_inc['b1'] <= b_q3) ]
panel_q3b.describe()

#sns.distplot(panel_q3b['b1'], kde=True).set_title("Beta Distribution - 3rd beta group")
sns.distplot(panel_q3b['ybar'], kde=True).set_title("Income Distribution - 3rd beta group")

#
panel_q4b=panel_inc.loc[ (b_q3 < panel_inc['b1']) & (panel_inc['b1'] <= b_q4) ]
panel_q4b.describe()

#sns.distplot(panel_q4b['b1'], kde=True).set_title("Beta Distribution - 4th beta group")
sns.distplot(panel_q4b['ybar'], kde=True).set_title("Income Distribution - 4th beta group")

#
panel_q5b=panel_inc.loc[panel_inc['b1'] > b_q4 ]
panel_q5b.describe()

#sns.distplot(panel_q5b['b1'], kde=True).set_title("Beta Distribution - 5th beta group")
sns.distplot(panel_q5b['ybar'], kde=True).set_title("Income Distribution - 5th beta group")

del panel_q1b,panel_q2b,panel_q3b,panel_q4b,panel_q5b

#%%
# =============================================================================
# Q2-C: Relationship between insurance and wealth/land size - WEALTH RANKING
# =============================================================================





#%%
# =============================================================================
# Q3: FULL INSURANCE - SAME COEFFICIENT
# =============================================================================

# same intercept for all
test_full=sm.ols(formula="Δc ~ Δy + aggct -1 ", data=ps_panel_diff).fit()
print(test_full.summary())
result_full = summary_col([test_full],stars=True)
print(result_full)


#%%
# =============================================================================
#
#                           RURAL VS URBAN
#
# =============================================================================
#%%
# =============================================================================
# Q1-A: Individual insurance  
# =============================================================================

### RURAL
panel_inc_rural=panel_inc.loc[panel_inc['urban_y'] == 0]
panel_inc_rural.describe()

panel_inc_rural=panel_inc_rural.reset_index(inplace=False)
b1f_rural= panel_inc_rural[['hh','b1']]
b1f_rural.set_index(["hh"],inplace=True)
b1f_rural.describe()
b1f_rural.hist()

b2f_rural= panel_inc_rural[['hh','b2']]
b2f_rural.set_index(["hh"],inplace=True)
b2f_rural.describe()
b2f_rural.hist()

### URBAN
panel_inc_urban=panel_inc.loc[panel_inc['urban_y'] == 1]
panel_inc_urban.describe()

panel_inc_urban=panel_inc_urban.reset_index(inplace=False)
b1f_urb= panel_inc_urban[['hh','b1']]
b1f_urb.set_index(["hh"],inplace=True)
b1f_urb.describe()
b1f_urb.hist()

b2f_urb= panel_inc_urban[['hh','b2']]
b2f_urb.set_index(["hh"],inplace=True)
b2f_urb.describe()
b2f_urb.hist()

#%%
# =============================================================================
# Q2-A/C: Relationship between insurance and income - INCOME AND BETA RANKINGS
# =============================================================================

###################################### RURAL

### INCOME RANKING
panel_inc_rural=panel_inc_rural.sort_values(["ybar"])
panel_inc=panel_inc_rural

# Quantiles
y_q1=panel_inc['ybar'].quantile(0.2)
y_q2=panel_inc['ybar'].quantile(0.4)
y_q3=panel_inc['ybar'].quantile(0.6)
y_q4=panel_inc['ybar'].quantile(0.8)

#
panel_q1=panel_inc.loc[panel_inc['ybar'] <= y_q1 ]
panel_q1.describe()

sns.distplot(panel_q1['b1'], kde=True).set_title("Beta Distribution - 1st income group")
#sns.distplot(panel_q1['ybar'], kde=True).set_title("Income Distribution - 1st income group")

#
panel_q2=panel_inc.loc[ (y_q1 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q2) ]
panel_q2.describe()

sns.distplot(panel_q2['b1'], kde=True).set_title("Beta Distribution - 2nd income group")
#sns.distplot(panel_q2['ybar'], kde=True).set_title("Income Distribution - 2nd income group")

#
panel_q3=panel_inc.loc[ (y_q2 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q3) ]
panel_q3.describe()

sns.distplot(panel_q3['b1'], kde=True).set_title("Beta Distribution - 3rd income group")
#sns.distplot(panel_q3['ybar'], kde=True).set_title("Income Distribution - 3rd income group")

#
panel_q4=panel_inc.loc[ (y_q3 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q4) ]
panel_q4.describe()

sns.distplot(panel_q4['b1'], kde=True).set_title("Beta Distribution - 4th income group")
#sns.distplot(panel_q4['ybar'], kde=True).set_title("Income Distribution - 4th income group")

#
panel_q5=panel_inc.loc[panel_inc['ybar'] > y_q4 ]
panel_q5.describe()

sns.distplot(panel_q5['b1'], kde=True).set_title("Beta Distribution - 5th income group")
#sns.distplot(panel_q5['ybar'], kde=True).set_title("Income Distribution - 5th income group")

del panel_q1,panel_q2,panel_q3,panel_q4,panel_q5


### BETA RANKING

b_q1=panel_inc['b1'].quantile(0.2)
b_q2=panel_inc['b1'].quantile(0.4)
b_q3=panel_inc['b1'].quantile(0.6)
b_q4=panel_inc['b1'].quantile(0.8)

#
panel_q1b=panel_inc.loc[panel_inc['b1'] <= b_q1 ]
panel_q1b.describe()

#sns.distplot(panel_q1b['b1'], kde=True).set_title("Beta Distribution - 1st beta group")
sns.distplot(panel_q1b['ybar'], kde=True).set_title("Income Distribution - 1st beta group")

#
panel_q2b=panel_inc.loc[ (b_q1 < panel_inc['b1']) & (panel_inc['b1'] <= b_q2) ]
panel_q2b.describe()

#sns.distplot(panel_q2b['b1'], kde=True).set_title("Beta Distribution - 2nd beta group")
sns.distplot(panel_q2b['ybar'], kde=True).set_title("Income Distribution - 2nd beta group")

#
panel_q3b=panel_inc.loc[ (b_q2 < panel_inc['b1']) & (panel_inc['b1'] <= b_q3) ]
panel_q3b.describe()

#sns.distplot(panel_q3b['b1'], kde=True).set_title("Beta Distribution - 3rd beta group")
sns.distplot(panel_q3b['ybar'], kde=True).set_title("Income Distribution - 3rd beta group")

#
panel_q4b=panel_inc.loc[ (b_q3 < panel_inc['b1']) & (panel_inc['b1'] <= b_q4) ]
panel_q4b.describe()

#sns.distplot(panel_q4b['b1'], kde=True).set_title("Beta Distribution - 4th beta group")
sns.distplot(panel_q4b['ybar'], kde=True).set_title("Income Distribution - 4th beta group")

#
panel_q5b=panel_inc.loc[panel_inc['b1'] > b_q4 ]
panel_q5b.describe()

#sns.distplot(panel_q5b['b1'], kde=True).set_title("Beta Distribution - 5th beta group")
sns.distplot(panel_q5b['ybar'], kde=True).set_title("Income Distribution - 5th beta group")

del panel_q1b,panel_q2b,panel_q3b,panel_q4b,panel_q5b, panel_inc


#%%################################### URBAN

### INCOME RANKING
panel_inc_urban=panel_inc_urban.sort_values(["ybar"])
panel_inc=panel_inc_urban

# Quantiles
y_q1=panel_inc['ybar'].quantile(0.2)
y_q2=panel_inc['ybar'].quantile(0.4)
y_q3=panel_inc['ybar'].quantile(0.6)
y_q4=panel_inc['ybar'].quantile(0.8)

#
panel_q1=panel_inc.loc[panel_inc['ybar'] <= y_q1 ]
panel_q1.describe()

sns.distplot(panel_q1['b1'], kde=True).set_title("Beta Distribution - 1st income group")
#sns.distplot(panel_q1['ybar'], kde=True).set_title("Income Distribution - 1st income group")


#
panel_q2=panel_inc.loc[ (y_q1 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q2) ]
panel_q2.describe()

sns.distplot(panel_q2['b1'], kde=True).set_title("Beta Distribution - 2nd income group")
#sns.distplot(panel_q2['ybar'], kde=True).set_title("Income Distribution - 2nd income group")

#
panel_q3=panel_inc.loc[ (y_q2 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q3) ]
panel_q3.describe()

sns.distplot(panel_q3['b1'], kde=True).set_title("Beta Distribution - 3rd income group")
#sns.distplot(panel_q3['ybar'], kde=True).set_title("Income Distribution - 3rd income group")

#
panel_q4=panel_inc.loc[ (y_q3 < panel_inc['ybar']) & (panel_inc['ybar'] <= y_q4) ]
panel_q4.describe()

sns.distplot(panel_q4['b1'], kde=True).set_title("Beta Distribution - 4th income group")
#sns.distplot(panel_q4['ybar'], kde=True).set_title("Income Distribution - 4th income group")

#
panel_q5=panel_inc.loc[panel_inc['ybar'] > y_q4 ]
panel_q5.describe()

sns.distplot(panel_q5['b1'], kde=True).set_title("Beta Distribution - 5th income group")
#sns.distplot(panel_q5['ybar'], kde=True).set_title("Income Distribution - 5th income group")

del panel_q1,panel_q2,panel_q3,panel_q4,panel_q5


### BETA RANKING

b_q1=panel_inc['b1'].quantile(0.2)
b_q2=panel_inc['b1'].quantile(0.4)
b_q3=panel_inc['b1'].quantile(0.6)
b_q4=panel_inc['b1'].quantile(0.8)

#
panel_q1b=panel_inc.loc[panel_inc['b1'] <= b_q1 ]
panel_q1b.describe()

#sns.distplot(panel_q1b['b1'], kde=True).set_title("Beta Distribution - 1st beta group")
sns.distplot(panel_q1b['ybar'], kde=True).set_title("Income Distribution - 1st beta group")

#
panel_q2b=panel_inc.loc[ (b_q1 < panel_inc['b1']) & (panel_inc['b1'] <= b_q2) ]
panel_q2b.describe()

#sns.distplot(panel_q2b['b1'], kde=True).set_title("Beta Distribution - 2nd beta group")
sns.distplot(panel_q2b['ybar'], kde=True).set_title("Income Distribution - 2nd beta group")

#
panel_q3b=panel_inc.loc[ (b_q2 < panel_inc['b1']) & (panel_inc['b1'] <= b_q3) ]
panel_q3b.describe()

#sns.distplot(panel_q3b['b1'], kde=True).set_title("Beta Distribution - 3rd beta group")
sns.distplot(panel_q3b['ybar'], kde=True).set_title("Income Distribution - 3rd beta group")

#
panel_q4b=panel_inc.loc[ (b_q3 < panel_inc['b1']) & (panel_inc['b1'] <= b_q4) ]
panel_q4b.describe()

#sns.distplot(panel_q4b['b1'], kde=True).set_title("Beta Distribution - 4th beta group")
sns.distplot(panel_q4b['ybar'], kde=True).set_title("Income Distribution - 4th beta group")

#
panel_q5b=panel_inc.loc[panel_inc['b1'] > b_q4 ]
panel_q5b.describe()

#sns.distplot(panel_q5b['b1'], kde=True).set_title("Beta Distribution - 5th beta group")
sns.distplot(panel_q5b['ybar'], kde=True).set_title("Income Distribution - 5th beta group")

del panel_q1b,panel_q2b,panel_q3b,panel_q4b,panel_q5b, panel_inc

#%%
# =============================================================================
# Q2-C: Relationship between insurance and wealth/land size - WEALTH RANKING
# =============================================================================



#%%
# =============================================================================
# Q3: FULL INSURANCE - SAME COEFFICIENT
# =============================================================================

ps_panel_diff_loc = ps_panel.groupby(level=0)['error_c','error_y'].diff()
ps_panel_diff_loc.columns = ['Δc','Δy']
ps_panel_diff_loc['urban']=ps_panel['urban_y']
ps_panel_diff_loc['wave']=ps_panel['wave']
ps_panel_diff_loc.reset_index(inplace=True)
ps_panel_diff_loc.set_index(["hh"],inplace=True)
ps_panel_diff_loc = ps_panel_diff_loc[ps_panel_diff_loc.wave != "2009-2010"]
#ps_panel_diff=ps_panel_diff.reset_index(inplace=False)        
ps_panel_diff_loc['aggctt']=ca    

##################### RURAL
panel_diff_rural=ps_panel_diff_loc.loc[ps_panel_diff_loc['urban'] == 0]
# same intercept for all
test_rural=sm.ols(formula="Δc ~ Δy + aggctt -1 ", data=panel_diff_rural).fit()
#test_rural.params
print(test_rural.summary())
result_rural = summary_col([test_rural],stars=True)
print(result_rural)

##################### URBAN
panel_diff_urban=ps_panel_diff_loc.loc[ps_panel_diff_loc['urban'] == 1]
# same intercept for all
test_urb=sm.ols(formula="Δc ~ Δy + aggctt -1 ", data=panel_diff_urban).fit()
#test_rural.params
print(test_urb.summary())
result_urb = summary_col([test_urb],stars=True)
print(result_urb)





