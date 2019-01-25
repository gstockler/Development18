#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:33:48 2019

@author: gabi
"""


# =============================================================================
#  CREATING DATA SET FOR UGANDA 2013-2014
# =============================================================================

import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as sm

from data_functions_albert import remove_outliers

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/UGA_2013_14_data')

from statsmodels.iolib.summary2 import summary_col
pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89    #https://data.worldbank.org/indicator/PA.NUS.FCRF

hhid=pd.read_csv('hh_ident.csv')
inflation = pd.read_excel("inflation.xlsx")

#%% IMPORT HH INFO DATA

basic = pd.read_stata('GSEC1.dta', convert_categoricals=False )
basic = basic[["HHID","year", "month"]] 
basic.rename(columns={'HHID':'hh'}, inplace=True)

# checking
pd.value_counts(basic["year"])
pd.value_counts(basic["month"])

basic = basic.dropna(subset = ['year'])
basic["index"] = range(0,len(basic))
basic.set_index(basic["index"], inplace=True)

# Inflation per HH
inflation = pd.read_excel("inflation.xlsx")
lulu = []
for i in range(0,len(basic)):
    a = inflation.loc[basic.loc[i,"year"], basic.loc[i,"month"]]
    lulu.append(a)    
basic["inflation"] = pd.Series(lulu)

del inflation, i, lulu, a

#%% C - Consumption
os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/')

cons = pd.read_csv("cons.csv")
cons = cons[["hh","ctotal","ctotal_dur","ctotal_gift","cfood","cnodur"]]

"""
where
    ctotal: food + nofood
    ctotal dur: food + nofood + durables
    ctotal gift: food + nofood of gifts
    cfood: total only on food
    cnodur: total only on non-durables
"""

data = pd.merge(basic, cons, on="hh", how="left")

del cons

#%% W - WEALTH

wealth = pd.read_csv('wealth.csv')

"""
where
    farming assets
    household assets
    total wealth
"""

data = pd.merge(data, wealth, on='hh', how='inner')

del wealth

#%% I - INCOME

### Labor & Business income: in US dollars

lab_inc = pd.read_csv('income_hhsec.csv', header=0, na_values='nan')
#lab_inc[["wage_total","bs_profit", "other_inc"]] = lab_inc[["wage_total","bs_profit", "other_inc"]]*dollars #Pass it to UGsh
#lab_inc[["wage_total","bs_profit", "other_inc"]] = remove_outliers(lab_inc[["wage_total","bs_profit", "other_inc"]], lq=0.001, hq=0.999)

### Agricultural income: in UG Shillings

# first, need to match household indentifier
hhid=pd.read_csv('hh_ident.csv')

ag_inc = pd.read_csv('income_agsec.csv', header=0, na_values='nan')
aa = pd.merge(hhid, ag_inc, on="hh", how="outer")
aa.rename(columns={'hh':'hh_dif'}, inplace=True)
aa.rename(columns={'hh2':'hh'}, inplace=True)

ag_inc=aa # now ag_inc has the correct identifier

del aa, hhid

### Combining Income Sources

inc = pd.merge(lab_inc, ag_inc, on="hh", how="outer")
inc=inc.drop(['hh_dif'],axis=1)
inc = inc.drop(inc.columns[[0,5]], axis=1)

inc["inctotal"] = inc.loc[:,["wage_total","bs_profit","total_agrls"]].sum(axis=1)
inc["inctotal_trans"] = inc.loc[:,["wage_total","bs_profit","other","total_agrls"]].sum(axis=1)
inc["inctotal"] = inc["inctotal"].replace(0,np.nan)

### Summary Statistics
suminc1 = inc.describe()


### Income share

inc["w_share"] = inc[["wage_total"]].divide(inc.inctotal, axis=0)
inc["worker"] = (inc.w_share>=0.329)*1
inc["agr_share"] = inc[["total_agrls"]].divide(inc.inctotal, axis=0)
inc["farmer"] =  (inc.agr_share>=0.329)*1
inc["bus_share"] = inc[["bs_profit"]].divide(inc.inctotal, axis=0)
inc["businessman"] = (inc.bus_share>=0.329)*1

### Income occupations

inc["ocupation"] = "nothing"
inc.loc[inc["worker"]==1, "ocupation"] = "worker"
inc.loc[inc["farmer"]==1, "ocupation"] = "farmer"
inc.loc[inc["businessman"]==1, "ocupation"] = "businessman"

### Merging data

data = data.merge( inc, on='hh', how='left')
del ag_inc, lab_inc, inc

#%%  ACCOUNTING FOR INFLATION

# desinflation and conversion to 2013 US$

# Substract for inflation and convert to US dollars
data[["ctotal","ctotal_dur","ctotal_gift","cfood","cnodur", "wage_total","bs_profit", "other_inc", "profit_agr","profit_ls", "total_agrls", "inctotal", "inctotal_trans","farm_asset","hh_asset","wtotal"]]= data[["ctotal","ctotal_dur","ctotal_gift","cfood","cnodur", "wage_total","bs_profit", "other_inc", "profit_agr","profit_ls", "total_agrls", "inctotal", "inctotal_trans","farm_asset","hh_asset","wtotal"]].div(data.inflation, axis=0)#/dollars

#%% SOCIODEMOGRAPHICS

socio = pd.read_csv("sociodem.csv")
#socio.drop(socio.columns[0], axis=1, inplace= True)
data = pd.merge(data, socio, on="hh", how="left")
data = data.drop_duplicates(subset=['hh'], keep=False)
#repeated = pd.DataFrame(pd.value_counts(data.hh))
#%% Summary Statistics

data.to_csv('data_13_14.csv')

# consumption
sum_c = data[["ctotal","ctotal_dur","ctotal_gift","cfood","cnodur"]].describe()
sum_c.to_csv('sum_c.csv')

# income
sum_inc = data[["wage_total","bs_profit","profit_agr","profit_ls","inctotal"]].describe()
sum_inc.to_csv('sum_inc.csv')

# wealth
sum_w = data[["farm_asset","hh_asset","wtotal"]].describe()
sum_w.to_csv('sum_w.csv')


#%% CIW DATA

#Store data
data_cwi = data[["hh", "ctotal", "inctotal","wtotal","sex","region","urban","classeduc"]]
data_cwi.to_csv('cwi_data.csv')

sum_data = data.describe()

# Dealing with outliers
# Trimming 0.1% and 0.1% each tail
"""
for serie in ["ctotal", "wtotal", "inctotal"]:
    data['percentiles'] = pd.qcut(data[serie], [0.001,0.999], labels = False)
    data.dropna(axis=0, subset=['percentiles'], inplace=True)
    data.drop('percentiles', axis=1, inplace=True)

"""


"""

CIW DATA SET DONE

"""
#%%

