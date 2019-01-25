#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:57:00 2019

@author: gabi
"""


# =============================================================================
#  WEALTH DATA
# =============================================================================

import pandas as pd
import numpy as np
import os

from data_functions_albert import remove_outliers

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/UGA_2013_14_data')

pd.options.display.float_format = '{:,.2f}'.format

dollars = 2586.89

#%% FARMING ASSETS
ag10 = pd.read_stata('AGSEC10.dta')
ag10 = ag10[["HHID", "A10itemcod","a10q2"]]
ag10 = ag10.groupby(by="HHID")[["a10q2"]].sum().fillna(0)
ag10.columns=["farm_asset"]
ag10["hh"] = np.array(ag10.index.values)

# fixing identifier
ident = pd.read_stata('AGSEC1.dta', convert_categoricals=False)
ident = ident[["hh","HHID"]]
ident.columns=["hh2","hh"] #correct is hh2
ident.to_csv("hh_ident.csv")

hhid=pd.read_csv('hh_ident.csv')

ii = pd.merge(hhid, ag10, on="hh", how="outer")
ii.rename(columns={'hh':'hh_dif'}, inplace=True)
ii.rename(columns={'hh2':'hh'}, inplace=True)
ii=ii.drop(['hh_dif'],axis=1)
ii=ii[["hh","farm_asset"]]

w_farm = ii

del ii, ag10

#%% LIVESTOCK ASSETS ???



#%% HH ASSETS
c14 = pd.read_stata('GSEC14A.dta')
c14 = c14[["HHID","h14q2","h14q5"]]
c14 = c14.groupby(by="HHID")[["h14q5"]].sum().fillna(0)
c14.columns=["hh_asset"]
c14["hh"] = np.array(c14.index.values)


#%% MERGING DATA
wealth = pd.merge(w_farm, c14, on="hh", how="inner")
wealth=wealth[["hh","farm_asset","hh_asset"]]
wealth["wtotal"] = wealth[["farm_asset","hh_asset"]].sum(axis=1)
wealth[["farm_asset","hh_asset","wtotal" ]]=remove_outliers(wealth[["farm_asset","hh_asset","wtotal" ]] ,lq=0.001, hq=0.999 )
wealth[["farm_asset","hh_asset","wtotal" ]]=wealth[["farm_asset","hh_asset","wtotal" ]]/dollars
wealth.to_csv("wealth.csv")

#%% SAVING
os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/')
wealth.to_csv("wealth.csv")

