#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 23:23:36 2019

@author: gabi
"""

# =============================================================================
#  MERGING SOCIODEMOGRAPHIC CHARACTERISTICS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

pd.options.display.float_format = '{:,.2f}'.format

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/UGA_2013_14_data')

# HH survey

#%% REGION
# at the HH level

region = pd.read_stata('GSEC1.dta', convert_categoricals=False)
region = region[["HHID","region"]]
region.columns =["hh","region"]
region.region = pd.to_numeric(region.region)

#%% RURAL/URBAN
# at HH level
# 1,2 = urban ; 3 = rural

urban = pd.read_stata('GSEC1.dta', convert_categoricals=False)
urban = urban[["HHID","urban"]]
urban.columns =["hh","urban"]
urban.urban = urban.urban.replace([2, 3],[1, 0])
urban.urban = pd.to_numeric(urban.urban)

#%% AGE
# at the individual level

age = pd.read_stata('GSEC2.dta', convert_categoricals=False)
age = age[["PID","h2q3","h2q4","h2q8"]]
age.columns = ["pid","sex","hh_member", "age"]

#%% HEALTH
# days being ill over the last month
# at the individual level

health = pd.read_stata('GSEC5.dta', convert_categoricals=False)
health = health[["PID","h5q5"]]
health.columns = ["pid","illdays"]


#%% BACKGROUND AND BEDNETS
# at the individual level

bck = pd.read_stata('GSEC3.dta', convert_categoricals=False)
bck = bck[["PID","h3q3","h3q4", "h3q9","h3q10"]]
bck.columns = ["pid","father_educ", "father_ocup","ethnic","bednet"]
bck.father_educ = bck.father_educ.replace(99,np.nan)
#Group bednet answer as yes I have, no 
bck.bednet = bck.bednet.replace([2 , 3, 9],[1 , 0, np.nan])


#%% EDUCATION

educ = pd.read_stata('GSEC4.dta', convert_categoricals=False)
educ = educ[["HHID","PID","h4q4", "h4q7"]]
educ.columns = ["hh","pid","writeread","classeduc"]
educ.writeread = educ.writeread.replace([2, 4, 5],0)
educ.loc[educ["classeduc"]==99, "classeduc"] = np.nan
#1 if able to read and write. 0 if unable both, unable writing, uses braille

#%% SOCIODEMO DATASET

socio = pd.merge(age, bck, on="pid", how="inner")
socio = pd.merge(socio, educ, on="pid", how="inner")
socio = pd.merge(socio, health, on="pid", how="inner")
socio = socio.loc[(socio.hh_member==1)]
socio.drop(["hh_member", "pid"], axis=1, inplace=True)
#socio.drop(socio.columns[0], axis=1, inplace=True)

socio = pd.merge(socio, region, on="hh", how="inner")
socio = pd.merge(socio, urban, on="hh", how="inner")

socio.to_csv("sociodem.csv")

#%% SAVING
os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/')
socio.to_csv("sociodem.csv")