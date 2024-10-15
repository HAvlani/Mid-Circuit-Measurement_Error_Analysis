import os
import pandas as pd
import numpy as np
import rbbuild
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sympy import *
from mpmath import *

rootdir ='/Users/harshilavlani/MCM Project'
for f0 in os.scandir(rootdir):
    if f0.is_dir() and f0.name != '__pycache__':
        print(f"f0 {f0}")
        dflist = []
        for f1 in os.scandir(f0):
            if f1.is_dir() :
                print(f"f1 {f1.name}")
                for f2 in os.scandir(f1):
                    if f2.name.startswith('e'):
                        print(f"f2 {f2.name}")
                        q = pd.read_csv(f2)
                        if {'Pair', 'Type', 'Unnamed: 0'}.issubset(q.columns):
                            errorcounts = q.drop(columns=['Pair', 'Type', 'Unnamed: 0'])
                        else:
                            errorcounts = q
                        gatelist = []
                        for i in errorcounts.columns:
                            gatelist.append(int(i)+1)
                        x = np.array(gatelist)
                        ymcm_0_1 = np.array(list(errorcounts.iloc[0]))
                        ydel_0_1 = np.array(list(errorcounts.iloc[1]))
                        ymcm_4_6 = np.array(list(errorcounts.iloc[2]))
                        ydel_4_6 = np.array(list(errorcounts.iloc[3]))
                        ymcm_0_6 = np.array(list(errorcounts.iloc[4]))
                        ydel_0_6 = np.array(list(errorcounts.iloc[5]))
                        p0 = (4000, .5, 100)
                        params_mcm_0_1, cv_mcm_0_1 = curve_fit(rbbuild.expfunction_alpha, x, ymcm_0_1, p0, maxfev=5000)
                        params_del_0_1, cv_del_0_1 = curve_fit(rbbuild.expfunction_alpha, x, ydel_0_1, p0, maxfev=5000)
                        params_mcm_4_6, cv_mcm_4_6 = curve_fit(rbbuild.expfunction_alpha, x, ymcm_4_6, p0, maxfev=5000)
                        params_del_4_6, cv_del_4_6 = curve_fit(rbbuild.expfunction_alpha, x, ydel_4_6, p0, maxfev=5000)
                        params_mcm_0_6, cv_mcm_0_6 = curve_fit(rbbuild.expfunction_alpha, x, ymcm_0_6, p0, maxfev=5000)
                        params_del_0_6, cv_del_0_6 = curve_fit(rbbuild.expfunction_alpha, x, ydel_0_6, p0, maxfev=5000)
                        labellist = ['ymcm_0_1', 'ydel_0_1', 'ymcm_4_6', 'ydel_4_6', 'ymcm_0_6', 'ydel_0_6']
                        pairlist = ['0_1', '0_1', '4_6', '4_6', '0_6', '0_6']
                        typelist = ['MCM', "Del", "MCM", "Del", "MCM", "Del"]
                        ylist = [ymcm_0_1, ydel_0_1, ymcm_4_6, ydel_4_6, ymcm_0_6, ydel_0_6]
                        paramlist = [params_mcm_0_1, params_del_0_1, params_mcm_4_6, params_del_4_6, params_mcm_0_6, params_del_0_6]
                        asymptotelist = []
                        xvallist = []
                        qcnamelist = []
                        joblist = []
                        adjpairlist = []
                        adjlabellist = []
                        adjtypelist = []
                        
                        for i in range(len(paramlist)):
                            squaredDiffs = np.square(ylist[i] - rbbuild.expfunction_alpha(x, paramlist[i][0], paramlist[i][1], paramlist[i][2]))
                            squaredDiffsFromMean = np.square(ylist[i] - np.mean(ylist[i]))
                            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
                            #if rSquared > 0.6:
                            plt.scatter(x, ylist[i], label=labellist[i])
                            plt.plot(x, rbbuild.expfunction_alpha(x, paramlist[i][0], paramlist[i][1], paramlist[i][2]), '--', label="fitted")
                            plt.text(0, 1000, f"{rSquared}")
                            plt.text(0, 900, f"Y = {paramlist[i][0]} * {paramlist[i][1]}^(x) + {paramlist[i][2]}")
                            expr = paramlist[i][0]*(paramlist[i][1]**x) + paramlist[i][2]
                            asymptote = limit(lambda n: paramlist[i][0]*(paramlist[i][1]**n) + paramlist[i][2], inf)
                            #print(type(asymptote))
                            for q in range(0, 400):
                                val = paramlist[i][0]*(paramlist[i][1]**q) + paramlist[i][2]
                                if abs(asymptote - val) < 30:
                                    xval = q
                                    break
                                else: 
                                    xval = 400000
                            #if asymptote > 1000 or asymptote < 1900:
                                #xval = 4
                            asymptotelist.append(asymptote)
                            #print(asymptote)
                            xvallist.append(xval)
                            qcnamelist.append(f0.name)
                            joblist.append(f1.name)
                            adjpairlist.append(pairlist[i])
                            adjlabellist.append(labellist[i])
                            adjtypelist.append(typelist[i])
                            plt.legend()
                            plt.savefig(f"{f0.name}/{f1.name}/{labellist[i]}")
                            plt.clf()                
                        df = pd.DataFrame({'Asymptotes': asymptotelist, 'Depth': xvallist, 'QC': qcnamelist, 'Job': joblist, 'Pair': adjpairlist, 'Type': adjtypelist})
                        dflist.append(df)
                    #plt.scatter(x, ydel_0_1, label=f" {delrb_0_1")
                                
        finaldf = pd.concat(dflist)
        finaldf.set_index('Job')
        finaldf.to_csv(f'{rootdir}/{f0.name}/finaldf.csv')