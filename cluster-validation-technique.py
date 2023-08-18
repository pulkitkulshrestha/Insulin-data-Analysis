
import pandas as pd
from pandas import DataFrame
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix


def binValue(Ins,len):
    Ins['min_val'] = Ins['ins'].min()
    Ins['bins'] = ((Ins['ins'] - Ins['min_val'])/20).apply(np.ceil)
    binT = pd.concat([len, Ins], axis=1)
    binT = binT[binT['len'].notna()]
    binT.drop(binT[binT['len'] < 30].index, inplace=True)
    Ins.reset_index(drop=True, inplace=True)
    return binT

def SSE(x):
    a = 0
    if len(x) != 0:
        mean = sum(x) / len(x)
        for i in x:
            a += (i - mean) * (i - mean)
    return a

def BinsFinal(correct,res,no_Clus):
    binRes = []
    bin = []
    for i in range(no_Clus):
        binRes.append([])
        bin.append([])
    for i in range(len(res)):
        binRes[res[i]-1].append(i)
    for i in range(no_Clus):
        for j in binRes[i]:
            bin[i].append(correct[j])
    return bin

data_insulin = pd.read_csv('./InsulinData.csv', dtype='unicode')
data_insulin['DateTime'] = pd.to_datetime(data_insulin['Date'] + " " + data_insulin['Time'])
data_insulin = data_insulin[["Date", "Time", "DateTime", "BWZ Carb Input (grams)"]]
data_insulin['ins'] = data_insulin['BWZ Carb Input (grams)'].astype(float)
data_insulin = data_insulin[(data_insulin.ins != 0)]
data_insulin = data_insulin[data_insulin['ins'].notna()]
data_insulin = data_insulin.drop(columns=['Date', 'Time','BWZ Carb Input (grams)']).sort_values(by=['DateTime'], ascending=True)
data_insulin.reset_index(drop=True, inplace=True)
data_insulinShift = data_insulin.shift(-1)
data_insulin = data_insulin.join(data_insulinShift.rename(columns=lambda x: x+"_lag"))
data_insulin['tot_mins_diff'] = (data_insulin.DateTime_lag - data_insulin.DateTime) / pd.Timedelta(minutes=1)
data_insulin['Patient'] = 'P1'
data_insulin.drop(data_insulin[data_insulin['tot_mins_diff'] < 120].index, inplace = True)
data_insulin = data_insulin[data_insulin['ins_lag'].notna()]

data_CGM = pd.read_csv('./CGMData.csv', sep=',', usecols=['Index','Date','Time','Sensor Glucose (mg/dL)'])
data_CGM['TimeStamp'] = pd.to_datetime(data_CGM['Date'] + ' ' + data_CGM['Time'])
data_CGM['CGM'] = data_CGM['Sensor Glucose (mg/dL)']
data_CGM = data_CGM[['Index','TimeStamp','CGM','Date','Time']]
data_CGM = data_CGM.sort_values(by=['TimeStamp'], ascending=True).fillna(method='ffill')
data_CGM = data_CGM.drop(columns=['Date', 'Time','Index']).sort_values(by=['TimeStamp'], ascending=True)
data_CGM = data_CGM[data_CGM['CGM'].notna()]
data_CGM.reset_index(drop=True, inplace=True)


Mealtime = []
for x in data_insulin.index:
    Mealtime.append([data_insulin['DateTime'][x] + pd.DateOffset(hours=-0.5),
                     data_insulin['DateTime'][x] + pd.DateOffset(hours=+2)])
Meal = []
for x in range(len(Mealtime)):
    data = data_CGM.loc[(data_CGM['TimeStamp'] >= Mealtime[x][0]) & (data_CGM['TimeStamp'] < Mealtime[x][1])]['CGM']
    Meal.append(data)
MealLength = []
MealF = []
b = 0
for a in Meal:
    b = len(a)
    MealLength.append(b)
    if len(a) == 30:
        MealF.append(a)
Length = DataFrame(MealLength, columns=['len'])
Length.reset_index(drop=True, inplace=True)
c, c_len = MealF, Length

true = binValue(data_insulin, c_len)
F_matrix = np.vstack(c)
df = StandardScaler().fit_transform(F_matrix)
no_Clus = int((data_insulin["ins"].max() - data_insulin["ins"].min()) / 20)


kmeans = KMeans(n_clusters=no_Clus, random_state=0).fit(df)
true_bin = true["bins"]
true_Label = np.asarray(true_bin).flatten()
for i in range(len(true_Label)):
    if math.isnan(true_Label[i]):
        true_Label[i] = 1
bins = BinsFinal(true_Label,kmeans.labels_, no_Clus)
kMeans_ss = 0
for i in range(len(bins)):
    kMeans_ss += (SSE(bins[i]) * len(bins[i]))
kMeansC = contingency_matrix(true_Label, kmeans.labels_)
entropy, purity = [], []

for clus in kMeansC:
    clus = clus / float(clus.sum())
    t_Ent = 0
    for x in clus:
        if x != 0:
            t_Ent = (clus * [math.log(x, 2)]).sum()*-1
        else:
            t_Ent = clus.sum()
    clus = clus*3.5
    entropy += [t_Ent]
    purity += [clus.max()]
cnt = np.array([c.sum() for c in kMeansC])
cef = cnt / float(cnt.sum())
kM_Ent = (cef * entropy).sum()
kM_Pur = (cef * purity).sum()


new_fea = []
for i in F_matrix:
    new_fea.append(i[1])
new_fea = (np.array(new_fea)).reshape(-1, 1)
X = StandardScaler().fit_transform(new_fea)
db = DBSCAN(eps=0.05, min_samples=2).fit(X)
bins = BinsFinal(true_Label,db.labels_, no_Clus)
db_ss = 0
for i in range(len(bins)):
     db_ss += (SSE(bins[i]) * len(bins[i]))
db_C = contingency_matrix(true_Label, db.labels_)
entropy, purity = [], []

for clus in db_C:
    clus = clus / float(clus.sum())
    t_ent = 0
    for x in clus:
        if x != 0:
            t_ent = (clus * [math.log(x, 2)]).sum()*-1
        else:
            t_ent = (clus * [math.log(x+1, 2)]).sum()*-1
    entropy += [t_ent]
    purity += [clus.max()]
cnt = np.array([c.sum() for c in kMeansC])
cef = cnt / float(cnt.sum())
db_ent = (cef * entropy).sum()
db_pur = (cef * purity).sum()

resultDF=pd.DataFrame([kMeans_ss, db_ss, kM_Ent, db_ent, kM_Pur, db_pur]).T
resultDF.to_csv('Results.csv', header = False, index = False)

