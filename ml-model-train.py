import numpy as np
import pandas as pd
import pickle
from scipy.stats import entropy
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

insulinDf1 = pd.read_csv('InsulinData.csv', low_memory=False)
insulinDf2 = pd.read_csv('Insulin_patient2.csv', low_memory=False)

insulinDf1 = insulinDf1[['Date', 'Time', 'BWZ Carb Input (grams)']]
insulinDf2 = insulinDf2[['Date', 'Time', 'BWZ Carb Input (grams)']]
insulinDf1['Date_Time'] = pd.to_datetime(insulinDf1['Date'] + ' ' + insulinDf1['Time'])
insulinDf2['Date_Time'] = pd.to_datetime(insulinDf2['Date'] + ' ' + insulinDf2['Time'])

insulinDf1 = insulinDf1.dropna(axis=0)
insulinDf2 = insulinDf2.dropna(axis=0)
insulin_date_time1 = insulinDf1['Date_Time'].tolist()
insulin_date_time1.sort()
insulin_date_time2 = insulinDf2['Date_Time'].tolist()
insulin_date_time2.sort()

cgm1 = pd.read_csv('CGMData.csv', low_memory=False)
cgm2 = pd.read_csv('CGM_patient2.csv', low_memory=False)
cgm1 = cgm1[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
cgm2 = cgm2[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
cgm1['Date_Time'] = pd.to_datetime(cgm1.Date + ' ' + cgm1.Time)
cgm2['Date_Time'] = pd.to_datetime(cgm2.Date + ' ' + cgm2.Time)



def getMeal(cgmDf, insulinData, gettingMeal):
    i = 0
    mealDf = pd.DataFrame()
    if gettingMeal:
        # Getting a meal
        while (i < len(insulinData)):
            start = insulinData[i] - pd.Timedelta(minutes=30)
            end = insulinData[i] + pd.Timedelta(hours=2)
            while (i + 1 < len(insulinData) and insulinData[i + 1] < end):
                i += 1
                start = insulinData[i] - pd.Timedelta(minutes=30)
                end = insulinData[i] + pd.Timedelta(hours=2)
            data = cgmDf[(cgmDf['Date_Time'] >= start) & (cgmDf['Date_Time'] <= end)]
            df = data[data['Sensor Glucose (mg/dL)'].notnull()]
            glucose = df['Sensor Glucose (mg/dL)'].tolist()
            if (len(glucose) == 30):
                data = {}
                for index, x in enumerate(glucose):
                    data[index] = x
                mealDf = mealDf.append(data, ignore_index=True)
            i = i + 1
    else:
        # Not getting a meal
        while (i < len(insulinData)):
            start = insulinData[i] + pd.Timedelta(hours=2)
            end = start + pd.Timedelta(hours=2)
            while (i + 1 < len(insulinData) and insulinData[i + 1] < end):
                i += 1
                start = insulinData[i] + pd.Timedelta(hours=2)
                end = start + pd.Timedelta(hours=2)
            data = cgmDf[(cgmDf['Date_Time'] >= start) & (cgmDf['Date_Time'] <= end)]
            df = data[data['Sensor Glucose (mg/dL)'].notnull()]
            glucose = df['Sensor Glucose (mg/dL)'].tolist()
            if (len(glucose) == 24):
                data = {}
                for index, x in enumerate(glucose):
                    data[index] = x
                mealDf = mealDf.append(data, ignore_index=True)
            i = i + 1
    return mealDf

def extractFeatures(mealDf):
    ansDf = pd.DataFrame()
    for _, row in mealDf.iterrows():
        r = row.tolist()
        feature1 = float(np.correlate(np.array(r), np.array(r)))
        feature2 = entropy(r)
        newRow = {'f1': feature1, 'f2': feature2}
        ansDf = ansDf.append(newRow, ignore_index=True)
    return ansDf

def train(xn, yn):
    model = SVC().fit(xn, yn)
    return model

def test(xt):
    model = pickle.load(open('model.pkl', 'rb'))
    p = model.predict(xt)
    return p

mealDf1 = getMeal(cgm1, insulin_date_time1, True)
mealDf2 = getMeal(cgm2, insulin_date_time2, True)
noMealDf1 = getMeal(cgm1, insulin_date_time1, False)
noMealDf2 = getMeal(cgm2, insulin_date_time2, False)
Meal = pd.concat([mealDf1, mealDf2])
NoMeal = pd.concat([noMealDf1, noMealDf2])

mealFeatures = extractFeatures(Meal)
mealFeatures['label'] = '1'

noMealFeatures = extractFeatures(NoMeal)
noMealFeatures['label'] = '0'

featuresDf = pd.concat([mealFeatures, noMealFeatures])
labels = featuresDf['label'].tolist()
dataDf = featuresDf.drop(['label'], axis=1)
data = dataDf.values.tolist()
data, labels = shuffle(data, labels)
xn, xt, yn, yt = train_test_split(data, labels, train_size=0.7)

scaler = MinMaxScaler()
xn = scaler.fit_transform(xn)
xt = scaler.transform(xt)
model = train(xn, yn)
pickle.dump(model, open('model.pkl', 'wb'))
yp = test(xt)

featuresDf = shuffle(featuresDf)
data = np.array_split(featuresDf, 5)

def evaluate(x_train, y_train):
    kf = KFold(n_splits=2, shuffle=True)
    model_scores = cross_val_score(SVC(), x_train, y_train, cv=kf)
    print(model_scores)

for cv in range(len(data)):
    trdf = pd.DataFrame()
    tdf = pd.DataFrame()
    for i in range(len(data)):
        if i == cv:
            tdf = tdf.append(data[i])
        else:
            trdf = trdf.append(data[i])
    yn = trdf['label'].tolist()
    df = trdf.drop(['label'], axis=1)
    xn = df.values.tolist()
    yt = tdf['label'].tolist()
    dft = tdf.drop(['label'], axis=1)
    xt = dft.values.tolist()

    scaler = MinMaxScaler()
    xn = scaler.fit_transform(xn)
    xt = scaler.transform(xt)

    clf = SVC().fit(xn, yn)
    yp = clf.predict(xt)
    print(evaluate(xt, yt))

