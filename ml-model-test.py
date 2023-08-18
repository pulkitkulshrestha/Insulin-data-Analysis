import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy
import pickle
import csv

def extractFeatures(testDf):
    f = pd.DataFrame()
    for _, row in testDf.iterrows():
        r = row.tolist()
        f1 = float(np.correlate(np.array(r), np.array(r)))
        f2 = entropy(r)
        newRow = {'f1': f1, 'f2': f2}
        f = f.append(newRow, ignore_index=True)
    return f

def test(model, featureList):
    return model.predict(featureList)

test_df = pd.read_csv("test.csv")
feature_df = extractFeatures(test_df)
model = pickle.load(open('model.pkl', 'rb'))
scaler = MinMaxScaler()
features = scaler.fit_transform(feature_df.values.tolist())
test_result = test(model, features)
with open('Results.csv', mode='wb') as result:
    fd = csv.writer(result, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in test_result:
        fd.writerow([i])
