import csv
import pandas as pd

insulin_data_csv = pd.read_csv('InsulinData.csv')
insulin_data_csv.head()

insulin_data_csv.columns

req_columns = ['Date', 'Time', 'Alarm']
insulin_data_csv.drop(insulin_data_csv.columns.difference(req_columns), 1, inplace=True)
insulin_data_csv.head()

insulin_data_csv.columns

# reverse insulin data file
insulin_data_csv = insulin_data_csv.reindex(index=insulin_data_csv.index[::-1])
insulin_data_csv.head()

MODE = 'AUTO MODE ACTIVE PLGM OFF'
first_occurence = insulin_data_csv[insulin_data_csv.Alarm == MODE].index[0]
first_occurence

# delete all data before the first occurrence
cnt=insulin_data_csv.shape[0]
cnt=cnt-first_occurence-1
cnt
insulin_data_csv_Auto = insulin_data_csv.iloc[cnt:]
insulin_data_csv_Manual = insulin_data_csv.iloc[:cnt,:]
#insulin_data_csv_Manual = insulin_data_csv.head[first_occurence:]
insulin_data_csv_Auto.head()
insulin_data_csv_Manual.head()

insulin_data_csv_date = insulin_data_csv.iloc[cnt]

threshold_date_str = insulin_data_csv_date['Date'] + ' ' + insulin_data_csv_date['Time']
print(threshold_date_str)
threshold_date = pd.to_datetime(threshold_date_str, format='%m/%d/%Y %H:%M:%S')
threshold_date

cgm_data_csv = pd.read_csv('CGMData.csv')

cgm_data_csv.head()

req_columns = ['Date', 'Time', 'Sensor Glucose (mg/dL)']
cgm_data_csv.drop(cgm_data_csv.columns.difference(req_columns), 1, inplace=True)
cgm_data_csv.head()

cgm_data_csv['date'] = pd.to_datetime(cgm_data_csv['Date'] + ' ' + cgm_data_csv['Time'], format='%m/%d/%Y %H:%M:%S')
cgm_data_csv.head()
#cgm_data_csv_date = pd.DataFrame((cgm_data_csv['date'], '%m/%d/%Y %H:%M:%S'))
#pd.to_datetime(cgm_data_csv['Date'] + cgm_data_csv['Time'], format='%m-%d-%Y %H:%M:%S')

#if threshold_date<cgm_data_csv['date']
cgm_data_csv_auto = cgm_data_csv[(cgm_data_csv['date'] > threshold_date)]
cgm_data_csv_auto.head()

#cgm_data_csv_auto['data_night'] = pd.to_datetime(cgm_data_csv_auto['Time'], format='%H:%M:%S')
# cgm_data_csv_auto.loc[cgm_data_csv_auto['date'].time < 6 , 'night'] = 'True'
print(cgm_data_csv_auto.columns)
cgm_data_csv_auto = cgm_data_csv_auto.set_index(['date'])

cgm_data_csv_auto_night = cgm_data_csv_auto.between_time('0:0', '6:0')
cgm_data_csv_auto_night

cgm_data_csv_auto_morning = cgm_data_csv_auto.between_time('6:0', '0:0')
cgm_data_csv_auto_morning

auto_values = [0.0] * 18
auto_night_count =0
for data in cgm_data_csv_auto_night.iterrows():
    if (data[0] >= threshold_date):
        if (data[1][2] > 180):
                auto_values[0] += 1
        if (data[1][2] > 250):
                auto_values[1] += 1
        if (data[1][2] >= 70 and data[1][2] <= 180):
                auto_values[2] += 1
        if (data[1][2] >= 70 and data[1][2] <= 150):
                auto_values[3] += 1
        if (data[1][2] < 70):
                auto_values[4] += 1
        if (data[1][2] < 54):
                auto_values[5] += 1
        auto_night_count += 1

        auto_morning_count =0
for data in cgm_data_csv_auto_morning.iterrows():
    if (data[0] >= threshold_date):
        if (data[1][2] > 180):
                auto_values[6] += 1
        if (data[1][2] > 250):
                auto_values[7] += 1
        if (data[1][2] >= 70 and data[1][2] <= 180):
                auto_values[8] += 1
        if (data[1][2] >= 70 and data[1][2] <= 150):
                auto_values[9] += 1
        if (data[1][2] < 70):
                auto_values[10] += 1
        if (data[1][2] < 54):
                auto_values[11] += 1
        auto_morning_count += 1

        for data in cgm_data_csv_auto.iterrows():
        if (data[1][2] > 180):
            auto_values[12] += 1
        if (data[1][2] > 250):
            auto_values[13] += 1
        if (data[1][2] >= 70 and data[1][2] <= 180):
            auto_values[14] += 1
        if (data[1][2] >= 70 and data[1][2] <= 150):
            auto_values[15] += 1
        if (data[1][2] < 70):
            auto_values[16] += 1
        if (data[1][2] < 54):
            auto_values[17] += 1

cgm_data_csv_Manual = cgm_data_csv[(cgm_data_csv['date'] < threshold_date)]
cgm_data_csv_Manual.head()

print(cgm_data_csv_Manual.columns)
cgm_data_csv_Manual = cgm_data_csv_Manual.set_index(['date'])

cgm_data_csv_Manual_night = cgm_data_csv_Manual.between_time('0:0', '6:0')
cgm_data_csv_Manual_morning = cgm_data_csv_Manual.between_time('6:0', '0:0')



manual_values = [0.0] * 18
manual_night_count =0
for data in cgm_data_csv_Manual_night.iterrows():
    if (data[0] <= threshold_date):
        if (data[1][2] > 180):
                manual_values[0] += 1
        if (data[1][2] > 250):
                manual_values[1] += 1
        if (data[1][2] >= 70 and data[1][2] <= 180):
                manual_values[2] += 1
        if (data[1][2] >= 70 and data[1][2] <= 150):
                manual_values[3] += 1
        if (data[1][2] < 70):
                manual_values[4] += 1
        if (data[1][2] < 54):
                manual_values[5] += 1
        manual_night_count += 1
        
manual_morning_count =0
for data in cgm_data_csv_Manual_morning.iterrows():
    if (data[0] <= threshold_date):
        if (data[1][2] > 180):
                manual_values[6] += 1
        if (data[1][2] > 250):
                manual_values[7] += 1
        if (data[1][2] >= 70 and data[1][2] <= 180):
                manual_values[8] += 1
        if (data[1][2] >= 70 and data[1][2] <= 150):
                manual_values[9] += 1
        if (data[1][2] < 70):
                manual_values[10] += 1
        if (data[1][2] < 54):
                manual_values[11] += 1
        manual_morning_count += 1
        
for data in cgm_data_csv_Manual_morning.iterrows():
        if (data[1][2] > 180):
            manual_values[12] += 1
        if (data[1][2] > 250):
            manual_values[13] += 1
        if (data[1][2] >= 70 and data[1][2] <= 180):
            manual_values[14] += 1
        if (data[1][2] >= 70 and data[1][2] <= 150):
            manual_values[15] += 1
        if (data[1][2] < 70):
            manual_values[16] += 1
        if (data[1][2] < 54):
            manual_values[17] += 1

for i in range(0, 17):
    manual_values[i] = (((manual_values[i]) / (manual_night_count + manual_morning_count)) * 100)
    auto_values[i] = ((float(auto_values[i]) / (auto_night_count + auto_morning_count)) * 100)
results = pd.DataFrame([manual_values, auto_values])
results.fillna(0).to_csv('Results.csv', header=False, index=False)

