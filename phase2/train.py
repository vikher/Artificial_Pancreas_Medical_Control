import pandas as pd
import numpy as np
import pickle
import statistics
from numpy import diff
from datetime import timedelta 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

def find_peaks_and_valleys(arr):
    n = len(arr)
    peaks = [0] if arr[0] > arr[1] else []
    valleys = [0] if arr[0] < arr[1] else []
    
    triples = zip(arr, arr[1:], arr[2:])
    peaks += [i for i, (prev, curr, next) in enumerate(triples, start=1) if prev < curr > next]
    valleys += [i for i, (prev, curr, next) in enumerate(triples, start=1) if prev > curr < next]
    
    if arr[-1] > arr[-2]:  
        peaks.append(n - 1)  
    elif arr[-1] < arr[-2]:  
        valleys.append(n - 1) 

    return peaks, valleys

def process_insulin_data(file_name):
    df = pd.read_csv(file_name)
    insulin_data = df.filter(['Index','Date','Time','BWZ Carb Input (grams)'])
    del df

    insulin_data['Date'] = insulin_data['Date'] + ' ' + insulin_data['Time']
    insulin_data['Date'] = pd.to_datetime(insulin_data['Date'])
    insulin_data.drop(['Time'], axis=1, inplace=True)
    insulin_data['BWZ Carb Input (grams)'].replace(0, np.nan, inplace=True)

    meal_times = insulin_data.loc[insulin_data['BWZ Carb Input (grams)'].notnull(), 'Date']
    pd.to_datetime(meal_times)

    meals = pd.Series()
    _ = 0
    for i in range((len(meal_times)-1), 0, -1):
        if (meal_times.iloc[i-1] - meal_times.iloc[i]) < timedelta(hours=2):
            pass
        else:
            meals.loc[_] = meal_times.iloc[i]
            _ += 1    
    meals.loc[_] = meal_times.iloc[0]
    pd.to_datetime(meals)

    return meals, meal_times

def get_data_for_meals(meals_list, cgm_data):
    lstm_data = []
    for meal_time in meals_list:
        filt = (cgm_data['Date'] > (meal_time - timedelta(hours=0.5))) & (cgm_data['Date'] <= (meal_time + timedelta(hours=2)))
        temp = cgm_data.loc[filt, 'Sensor Glucose (mg/dL)'].tolist()
        temp.reverse()
        lstm_data.append(temp)
    return lstm_data

def get_no_meal_times(mealtimes):
    nomeals = []
    for i in range(len(mealtimes)-1, 0, -1):
        if (mealtimes.iloc[i-1] - mealtimes.iloc[i]) > timedelta(hours=4):
            nomeals.append(mealtimes.iloc[i])
    nomeals.append(mealtimes.iloc[0])
    return pd.to_datetime(pd.Series(nomeals))

def process_data(no_meals, cgm_data):
    data_list = []
    no_meal_times = pd.Series()
    index = 0
    
    for i in range(len(no_meals) - 1):
        interval = (no_meals.iloc[i + 1] - no_meals.iloc[i] - timedelta(hours=2)) // timedelta(hours=2)
        for j in range(interval):
            start = 2 * (j + 1)
            end = start + 2
            no_meal_times.loc[index] = no_meals.iloc[i] + timedelta(hours=start)
            index += 1
            filter_condition = (cgm_data['Date'] > (no_meals.iloc[i] + timedelta(hours=start))) & (cgm_data['Date'] <= (no_meals.iloc[i] + timedelta(hours=end)))
            glucose_values = cgm_data.loc[filter_condition, 'Sensor Glucose (mg/dL)'].tolist()
            glucose_values.reverse()
            data_list.append(glucose_values)
    
    return data_list, no_meal_times

def compute_features(data_frame, method):
    feature_list = []
    for i in range(len(data_frame)):
        highest_value = max(data_frame.iloc[i].loc[data_frame.iloc[i].index > 6]) if method == 'meal' else max(data_frame.iloc[i])
        peak_index = data_frame.iloc[i].loc[data_frame.iloc[i] == highest_value].index[0] if method == 'meal' else data_frame.iloc[i].idxmax()
        time_to_peak = (peak_index - 6) * 5 if method == 'meal' else peak_index * 5
        glucose_change = (highest_value - data_frame.iloc[i].loc[6]) / data_frame.iloc[i].loc[6] if method == 'meal' else (highest_value - data_frame.iloc[i].iloc[0]) / data_frame.iloc[i].iloc[0]
        frequency_array = 20 * np.log10(np.abs(np.fft.fft(data_frame.iloc[i])))
        (maxima, minima) = find_peaks_and_valleys(frequency_array)
        first_peak_index = maxima[1]
        peak_frequency = frequency_array[first_peak_index]
        glucose_derivative = np.diff(data_frame.iloc[i]) / 1
        derivative_range = np.max(glucose_derivative) - np.min(glucose_derivative)
        second_derivative = np.diff(glucose_derivative) / 1
        second_derivative_range = np.max(second_derivative) - np.min(second_derivative)
        standard_deviation = statistics.stdev(data_frame.iloc[i])
        feature_set = np.array([highest_value, time_to_peak, glucose_change, derivative_range, second_derivative_range, standard_deviation, first_peak_index, peak_frequency], dtype='float32')
        feature_list.append(feature_set)
    return pd.DataFrame(feature_list)

def process_cgm_data(file_path):
    df = pd.read_csv(file_path)
    cgmD = df.filter(['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgmD['Date'] = pd.to_datetime(cgmD['Date'] + ' ' + cgmD['Time'])
    cgmD.drop(['Time'], axis=1, inplace=True)
    return cgmD

meals_1, meal_times_1 = process_insulin_data('InsulinData.csv')
meals_2, meal_times_2 = process_insulin_data('Insulin_patient2.csv')

meals = meals_1.append(meals_2, ignore_index=True)

cgm_data = process_cgm_data('CGMData.csv')
cgm_data_2 = process_cgm_data('CGM_patient2.csv')
cgm_data_2.drop(range(16478, 16490), inplace=True)

data_lstm = get_data_for_meals(meals_1, cgm_data) + get_data_for_meals(meals_2, cgm_data_2)
meal_data = pd.DataFrame(data_lstm, dtype='float32')
data_lengths_meals = [len(i) for i in data_lstm]

no_meals_1 = get_no_meal_times(meal_times_1)
no_meals_2 = get_no_meal_times(meal_times_2)

data_lst_no_meal_1, no_meal_times_1 = process_data(no_meals_1, cgm_data)
data_lst_no_meal_2, no_meal_times_2 = process_data(no_meals_2, cgm_data_2)

data_lst_no_meal = data_lst_no_meal_1 + data_lst_no_meal_2

no_meal_data = pd.DataFrame(data_lst_no_meal, dtype='float32')
no_meal_data.drop([i for i in range(24,37)], axis='columns', inplace=True)
data_lengths_no_meals = [len(i) for i in data_lst_no_meal]

no_meal_times = no_meal_times_1.append(no_meal_times_2, ignore_index=True)

threshold = 0.15

meal_nan_count = meal_data.isnull().sum(axis=1)
no_meal_nan_count = no_meal_data.isnull().sum(axis=1)

drop_indices_meal = meal_nan_count.loc[meal_nan_count > (threshold*30)].index
meal_data.drop(drop_indices_meal, inplace=True)
meals.drop(drop_indices_meal, inplace=True)

drop_indices_no_meal = no_meal_nan_count.loc[no_meal_nan_count > (threshold*24)].index
no_meal_data.drop(drop_indices_no_meal, inplace=True)
no_meal_times.drop(drop_indices_no_meal, inplace=True)

meal_data.fillna(method='ffill', inplace=True)
meal_data.fillna(method='bfill', inplace=True)
no_meal_data.fillna(method='ffill', inplace=True)
no_meal_data.fillna(method='bfill', inplace=True)
  
meal_features = compute_features(meal_data, 'meal')
meal_features_df = pd.DataFrame(meal_features, dtype='float32')

no_meal_features = compute_features(no_meal_data, 'no_meal')
no_meal_features_df = pd.DataFrame(no_meal_features, dtype='float32')

dataset = meal_features_df.append(no_meal_features_df, ignore_index=True)
labels = pd.Series(np.append(np.ones(818), np.zeros(2370)))

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.20)

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(X_train, y_train)

y_pred = GBC.predict(X_test)

pickle.dump(GBC, open('model.sav', 'wb'))