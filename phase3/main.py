import pandas as pd
import numpy as np
import statistics
from datetime import timedelta
from numpy import diff
from sklearn.cluster import KMeans
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import contingency_matrix
from scipy.signal import find_peaks
from statistics import stdev

def load_data(file_name, usecols):
    data = (pd.read_csv(file_name, usecols=usecols)
            .assign(DateTime=lambda df: pd.to_datetime(df['Date'] + ' ' + df['Time']))
            .set_index('DateTime')
            .drop(['Date', 'Time'], axis=1))
    return data

def calculate_entropy(contingency, cluster_totals, total):
    cluster_entropy = [-sum((j / cluster_totals[i]) * np.log2(j / cluster_totals[i]) for j in contingency.iloc[i] if j != 0) for i in range(contingency.shape[0])]
    entropy = sum((entropy / total) * np.log2(entropy / total) for entropy in cluster_entropy if entropy != 0)
    return -entropy

def calculate_purity(contingency, total):
    return np.sum([contingency.iloc[i].max() for i in range(contingency.shape[0])]) / total

continuous_glucose_data = load_data('CGMData.csv', ['Date', 'Time', 'Sensor Glucose (mg/dL)'])
insulin_data = load_data('InsulinData.csv', ['Date', 'Time', 'BWZ Carb Input (grams)'])

insulin_data['BWZ Carb Input (grams)'].replace(0, np.nan, inplace=True)

meal_times = insulin_data.loc[insulin_data['BWZ Carb Input (grams)'].notnull()].index
meal_times = pd.to_datetime(meal_times)

meal_times_list = [meal_times[i] for i in range(len(meal_times) - 1, 0, -1) if (meal_times[i - 1] - meal_times[i]) >= timedelta(hours=2)]
meal_times_list.append(meal_times[0])
meal_times_index = pd.to_datetime(pd.Series(meal_times_list))

meal_data_list = []
for i in range(len(meal_times_index)):
    filter_condition = (continuous_glucose_data.index > (meal_times_index[i] - timedelta(hours=0.5))) & (
                continuous_glucose_data.index <= (meal_times_index[i] + timedelta(hours=2)))
    temp_data = continuous_glucose_data.loc[filter_condition, 'Sensor Glucose (mg/dL)'].tolist()
    temp_data.reverse()
    meal_data_list.append(temp_data)
meal_data = pd.DataFrame(meal_data_list, dtype='float32')
data_lengths = [len(data) for data in meal_data_list]

nan_threshold = 0.15
meal_nan_counts = meal_data.isnull().sum(axis=1)
drop_indices = meal_nan_counts.loc[meal_nan_counts > (nan_threshold * 30)].index
meal_data.drop(drop_indices, inplace=True)
meal_times_index.drop(drop_indices, inplace=True)

for i in range(len(meal_data)):
    meal_data.iloc[i].fillna(method='ffill', inplace=True)
    meal_data.iloc[i].fillna(method='bfill', inplace=True)

meal_carbohydrates = insulin_data.loc[meal_times_index]
meal_carbohydrates.dropna(inplace=True)
meal_carbohydrates.columns = ['Carbohydrates']

meal_features = []
for i in range(len(meal_data)):
    meal_row = meal_data.iloc[i]
    meal_carb_value = meal_carbohydrates.iloc[i].values[0]
    
    highest_glucose = meal_row.iloc[meal_row.index > 6].max()
    peak_glucose_index = meal_row.idxmax()
    time_to_peak = (peak_glucose_index - 6) * 5
    delta_glucose_normalized = (highest_glucose - meal_row[6]) / meal_row[6]
    delta_glucose = highest_glucose - meal_row[6]
    
    fft_array = 20 * np.log10(np.abs(np.fft.fft(meal_row)))
    peaks, _ = find_peaks(fft_array)
    fft_peak_1 = peaks[1] if len(peaks) > 1 else 0
    peak_fft_1 = fft_array[fft_peak_1] if len(peaks) > 1 else 0
    
    derivatives = np.diff(meal_row) / 1
    derivative_1 = np.max(derivatives) - np.min(derivatives)
    second_derivatives = np.diff(derivatives) / 1
    derivative_2 = np.max(second_derivatives) - np.min(second_derivatives)
    
    standard_deviation = stdev(meal_row)
    
    feature_set = np.array([highest_glucose, delta_glucose_normalized, derivative_1, meal_carb_value], dtype='float32')
    meal_features.append(feature_set)

meal_features_df = pd.DataFrame(meal_features, columns=['Highest_Glucose', 'Delta_Glucose_Normalized', 'Derivative_1', 'Carbohydrates'], dtype='float32')

bin_numbers = pd.Series()
index_counter = 0
for i in meal_carbohydrates.index:
    bin_num = 1 + (meal_carbohydrates.loc[i, 'Carbohydrates'] - min(meal_carbohydrates['Carbohydrates'])) // 20
    bin_numbers.loc[index_counter] = bin_num
    index_counter += 1
bin_numbers = bin_numbers.astype('int_')

data_array = meal_features_df.values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_array)

k_means = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300)
k_means_labels = k_means.fit_predict(scaled_data)
k_means_sse = k_means.inertia_

dbscan = DBSCAN(eps=.38, min_samples=6)
db_labels = dbscan.fit_predict(scaled_data)
db_centroids = np.array([scaled_data[db_labels == i].mean(axis=0) for i in range(-1, 5)])
db_cluster_sse = [np.sum(np.square(scaled_data[db_labels == i] - db_centroids[i])) for i in range(-1, 5)]
db_sse = np.sum(db_cluster_sse)

k_means_contingency = pd.DataFrame(contingency_matrix(k_means_labels, bin_numbers))
db_contingency = pd.DataFrame(contingency_matrix(db_labels, bin_numbers))
db_contingency.drop(0, axis=0, inplace=True)

k_means_cluster_totals = [k_means_contingency.iloc[i].sum() for i in range(k_means_contingency.shape[0])]
db_cluster_totals = [db_contingency.iloc[i].sum() for i in range(db_contingency.shape[0])]
k_means_total = np.sum(k_means_cluster_totals)
db_total = np.sum(db_cluster_totals)

k_means_entropy = calculate_entropy(k_means_contingency, k_means_cluster_totals, k_means_total)
db_entropy = calculate_entropy(db_contingency, db_cluster_totals, db_total)

k_means_purity = calculate_purity(k_means_contingency, k_means_total)
db_purity = calculate_purity(db_contingency, db_total)

results_df = pd.DataFrame([[k_means_sse, db_sse, k_means_entropy, db_entropy, k_means_purity, db_purity]])
results_df.to_csv('Result.csv', index=False, header=False)