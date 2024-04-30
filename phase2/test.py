import pandas as pd
import numpy as np
import pickle
import statistics

def find_local_extremas(array):
    local_maxima_indices = []
    local_minima_indices = []
    n = len(array)

    if array[0] > array[1]:
        local_maxima_indices.append(0)
    elif array[0] < array[1]:
        local_minima_indices.append(0)

    for i in range(1, n - 1):
        prev, curr, next = array[i - 1], array[i], array[i + 1]
        if prev > curr < next:
            local_minima_indices.append(i)
        elif prev < curr > next:
            local_maxima_indices.append(i)

    if array[-1] > array[-2]:
        local_maxima_indices.append(n - 1)
    elif array[-1] < array[-2]:
        local_minima_indices.append(n - 1)

    return (local_maxima_indices, local_minima_indices)

try:
    data = pd.read_csv('test.csv', header=None)
except FileNotFoundError:
    print("Error: File 'test.csv' not found.")
    exit()

def compute_features(row):
    high_value = row.max()
    peak_index = row.idxmax()
    time = peak_index * 5

    range_val = high_value - row.min()

    dGn_val = (high_value - row[0]) / row[0]

    fft_arr = 20 * np.log10(np.abs(np.fft.fft(row)))
    (maxima, minima) = find_local_extremas(fft_arr)

    f1_value = maxima[1] if maxima else np.nan
    pf1_value = fft_arr[f1_value] if f1_value is not np.nan else np.nan

    d = np.diff(row) / 1
    d1_value = d.max() - d.min()
    dd = np.diff(d) / 1
    d2_value = dd.max() - dd.min()

    std_dev = statistics.stdev(row)

    return [high_value, range_val, time, dGn_val, d1_value, d2_value, std_dev, f1_value]

features = data.apply(compute_features, axis=1)
feature_df = pd.DataFrame(features.tolist())

try:
    with open('model.sav', 'rb') as model_file:
        classifier_model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: Model file 'model.sav' not found.")
    exit()
except Exception as e:
    print("Error loading model:", e)
    exit()

try:
    predictions = classifier_model.predict(feature_df)
except Exception as e:
    print("Error during prediction:", e)
    exit()

try:
    result_df = pd.DataFrame(predictions)
    result_df.to_csv('Result.csv', index=False, header=False)
except Exception as e:
    print("Error writing result to file:", e)
    exit()
