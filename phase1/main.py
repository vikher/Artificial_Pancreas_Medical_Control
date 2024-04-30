import pandas as pd

def read_and_process_data(input_file_name, selected_columns):
    try:
        data_frame = pd.read_csv(input_file_name)
    except FileNotFoundError:
        print("File not found. Please provide a valid file name.")
        return None
    except pd.errors.EmptyDataError:
        print("The provided CSV file is empty.")
        return None

    if not set(selected_columns).issubset(data_frame.columns):
        print("Selected columns not found in the data.")
        return None

    selected_data = data_frame[selected_columns].copy()
    if 'Date' in selected_data.columns and 'Time' in selected_data.columns:
        selected_data['Datetime'] = pd.to_datetime(selected_data['Date'] + ' ' + selected_data['Time'])
        selected_data['Date'] = pd.to_datetime(selected_data['Date'])
        selected_data['Time'] = pd.to_datetime(selected_data['Time'])
    return selected_data

def remove_glucose_outliers(df):
    try:
        null_values = df['Sensor Glucose (mg/dL)'].notnull().resample('D').sum()
        index_list = null_values[null_values < 231].index.strftime('%Y-%m-%d').tolist()
        for i in index_list:
            df.drop(df[i].dropna().index, inplace=True)
    except KeyError:
        print("Column 'Sensor Glucose (mg/dL)' not found.")
        return None

def generate_metrics(data):
    def calculate_metrics(data, glucose_range):
        return len(data[(data['Sensor Glucose (mg/dL)'] >= glucose_range[0]) & (data['Sensor Glucose (mg/dL)'] < glucose_range[1])])/2.88

    try:
        mean_glucose = data['Sensor Glucose (mg/dL)'].resample('D').mean()
        index_list = mean_glucose.index
    except KeyError:
        print("Column 'Sensor Glucose (mg/dL)' not found.")
        return None

    metrics_data = []

    for index in index_list:
        date = str(index)[:10]
        date_day6am = pd.Timestamp(date + ' 06:00:00')
        date_day12am = pd.Timestamp(date + ' 23:59:59')
        date_night12am = pd.Timestamp(date + ' 00:00:00')
        
        whole_day_data = data[data['Date'] == index].copy()
        whole_day_data.fillna(mean_glucose[index], inplace=True)

        day_filt = (whole_day_data['Time'] >= date_day6am) & (whole_day_data['Time'] <= date_day12am)
        night_filt = (whole_day_data['Time'] >= date_night12am) & (whole_day_data['Time'] < date_day6am)

        day_data = whole_day_data[day_filt].copy()
        night_data = whole_day_data[night_filt].copy()

        glucose_ranges = [(180, float('inf')), (250, 180), (70, 180), (70, 150), (0, 70), (0, 54)]
        metrics_data.append([calculate_metrics(time_data, glucose_range) for time_data in [night_data, day_data, whole_day_data] for glucose_range in glucose_ranges])

    return pd.DataFrame(metrics_data).transpose()
     
# Descriptive variable names
cgm_columns = ['Index', 'Date', 'Time', 'Sensor Glucose (mg/dL)']
idata_columns = ['Index', 'Date', 'Time', 'Alarm']

# Reading and processing data
cgm_data = read_and_process_data('CGMData.csv', cgm_columns)
if cgm_data is None:
    exit()

idata_data = read_and_process_data('InsulinData.csv', idata_columns)
if idata_data is None:
    exit()

# Filtering data for auto mode active with PLGM off and getting the last date
try:
    last_auto_mode_date = idata_data[idata_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[-1, -1]
except IndexError:
    print("No record found for 'AUTO MODE ACTIVE PLGM OFF'.")
    exit()

del idata_data

# Filtering CGM data for auto and manual modes
auto_mode_filt_cgm = cgm_data['Datetime'] > last_auto_mode_date
cgm_auto_mode = cgm_data[auto_mode_filt_cgm].set_index('Datetime')

manual_mode_filt_cgm = cgm_data['Datetime'] <= last_auto_mode_date
cgm_manual_mode = cgm_data[manual_mode_filt_cgm].set_index('Datetime')

del cgm_data

# Removing glucose outliers
remove_glucose_outliers(cgm_auto_mode)
remove_glucose_outliers(cgm_manual_mode)

# Generating metrics
auto_mode_metrics = generate_metrics(cgm_auto_mode)
if auto_mode_metrics is None:
    exit()

manual_mode_metrics = generate_metrics(cgm_manual_mode)
if manual_mode_metrics is None:
    exit()

# Calculating mean metrics
metrics = [list(manual_mode_metrics.mean(axis=1))]
metrics.append(list(auto_mode_metrics.mean(axis=1)))

# Creating output DataFrame and saving to CSV
output_data = pd.DataFrame(metrics)
output_data.to_csv('Result.csv', index=False, header=False)