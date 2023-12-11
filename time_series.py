import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create a DataFrame
# data = pd.DataFrame({
#     'Year': np.repeat(range(1, 5), 4),
#     'Quarter': np.tile(range(1, 5), 4),
#     'yt': [10, 31, 43, 16, 11, 33, 45, 17, 13, 34, 48, 19, 15, 37, 51, 21],
#     't': range(1, 17)
# })
# #print(data.t)

csv_file_path = 'data_akbilgic_og.csv'  # Update this with the correct file path

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_file_path)

print(data.head())

data['cumulative_sp'] = data['SP'].cumsum()
# data.rename(columns={'cumulative_sp': 'yt'}, inplace=True)
data.rename(columns={'SP': 'yt'}, inplace=True)
data['t'] = data.index


# # Create seasonal indicators for each quarter
# data['Qtr2'] = (data['Quarter'] == 2).astype(int)
# data['Qtr3'] = (data['Quarter'] == 3).astype(int)
# data['Qtr4'] = (data['Quarter'] == 4).astype(int)

# Create a time series
y = pd.Series(data['yt'])

# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(data['t'], y, marker='o', linestyle='-')
plt.xlabel('Time (t)')
plt.ylabel('yt')
plt.title('Time Series Plot')
plt.grid(True)
plt.show()

# Fit a linear trend model with seasonal indicators
X = sm.add_constant(data[['t', 'Qtr2', 'Qtr3', 'Qtr4']])
model = sm.OLS(y, X).fit()

# Summary of the model
print(model.summary())

print(data)

#Year 5 Forecasts not yet accurate
# Quarter 1
new_data = pd.DataFrame({'t': [17], 'Qtr2': [0], 'Qtr3': [0], 'Qtr4': [0], 'const': [1]})  # Add a constant term
prediction = model.get_prediction(new_data)
print("Quarter 1 Prediction:")
print(prediction.summary_frame(alpha=0.05))

# Quarter 1
new_data = pd.DataFrame({'t': [17], 'Qtr2': [0], 'Qtr3': [0], 'Qtr4': [0], 'const': [1]})  # Add a constant term
quarter1_coef = model.params['Qtr2']
quarter1_prediction = model.predict(new_data) + quarter1_coef
print("Quarter 1 Prediction:", quarter1_prediction)

# Quarter 2
new_data = pd.DataFrame({'t': [18], 'Qtr2': [1], 'Qtr3': [0], 'Qtr4': [0], 'const': [1]})  # Add a constant term
quarter2_coef = model.params['Qtr2']
quarter2_prediction = model.predict(new_data) + quarter2_coef
print("Quarter 2 Prediction:", quarter2_prediction)

# Quarter 3
new_data = pd.DataFrame({'t': [19], 'Qtr2': [0], 'Qtr3': [1], 'Qtr4': [0], 'const': [1]})  # Add a constant term
quarter3_coef = model.params['Qtr3']
quarter3_prediction = model.predict(new_data) + quarter3_coef
print("Quarter 3 Prediction:", quarter3_prediction)

# Quarter 4
new_data = pd.DataFrame({'t': [20], 'Qtr2': [0], 'Qtr3': [0], 'Qtr4': [1], 'const': [1]})  # Add a constant term
quarter4_coef = model.params['Qtr4']
quarter4_prediction = model.predict(new_data) + quarter4_coef
print("Quarter 4 Prediction:", quarter4_prediction)