import pandas as pd
import statsmodels.api as sm

csv_file_path = 'data_akbilgic_og.csv'  # Update this with the correct file path

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv(csv_file_path)

# Assuming 'dependent_variable' is your dependent variable column name
# and 'independent_variable_1' to 'independent_variable_7' are your independent variables
y = data['SP']
X = data[['ISE USD BASED', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU', 'EM']]

# Fit ARIMA model
model = sm.tsa.ARIMA(y, order=(p, d, q))  # Set appropriate values for p, d, and q
results = model.fit()

# Print model summary
print(results.summary())

# Plot residuals
results.plot_diagnostics()