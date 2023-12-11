import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import probplot
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import shapiro, anderson
import numpy as np

# Replace 'your_file.xlsx' with the actual name of your Excel file
# Read the Excel file into a pandas DataFrame
file_path = 'data_akbilgic.xlsx'
df = pd.read_excel(file_path)
df['cumulative_sp'] = df['SP'].cumsum()
df['cumulative_dax'] = df['DAX'].cumsum()
df['cumulative_bovespa'] = df['BOVESPA'].cumsum()
df['cumulative_em'] = df['EM'].cumsum()
df['cumulative_ise'] = df['ISE USD BASED'].cumsum()
df['cumulative_ftse'] = df['FTSE'].cumsum()
df['cumulative_nikkei'] = df['NIKKEI'].cumsum()
df['cumulative_eu'] = df['EU'].cumsum()


# Now, 'df' contains your Excel data, and you can perform various operations on it
x = df[['cumulative_ise', 'cumulative_dax', 'cumulative_ftse', 'cumulative_nikkei', 'cumulative_eu']]
y = df['cumulative_sp'] 
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
  
print(model.summary())

# Residuals
residuals = y - predictions

# Q-Q plot
probplot(residuals, plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()

# Histogram of residuals with mean zero test
plt.hist(residuals, bins='auto', density=True, alpha=0.7, color='blue')
plt.axvline(x=0, color='red', linestyle='--', label='Mean')
plt.title('Histogram of Residuals with Mean Zero Test')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Anderson-Darling normality test
result = anderson(residuals)
print('Anderson-Darling test statistic: ', result.statistic)
print('Critical values: ', result.critical_values)
print('Significance levels: ', result.significance_level)


# Scatter plot
# plt.scatter(df['ISE USD BASED'], df['SP'], label='Data')

# # Polynomial fit
# degree = 1  # You can adjust the degree as needed
# coefficients = np.polyfit(df['ISE USD BASED'], df['SP'], degree)
# polynomial = np.poly1d(coefficients)
# x_fit = np.linspace(df['ISE USD BASED'].min(), df['ISE USD BASED'].max(), 100)
# y_fit = polynomial(x_fit)

# # Plot the polynomial fit
# plt.plot(x_fit, y_fit, color='red', label=f'Polynomial Fit (Degree {degree})')

# plt.xlabel('ISE USD BASED')
# plt.ylabel('SP')
# plt.legend()
# plt.show()

# plt.scatter(df['date'], df['SP'])
# plt.show()

# plt.scatter(df['date'], df['ISE USD BASED'])
# plt.show()

# Exponential smoothing.
# https://www.statsmodels.org/dev/examples/notebooks/generated/exponential_smoothing.html

# step-wise regression
# https://medium.com/@garrettwilliams90/stepwise-feature-selection-for-statsmodels-fda269442556
