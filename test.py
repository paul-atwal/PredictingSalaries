import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv('SalaryData.csv')

# Drop rows without gender 
df = df.dropna(subset=['Gender'])

# Find means/modes
meanAge = df['Age'].mean()
meanExp = df['Years of Experience'].mean()
meanSal = df['Salary'].mean()
modeEd = df['Education Level'].mode()[0]
modeJob = df['Job Title'].mode()[0]

# Replace empty with means/modes 
df['Age'] = df['Age'].fillna(meanAge)
df['Years of Experience'] = df['Years of Experience'].fillna(meanExp)
df['Salary'] = df['Salary'].fillna(meanSal)
df['Education Level'] = df['Education Level'].fillna(modeEd)
df['Job Title'] = df['Job Title'].fillna(modeJob)

numerical_cols = ['Age', 'Years of Experience']
categorical_cols = ['Gender', 'Education Level', 'Job Title']

# Apply MinMaxScaler to numerical columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Encode categorical variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(df[categorical_cols])

# Convert encoded result into dataframe with proper column names
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))

# Concatenate encoded result with original dataframe
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop any remaining missing values
df = df.dropna()

# Split the data into features and target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(len(y_test), len(y_pred))

# Plot residuals
residuals = y_test - y_pred 
plt.scatter(y_pred, residuals, edgecolors='k')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()
