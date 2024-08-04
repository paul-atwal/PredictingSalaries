import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# load data
df = pd.read_csv('SalaryData.csv')

# drop rows without gender 
df = df.dropna(subset=['Gender'])

# find means/modes
meanAge = df['Age'].mean()
meanExp = df['Years of Experience'].mean()
meanSal = df['Salary'].mean()
modeEd = df['Education Level'].mode()
modeJob = df['Job Title'].mode()

# replace empty with means/modes 
df['Age'] = df['Age'].fillna(meanAge)
df['Years of Experience'] = df['Years of Experience'].fillna(meanExp)
df['Salary'] = df['Salary'].fillna(meanSal)
df['Education Level'] = df['Education Level'].fillna(modeEd)
df['Job Title'] = df['Job Title'].fillna(modeJob)

numerical_cols = ['Age', 'Years of Experience']
categorical_cols = ['Gender', 'Education Level', 'Job Title']

# apply MinMaxScaler to numerical columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(df[categorical_cols])

# convert encoded result into dataframe
encoded_df = pd.DataFrame(encoded_categorical)

# concatenate encoded result with original df 
df = df.drop(categorical_cols, axis=1)
df = pd.concat([df, encoded_df], axis=1)
df.columns = df.columns.astype(str)
df = df.dropna()

X = df.drop('Salary', axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

residuals = y_test - y_pred 

plt.hist(residuals)
plt.show()