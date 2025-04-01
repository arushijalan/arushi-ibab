## Implement decision tree classifier without using scikit-learn using the breast cancer dataset.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
df = pd.read_csv("breast-cancer.csv")

# Display basic dataset information
print(df.head())  # Print first few rows
print(df.columns)  # Print column names
print(df.info())  # Print dataset information (datatypes, null values, etc.)
print(df.shape)  # Print dataset dimensions

# Split dataset into features (X) and target variable (y)
X = df.iloc[:, :-1].astype(str)  # Selecting all columns except the last one as features
y = df.iloc[:, -1].astype(str)  # Selecting the last column as the target variable

print(X.shape)  # Print feature matrix shape
print(y.shape)  # Print target variable shape

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Encode categorical input variables using Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train)
X_train = ordinal_encoder.transform(X_train)
X_test = ordinal_encoder.transform(X_test)

print(X_train[:5, :])  # Print first 5 rows of transformed feature matrix

# Encode target variable using Label Encoding
label_encoder = LabelEncoder()
label_encoder.fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# Define and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy * 100))

