# =========================
# IMPORT LIBRARIES
# =========================
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("car_data.csv")

print("Columns:", data.columns)

# =========================
# DATA CLEANING
# =========================

# Fix Levy column
data['Levy'] = data['Levy'].replace('-', np.nan)
data['Levy'] = pd.to_numeric(data['Levy'])

# Drop missing values
data = data.dropna()

# =========================
# FEATURE SELECTION
# =========================

# Numerical features
num_features = ['Prod. year', 'Levy', 'Cylinders', 'Airbags']

# Categorical features (VERY IMPORTANT)
cat_features = ['Manufacturer', 'Category', 'Fuel type', 'Gear box type', 'Drive wheels']

# Combine
X = data[num_features + cat_features]

# Target
y = data['Price']

# =========================
# ENCODE CATEGORICAL DATA
# =========================

# Convert categorical to dummy variables
X = pd.get_dummies(X, drop_first=True)

# =========================
# TRAIN MODEL
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# MODEL PERFORMANCE
# =========================

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Improved R2:", r2)
print("Improved RMSE:", rmse)

# =========================
# PLOT (use Prod. year)
# =========================

plt.figure()
plt.scatter(data['Prod. year'], data['Price'])

# Sort index
sorted_index = data['Prod. year'].argsort()

plt.plot(
    data['Prod. year'].iloc[sorted_index],
    model.predict(X)[sorted_index],
    color='red'
)

plt.xlabel("Production Year")
plt.ylabel("Price")
plt.title("Improved Car Price Regression")

plt.savefig("static/plot.png")
plt.close()

# =========================
# FLASK ROUTE
# =========================

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None

    if request.method == 'POST':
        try:
            # Get numeric inputs
            year = float(request.form['year'])
            levy = float(request.form['levy'])
            cylinders = float(request.form['cylinders'])
            airbags = float(request.form['airbags'])

            # Default category values (simple approach)
            input_dict = {
                'Prod. year': year,
                'Levy': levy,
                'Cylinders': cylinders,
                'Airbags': airbags
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])

            # Add missing dummy columns
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            # Predict
            prediction = model.predict(input_df)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        'index.html',
        prediction=prediction,
        r2=round(r2, 3),
        rmse=round(rmse, 2)
    )

# =========================
# RUN APP
# =========================

if __name__ == '__main__':
    app.run(debug=True)