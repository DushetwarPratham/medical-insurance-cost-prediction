# Medical Insurance Cost Prediction

An end-to-end Machine Learning project for predicting medical insurance charges using demographic and lifestyle information. The project covers data understanding, exploratory data analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, model serialization, and deployment through Streamlit.

## Project Objective

The objective of this project is to build a regression model that can estimate medical insurance charges based on:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

The target variable is **`charges`**.

## Dataset

The project uses the Medical Insurance dataset containing the following columns:

| Feature | Description |
|---|---|
| `age` | Age of the insured person |
| `sex` | Gender |
| `bmi` | Body Mass Index |
| `children` | Number of dependent children |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance charges |

During data preparation, duplicate records were checked and removed. Missing values, distributions, relationships, and outliers were also examined before model development.

## Exploratory Data Analysis

EDA was performed to understand the major factors associated with medical insurance charges.

Key observations included:

- Smoking status has a strong relationship with insurance charges.
- Age shows a positive relationship with charges.
- BMI also contributes to variation in insurance costs.
- The number of children has a comparatively weaker relationship with charges.
- Some high insurance-charge observations were retained because they appeared to represent genuine values rather than data-entry errors.

## Data Preprocessing

The preprocessing workflow includes:

- Train-test split using an 80:20 ratio
- One-hot encoding of categorical variables using `pandas.get_dummies()`
- Alignment of train and test feature columns
- Standardization of numerical variables:
  - `age`
  - `bmi`
  - `children`
- Saving the fitted scaler and final feature-column structure for deployment

## Machine Learning Models

The following regression algorithms were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor
6. K-Nearest Neighbors Regressor
7. Support Vector Regressor

Models were evaluated using:

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² Score

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| RandomSearch XGBoost | 2444.813 | 4226.475 | 0.903 |
| GridSearch XGBoost | 2490.480 | 4239.826 | 0.902 |
| Original XGBoost | 2484.491 | 4243.800 | 0.902 |
| Gradient Boosting | 2517.468 | 4268.283 | 0.901 |
| Random Forest | 2644.928 | 4700.918 | 0.880 |
| Decision Tree | 2804.812 | 5912.109 | 0.810 |
| Linear Regression | 4177.046 | 5956.343 | 0.807 |
| KNN | 4496.305 | 8062.697 | 0.646 |
| SVR | 9261.821 | 14431.117 | -0.133 |

## Final Model

The final model selected for deployment is:

**RandomizedSearchCV-tuned XGBoost Regressor**

It achieved approximately:

- **MAE:** 2,444.81
- **RMSE:** 4,226.47
- **R²:** 0.903

RandomizedSearchCV produced the strongest overall performance among the evaluated models on the held-out test dataset.

## Hyperparameter Tuning

Both of the following techniques were evaluated:

- `GridSearchCV`
- `RandomizedSearchCV`

The search explored XGBoost parameters including:

- `n_estimators`
- `learning_rate`
- `max_depth`
- `subsample`
- `colsample_bytree`

The RandomizedSearchCV model was retained as the final model.

## Model Artifacts

The deployment workflow saves three Pickle files:

```text
model.pkl
scaler.pkl
columns.pkl
```

- `model.pkl` — trained RandomizedSearchCV XGBoost model
- `scaler.pkl` — fitted StandardScaler
- `columns.pkl` — ordered list of model input columns

These files must remain synchronized with the Streamlit application.

## Streamlit Deployment

The trained model is integrated into a Streamlit web application where users can enter their details and receive an estimated medical insurance charge.

### Run locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

If your Streamlit file has a different name, replace `app.py` with that filename.

## Suggested Project Structure

```text
Medical-Insurance-Cost-Prediction/
│
├── Insurance Cost Prediction.ipynb
├── insurance.csv
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Pickle
- Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <repository-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Conclusion

This project demonstrates a complete Machine Learning regression workflow, from understanding and preprocessing the data through model comparison, validation, tuning, and deployment.

The final RandomizedSearchCV-tuned XGBoost model achieved an R² score of approximately **0.903**, explaining about 90.3% of the variation in insurance charges on the held-out test dataset.

## Author

**Prathamesh Dushetwar**

Data Science & Healthcare Analytics
