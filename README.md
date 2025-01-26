# Proxy-Machine-Learning-Models-for-Explainability

## Overview
This project explores the use of proxy models as interpretable stand-ins for black-box machine learning models. By leveraging tools such as SHAP and LIME, we aim to improve the explainability of complex models. This is especially useful for domains like healthcare, where decision transparency is critical.

## Features
- **Proxy Models**: Simplified models to interpret predictions.
- **Explainability**: Tools such as SHAP and LIME for understanding feature importance.
- **Data Analysis**: Comprehensive exploratory data analysis (EDA) and preprocessing.
- **Use Case**: Applied on a healthcare dataset (`meningitis_dataset.csv`) to predict outcomes.

## Installation
Ensure you have Python 3.7+ installed. Use the following commands to set up the environment:

```bash
pip install shap lime scikit-learn pandas numpy matplotlib seaborn
```

## Dataset
The dataset contains anonymized information related to meningitis cases. Key steps:
1. Load the dataset from a CSV file stored in Google Drive.
2. Perform EDA to identify missing values and data types.
3. Preprocess the data by renaming columns and removing personal identifiers.

## Code Highlights

### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lime
import lime.lime_tabular
```

### Loading the Dataset
```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/meningitis_dataset.csv')
```

### Exploratory Data Analysis
```python
# Check for missing values
df.isna().sum().sum()

# Dataset info
df.info()
```

### Data Preprocessing
```python
# Drop unnecessary columns
df_cdat = df.drop(['surname', 'firstname', 'middlename', 'date_of_birth'], axis=1)

# Rename columns
df_cdat.rename(columns={'report_date': 'date', 'health_status': 'status', 'report_outcome': 'outcome'}, inplace=True)
```

## Usage
1. Clone the repository and install the required libraries.
2. Load your dataset and follow the preprocessing steps provided.
3. Use LIME and SHAP for model explainability:

### Example with LIME
```python
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=df.values, 
    feature_names=df.columns,
    class_names=['Outcome1', 'Outcome2'], 
    mode='classification'
)
```

## Results
This project demonstrates how proxy models and explainability techniques can make machine learning models more interpretable, especially in critical applications like healthcare.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request.

## License
This project is licensed under the MIT License.

