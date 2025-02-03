# Customer Churn Prediction with Random Forest and XGBoost

This project aims to predict customer churn using machine learning techniques such as Random Forest and XGBoost. The goal is to classify whether a customer will churn (leave the company) based on historical data.

## Overview

Customer churn prediction is a critical task for businesses as it helps identify customers at risk of leaving. This project uses a dataset with various customer features, including demographics, account information, and service usage. It then trains machine learning models to predict whether the customer will churn or not.

### Models Used:
- **Random Forest Classifier**
- **XGBoost Classifier**

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1-Score

## Requirements

To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- matplotlib
- seaborn

You can install all dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
imbalanced-learn==0.8.1
xgboost==1.5.1
matplotlib==3.4.3
seaborn==0.11.2
```

## Dataset

This project uses a **customer churn dataset** that includes features such as customer demographics, account information, and service details. You can replace it with your own dataset, but ensure it contains information related to customer behavior.

If the dataset is too large, it's recommended to store it in an external location (Google Drive, Kaggle, etc.) and provide the download link.

## Project Structure

```bash
.
├── data/
│   ├── customer_churn.csv        # Your dataset (or a link to download it)
├── scripts/
│   ├── preprocess_data.py        # Script for data preprocessing
│   ├── train_models.py           # Script to train Random Forest and XGBoost models
│   └── evaluate_models.py        # Script for evaluating model performance
├── notebooks/
│   └── churn_analysis.ipynb      # Jupyter Notebook with exploratory data analysis
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Usage

### Step 1: Preprocess Data

Run the data preprocessing script to clean and prepare the data for model training.

```bash
python scripts/preprocess_data.py
```

### Step 2: Train Models

To train both the Random Forest and XGBoost models, run the following script:

```bash
python scripts/train_models.py
```

### Step 3: Evaluate Models

Evaluate the performance of both models using metrics like accuracy, precision, recall, and F1-score:

```bash
python scripts/evaluate_models.py
```

This will print the performance of each model to the console.

## Results

**Random Forest**:
- Accuracy: 0.78
- Precision: 0.58
- Recall: 0.60
- F1-score: 0.59

**XGBoost**:
- Accuracy: 0.79
- Precision: 0.60
- Recall: 0.57
- F1-score: 0.59

Both models show good performance with similar F1-scores. Further hyperparameter tuning can be performed to improve these metrics.

## Next Steps

1. **Model Tuning**: Experiment with hyperparameter optimization to improve the model's performance.
2. **Deploying the Model**: You can deploy the final trained model as a REST API using frameworks like Flask or FastAPI.
3. **Handle Imbalanced Data**: Explore other techniques for handling class imbalance, such as adjusting class weights or experimenting with different sampling strategies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
