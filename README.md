# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using advanced classification algorithms and data analysis techniques.

## 🎯 Objective

Develop an accurate and efficient model to identify fraudulent credit card transactions while minimizing false positives to ensure legitimate transactions are not blocked.

## 🚀 Features

- **Data Preprocessing**: Handle missing values, outliers, and feature scaling
- **Exploratory Data Analysis**: Comprehensive analysis of transaction patterns
- **Feature Engineering**: Create meaningful features for better model performance
- **Multiple ML Algorithms**: Implementation of various classification models
- **Model Evaluation**: Detailed performance metrics and comparison
- **Visualization**: Interactive plots and charts for data insights

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Jupyter Notebook** - Interactive development environment

## 📊 Dataset

The project uses a credit card transactions dataset containing:
- **284,807 transactions** over 2 days
- **30 features** (V1-V28 are PCA components, Time, Amount)
- **Highly imbalanced dataset** (0.172% fraudulent transactions)

## 🧠 Machine Learning Models

### Algorithms Implemented:
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **Decision Tree Classifier**
5. **Gradient Boosting Classifier**

### Performance Metrics:
- Accuracy Score
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lakshayy1509/Credit-card-fraud-detection.git
   cd Credit-card-fraud-detection
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Download the dataset**
   - Download the credit card dataset (usually from Kaggle)
   - Place it in the project directory

4. **Run the notebook**
   ```bash
   jupyter notebook
   ```

## 📈 Results

### Model Performance Summary:
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 99.95% | 0.88 | 0.81 | 0.84 | 0.94 |
| Logistic Regression | 99.92% | 0.85 | 0.78 | 0.81 | 0.92 |
| SVM | 99.93% | 0.86 | 0.79 | 0.82 | 0.93 |

*Note: Update these with your actual results*

## 🔍 Key Insights

- **Transaction Amount**: Higher amounts show different fraud patterns
- **Time Patterns**: Fraudulent transactions have specific timing characteristics
- **Feature Importance**: V14, V4, V11 are among the most important features
- **Class Imbalance**: Addressed using SMOTE and stratified sampling

## 📱 Usage

```python
# Load the trained model
import pickle
model = pickle.load(open('fraud_detection_model.pkl', 'rb'))

# Make predictions
prediction = model.predict(new_transaction_data)
probability = model.predict_proba(new_transaction_data)
```

## 📊 Visualizations

The project includes:
- Distribution plots for transaction amounts
- Correlation heatmaps
- ROC curves for model comparison
- Feature importance plots
- Confusion matrices

## 🎯 Future Improvements

- [ ] Real-time fraud detection system
- [ ] Deep learning models (Neural Networks)
- [ ] Ensemble methods for better accuracy
- [ ] Feature selection optimization
- [ ] Deployment using Flask/FastAPI
- [ ] Integration with banking systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/ModelImprovement`)
3. Commit your changes (`git commit -m 'Add new ML algorithm'`)
4. Push to the branch (`git push origin feature/ModelImprovement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Lakshay**
- GitHub: [@lakshayy1509](https://github.com/lakshayy1509)
## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Scikit-learn community for excellent ML tools
- Research papers on fraud detection methodologies

## 📚 References

- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Machine Learning for Fraud Detection Research Papers
- Scikit-learn Documentation

⭐ If this project helped you, please give it a star!
