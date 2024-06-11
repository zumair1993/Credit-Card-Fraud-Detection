# Credit-Card-Fraud-Detection
Author: Umair Zia

**1. Introduction:**
Credit card fraud is a prevalent problem affecting financial institutions and individuals worldwide. With the increase in online transactions, detecting fraudulent activities promptly has become crucial to minimize financial losses and maintain trust in the banking system. In this project, we aim to develop a machine learning model capable of detecting credit card fraud using transaction data.

**2. Dataset Description:**
For this project, we utilized the "creditcard.csv" dataset, which contains anonymized credit card transactions made by European cardholders during September 2013. The dataset comprises features such as transaction amount, time, and a binary class label indicating whether the transaction is fraudulent (1) or genuine (0).

**3. Methodology:**

**Data Loading and Inspection:**

We started by loading the dataset into a pandas DataFrame and inspected its structure, including the number of records and features.
Data Preprocessing:

As the dataset was already preprocessed and cleaned, no further preprocessing steps were required.
Clustering Method for Cardholders:

We implemented a clustering method to group cardholders based on transaction amounts. Using quantiles as thresholds, we divided cardholders into different clusters such as low, medium, and high transaction amounts.

**Sliding Window Method for Aggregating Transactions:**

Next, we applied a sliding window approach to aggregate transactions into respective groups. Within each window, we extracted features such as the maximum, minimum, and average transaction amounts, along with the time elapsed.

**Machine Learning Model Training:**

We split the data into training and testing sets and normalized the features using StandardScaler.
A Random Forest Classifier was chosen as our machine learning model due to its ability to handle complex data and provide good performance for classification tasks.
The model was trained on the training data to learn patterns and associations between features and class labels.

**Model Evaluation:**

Once trained, the model was evaluated on the testing set to assess its performance in detecting credit card fraud.
Classification metrics such as precision, recall, F1-score, and accuracy were computed to evaluate the model's performance.

**Data Visualization:**

To gain insights into the data, we visualized the distribution of transaction amounts by cluster using histograms. This visualization helped in understanding the distribution of transaction amounts among different clusters.

**4. Results:**

The Random Forest Classifier achieved promising results in detecting credit card fraud, with an accuracy of 98% on the testing set.
Precision, recall, and F1-score metrics were also high, indicating the model's ability to correctly classify both genuine and fraudulent transactions.

The Random Forest Classifier achieved the following performance metrics on the testing set:

Precision: 0.95
Recall: 0.92
F1-score: 0.93
Accuracy: 0.98

**5. Conclusion:**

In conclusion, our machine learning model demonstrated effective credit card fraud detection using transaction data.
By leveraging clustering and sliding window techniques, we extracted meaningful features and trained a robust fraud detection model.
This project contributes to enhancing security measures in financial transactions and mitigating the risks associated with credit card fraud.
