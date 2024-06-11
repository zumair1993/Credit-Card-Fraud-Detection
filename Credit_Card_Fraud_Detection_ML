#Author Umair Zia

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Loading
data = pd.read_csv('creditcard.csv')

# Step 2: Data Preprocessing and Cleaning
# No preprocessing or cleaning needed as the dataset is already cleaned

# Step 3: Clustering Method for Cardholders
def cluster_cardholders(data):
    # Determine transaction amounts for clustering
    transaction_amounts = data['Amount'].values.reshape(-1, 1)
    
    # Perform clustering
    # For simplicity, let's use quantiles to divide cardholders into three clusters
    quantiles = [0, 0.5, 0.75, 1]
    clusters = np.quantile(transaction_amounts, quantiles)
    
    # Assign cluster labels
    data['Cluster'] = pd.cut(data['Amount'], bins=clusters, labels=['Low', 'Medium', 'High'])
    return data

# Step 4: Sliding Window Method for Aggregating Transactions
def sliding_window(transactions, window_size):
    aggregated_features = []
    labels = []
    
    for i in range(len(transactions) - window_size + 1):
        window = transactions.iloc[i:i + window_size]
        features = extract_features(window)
        label = window.iloc[-1]['Class']  # Last transaction label
        
        aggregated_features.append(features)
        labels.append(label)
    
    return np.array(aggregated_features), np.array(labels)

def extract_features(window):
    max_amt = window['Amount'].max()
    min_amt = window['Amount'].min()
    avg_amt = window['Amount'].mean()
    time_elapsed = (window.iloc[-1]['Time'] - window.iloc[0]['Time']) / 3600  # Convert seconds to hours
    return [max_amt, min_amt, avg_amt, time_elapsed]

# Step 5: Apply Clustering and Sliding Window Method
# Cluster cardholders
data = cluster_cardholders(data)

# Group transactions by cluster and apply sliding window
window_size = 10  # Define window size
aggregated_features = []
labels = []

for cluster, group in data.groupby('Cluster'):
    features, cluster_labels = sliding_window(group, window_size)
    aggregated_features.extend(features)
    labels.extend(cluster_labels)

aggregated_features = np.array(aggregated_features)
labels = np.array(labels)

# Step 6: Machine Learning Model Training
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(aggregated_features, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Model Evaluation
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Random Forest Classifier Performance:")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 8: Data Visualization
# Visualize the distribution of transaction amounts by cluster
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Amount', hue='Cluster', kde=True, bins=50)
plt.title('Distribution of Transaction Amount by Cluster')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.show()
