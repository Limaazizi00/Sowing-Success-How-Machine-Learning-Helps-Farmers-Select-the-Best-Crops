import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Extract features and target
X = crops.drop("crop", axis=1)
y = crops["crop"]

# Store scores
feature_scores = {}

# Loop through each feature individually
for feature in X.columns:
    X_train, X_test, y_train, y_test = train_test_split(
        X[[feature]], y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train logistic regression on a single feature
    model = LogisticRegression(max_iter=500, multi_class="multinomial")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    feature_scores[feature] = acc

# Find the best predictive feature
best_feature = max(feature_scores, key=feature_scores.get)
best_predictive_feature = {best_feature: feature_scores[best_feature]}

best_predictive_feature
