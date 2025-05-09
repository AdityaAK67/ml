# Assignment 1: PDA
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# df = pd.read_csv('iris.csv')

# # 1. View the first 5 rows of the dataset
# print("First 5 rows:")
# print(df.head())

# # 2. Check for missing values in the dataset
# print("\nMissing values:")
# print(df.isnull().sum())

# # 3. Summary statistics for numerical columns
# print("\nSummary statistics:")
# print(df.describe())

# # 4. Check the data types of the columns
# print("\nData types:")
# print(df.dtypes)

# # 5. Value counts for the target column (species)
# print("\nValue counts for species:")
# print(df['species'].value_counts())

# # 6. Correlation matrix (only for numerical columns)
# # Exclude the 'species' column before calculating correlation
# print("\nCorrelation matrix:")
# print(df.drop(columns='species').corr())

# # 7. Distribution of each numerical column using histograms
# df.hist(bins=20, figsize=(10, 8))
# plt.suptitle('Histograms for Numerical Columns')
# plt.show()

# # 8. Pairplot to visualize relationships between features
# sns.pairplot(df, hue='species')
# plt.suptitle('Pairplot', y=1.02)
# plt.show()

# # 9. Boxplot for numerical columns grouped by species
# sns.boxplot(data=df, x='species', y='sepal_length')
# plt.title('Boxplot: Sepal Length by Species')
# plt.show()

# # 10. Heatmap of correlation matrix
# sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()


# Assignment 2 Linear Regression
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load local CSV file
# df = pd.read_csv('hw_200.csv')

# # Rename columns if needed
# df.columns = ['Index', 'Height', 'Weight']

# # Drop the 'Index' column
# df = df.drop('Index', axis=1)

# # Features and target
# X = df[['Height']]  # Independent variable
# y = df['Weight']    # Dependent variable

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation metrics
# print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
# print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
# print("R² Score:", r2_score(y_test, y_pred))

# Assignment 3 Logistic Regression
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# # Load dataset
# df = pd.read_csv('Social_Network_Ads.csv')

# # View columns and keep necessary ones
# print(df.columns)
# # Use 'Age' and 'EstimatedSalary' to predict 'Purchased'
# X = df[['Age', 'EstimatedSalary']]
# y = df['Purchased']

# # Split into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train logistic regression
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Assignment 4 Decision Tree Classifier

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load dataset
# df = pd.read_csv('iris.csv')

# # Features and target
# X = df.drop('species', axis=1)  # All columns except 'species'
# y = df['species']               # Target column

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the decision tree model
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))


# Assignment 5 Random Forest Classifier
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load dataset
# df = pd.read_csv('winequality-red.csv', sep=';')

# # Convert quality into binary classification: good (1) or not good (0)
# df['quality'] = df['quality'].apply(lambda q: 1 if q >= 7 else 0)

# # Features and target
# X = df.drop('quality', axis=1)
# y = df['quality']

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))






# Assignment 6 naive_bayes

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Load dataset
# df = pd.read_csv('iris.csv')

# # Features and target
# X = df.drop('species', axis=1)
# y = df['species']

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the Naive Bayes model
# model = GaussianNB()
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluation
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))






# Assignment 7 Kmeans

# import pandas as pd
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv('iris.csv')

# # Features (we won't use the target column for clustering)
# X = df.drop('species', axis=1)

# # Apply KMeans with 3 clusters (since we know there are 3 species)
# kmeans = KMeans(n_clusters=3, random_state=42)
# kmeans.fit(X)

# # Get the cluster centroids
# centroids = kmeans.cluster_centers_

# # Predict the clusters
# y_kmeans = kmeans.predict(X)

# # Add cluster labels to the original dataset
# df['Cluster'] = y_kmeans

# # Plot the clusters
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')  # Mark the centroids
# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')
# plt.title('K-Means Clustering (Iris Dataset)')
# plt.show()

# # Show the cluster centers
# print("Cluster Centers:")
# print(centroids)

