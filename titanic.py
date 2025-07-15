# Importing required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn modules for preprocessing, modeling, and evaluation
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\Users\maraj\Downloads\titanic.csv")

# Overview
print("Dataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nFirst 5 rows:")
print(df.head())

# Checking Null values
print(df.isnull().sum())

# Checking Duplicates
print("\nNumber of duplicate rows before dropping:", df.duplicated().sum())
print("Shape before droppingduplicates: ",df.shape)

# Dropping duplicates if exists
df.drop_duplicates(inplace=True)

print("\nDuplicates dropped. New shape:", df.shape)
print("Number of duplicate rows after dropping:", df.duplicated().sum())

# Rename columns for ease
df.rename(columns={
    'Siblings/Spouses Aboard': 'SibSp',
    'Parents/Children Aboard': 'Parch'
}, inplace=True)

# Creating FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Creating IsAlone feature
df['IsAlone'] = 1  # default alone
df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0  # not alone if family size >1

# Creating AgeGroup feature
df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,35,60,80], labels=['Child','Teen','YoungAdult','Adult','Senior'])
# Creating FareGroup feature
df['FareGroup'] = pd.cut(df['Fare'], bins=[0,20,50,100,600], labels=['Low','Medium','High','VeryHigh'])
print("\nFeature Engineering Done. Sample of new features:")
print(df[['FamilySize','IsAlone','AgeGroup','FareGroup']].head())

plt.figure(figsize=(6,6))
gender_counts = df['Sex'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue','pink'])
plt.title("Gender Distribution")
plt.axis('equal')
plt.show()

sns.countplot(x='Survived', data=df, palette='pastel')
plt.title("Overall Survival Count (0=Died,1=Survived)")
plt.show()

# Bar plot: Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by Gender")
plt.show()

# Bar plot: Survival by Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by Passenger Class")
plt.show()

# KDE plot: Age distribution by Survival
sns.kdeplot(data=df[df['Survived']==1]['Age'], fill=True, label='Survived', color='green')
sns.kdeplot(data=df[df['Survived']==0]['Age'], fill=True, label='Did Not Survive', color='red')
plt.title("Age Distribution: Survived vs Did Not Survive")
plt.xlabel("Age")
plt.legend()
plt.show()

# Boxplot: Fare by Pclass
sns.boxplot(x='Pclass', y='Fare', data=df, palette='pastel')
plt.title("Fare Distribution by Passenger Class")
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Scatter plot: Age vs Fare by Survival
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='muted')
plt.title("Age vs Fare colored by Survival")
plt.show()

# Violin plot: Fare by Gender
sns.violinplot(x='Sex', y='Fare', data=df, palette='pastel')
plt.title("Fare Distribution by Gender")
plt.show()

sns.violinplot(x='Survived', y='FamilySize', data=df, palette='plasma')
plt.title("Violin Plot: Family Size vs Survival")
plt.show()

# Boxplot: Fare outlier detection
sns.boxplot(y='Fare', data=df, palette='inferno')
plt.title("Fare Outliers Detection")
plt.show()

high_fare = df[df['Fare'] > 100][['Name', 'Fare']]
high_fare_sorted = high_fare.sort_values(by='Fare', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=high_fare_sorted, x='Fare', y='Name', palette='coolwarm')
plt.title("Passengers Who Paid Fare > 100")
plt.xlabel("Fare")
plt.ylabel("Passenger Name")
plt.tight_layout()
plt.show()

# Using IQR method to remove outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
Lower_limit = Q1 - 1.5 * IQR
Upper_limit = Q3 + 1.5 * IQR

# Filtering out the outliers
df = df[(df['Fare'] >= Lower_limit) & (df['Fare'] <= Upper_limit)]

plt.figure(figsize=(8,6))
sns.boxplot(x='Fare', data=df, palette='coolwarm')
plt.title("After Fare Outlier Removal")
plt.show()

# Count plot: AgeGroup vs Survival
sns.countplot(x='AgeGroup', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by Age Group")
plt.show()

# Count plot: FareGroup vs Survival
sns.countplot(x='FareGroup', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by Fare Group")
plt.show()

# Count plot: IsAlone vs Survival
sns.countplot(x='IsAlone', hue='Survived', data=df, palette='coolwarm')
plt.title("Survival by IsAlone")
plt.show()

plt.figure(figsize=(10,6))
sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=df, palette='viridis')
plt.title("Swarmplot: Age vs Class ")
plt.show()

# Pairplot for numeric features with hue=Survived
sns.pairplot(df, vars=['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize'], hue='Survived', palette='inferno')
plt.show()

print("\nMean Survival Rate by Passenger Class:")
print(df.groupby('Pclass')['Survived'].mean())

print("\nMean Survival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())

print("\nMean Survival Rate by IsAlone:")
print(df.groupby('IsAlone')['Survived'].mean())

print("\nMean Survival Rate by Age Group:")
print(df.groupby('AgeGroup')['Survived'].mean())

print("\nMean Survival Rate by Fare Group:")
print(df.groupby('FareGroup')['Survived'].mean())

# Encoding the categories
le = LabelEncoder()
df['Sex_enc'] = le.fit_transform(df['Sex'])
df['AgeGroup_enc'] = le.fit_transform(df['AgeGroup'].astype(str))
df['FareGroup_enc'] = le.fit_transform(df['FareGroup'].astype(str))

# Training Model
X = df[['Pclass','Age','Fare','SibSp','Parch','FamilySize','IsAlone','Sex_enc','AgeGroup_enc','FareGroup_enc']]
y = df['Survived']

# Splitting the training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("\n KNN Model Results (k=3)")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))

# KNN Accuracy for Multiple k values
accuracy_scores = []
k_range = range(1,21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_k = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracy_scores.append(acc)

plt.figure(figsize=(10,6))
plt.plot(k_range, accuracy_scores, marker='o', color='teal')
plt.title("KNN Accuracy for Different k Values")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Decision Tree Model
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)
print("\n Decision Tree Model Results (max_depth=5)")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

# Visualization of the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Decision Tree Model with max depth 3
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)
print("\n Decision Tree Model Results (max_depth=3)")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
