import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
df = pd.read_csv('gym_members_exercise_tracking.csv')

print(df.shape)
print(df.head())
print(df.tail())
print(df.columns)
print(df.info())


print(df.describe())

print(df['Calories_Burned'].describe())

df['Calories_Burned'].hist(bins=20 , edgecolor='black')
plt.title("Distiution of Calories Burned")
plt.xlabel("Calories Burned")
plt.ylabel("Number of people")
plt.show()


plt.scatter(df['Session_Duration (hours)'] , df['Calories_Burned'] , alpha=0.5)
plt.title("Session Duration vs Calories Burned ")
plt.xlabel("Session Duration (hours)")
plt.ylabel("Calories Burned")
plt.show()


correlation = df['Session_Duration (hours)'].corr(df['Calories_Burned'])
print(correlation)


# select only numeric columns first
numeric_df = df.select_dtypes(include='number')

correlation_matrix = numeric_df.corr()
print(correlation_matrix['Calories_Burned'].sort_values(ascending=False))


plt.figure(figsize=(10,8))
sns.heatmap(numeric_df.corr() , annot=True , cmap='coolwarm' , fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()



# step - 1 : pick out feature(s) and target
X = df[['Session_Duration (hours)']]  # input(s) - note the double brackets , sklearn needs a 2d shape
y = df['Calories_Burned']             # target - what we're predicting


# step - 2 : split into train/test
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

# step - 3 : create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# step - 4 : see what it learned
print("Slope : " , model.coef_[0])
print("Intercept : " , model.intercept_)




# # predict on the test set (data the model never saw during training)
y_pred = model.predict(X_test)

r2 = r2_score(y_test , y_pred)
print("R^2 Score :" , r2)

plt.scatter(X_test , y_pred , alpha=0.5 , label='Actual Data')
plt.plot(X_test , y_pred , color='red' , linewidth=2 , label='Model prediction (line)')
plt.title('Session Duration vs Calories Burned — with Regression Line')
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.legend()
plt.show()


# step - 1 : pick multiple features this time
X = df[['Session_Duration (hours)', 'Experience_Level', 'Fat_Percentage']]
y = df['Calories_Burned']

# step - 2 : split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state=42)

# step - 3 : Train
model2 = LinearRegression()
model2.fit(X_train , y_train)

# step - 4 : check the weights it learned
for features , weight in zip(X.columns , model2.coef_):
    print(f"{features}: {weight:.2f}")
print("Intercept : ", model2.intercept_)


# step - 5 :evalute on unseen test data
y_pred2 = model2.predict(X_test)
print("R^2 Score : " ,r2_score(y_test , y_pred2))


# Step 1: pick features that describe workout BEHAVIOR (not target-leaking columns)
cluster_features = df[['Session_Duration (hours)', 'Calories_Burned', 
                        'Workout_Frequency (days/week)', 'Experience_Level', 'Fat_Percentage']]

# Step 2: scale them
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)


# step - 3 clustering . finding optimal K
inertia_values = []
k_range = range(1,11) # test K from 1 to 11

for k in k_range:
    kmeans = KMeans(n_clusters=k  , random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method - Finding Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.show()

# TODO : After run this , taking K = 4  , reason in Doc !!

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print(df['Cluster'].value_counts())

cluster_profile = df.groupby('Cluster')[['Session_Duration (hours)', 'Calories_Burned', 
                                            'Workout_Frequency (days/week)', 
                                            'Experience_Level', 'Fat_Percentage']].mean()
print(cluster_profile)

plt.figure(figsize=(8,6))
scatter = plt.scatter(df['Session_Duration (hours)'], df['Calories_Burned'], 
                        c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Session Duration (hours)')
plt.ylabel('Calories Burned')
plt.title('Gym Member Clusters')
plt.colorbar(scatter, label='Cluster')
plt.show()

cluster_name = {
    0: 'Elite Trainer',
    1: 'Occasional Intense Trainer',
    2: 'Consistent Intermediate Trainer',
    3: 'Beginner / Light Trainer'
}

df['Cluster_Name'] = df['Cluster'].map(cluster_name)
print(df[['Cluster' , 'Cluster_Name']].head(10))


# save the regression model (predicts calories burned)
joblib.dump(model , 'calorie_model.pkl')

# save the scaler (needed to transform new user input to the same way as training data)
joblib.dump(scaler , 'scaler.pkl')

# save the trained KMean model (assigns new user to a cluster / persona)
joblib.dump(kmeans , 'kmeans_model.pkl')

# save the cluster name mapping too , so the app can show text , not numbers
joblib.dump(cluster_name , 'cluster_names.pkl')

print("All files saved !")

joblib.dump(model2 , 'calories_model_multi.pkl')
print("Multi-feature calories model saved !")