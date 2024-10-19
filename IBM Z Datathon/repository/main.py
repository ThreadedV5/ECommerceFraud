import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("IBM Z Datathon\Fraudulent_E-Commerce_Transaction_Data.csv")
#print(df.head())

df = df.drop(columns=["Transaction ID","Customer ID","Billing Address","Transaction Date","IP Address"])
df = df.dropna()



from sklearn.utils import resample

# split data
df_no_fraud = df.loc[(df['Is Fraudulent']==0)]
print(len(df_no_fraud.axes[0]))
df_fraud = df.loc[(df['Is Fraudulent']==1)]
print(len(df_fraud.axes[0]))

# downsample the data set
df_no_fraud_downsampled = resample(df_no_fraud, replace=False, n_samples=10000, random_state=42 )
df_fraud_downsampled = resample(df_fraud, replace=False, n_samples=10000, random_state=42 )

#check ouput
#print(len(df_no_fraud_downsampled))
#len(df_fraud_downsampled)

# merge the data sets
df_downsample = pd.concat([df_no_fraud_downsampled, df_fraud_downsampled ])
df_downsample.shape

# isolate independent variables
#X = df_downsample.drop(['DEFAULT','SEX', 'EDUCATION', 'MARRIAGE','AGE'], axis=1).copy()

X_encoded = pd.get_dummies(data=df_downsample, columns=["Transaction Amount", "Payment Method", "Product Category", "Quantity",
                                            "Customer Age", "Customer Location", "Device Used","Shipping Address",
                                            "Account Age Days", "Transaction Hour"])
#, "Transaction Date", "IP Address"
X_encoded.head()


from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
y = df_downsample['Is Fraudulent'].copy()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=69)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
clf_rf.fit(X_train, y_train)

#calculate overall accuracy
y_pred = clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')

class_names = ['Not Fraud', 'Fraud']
disp = ConfusionMatrixDisplay.from_estimator(
        clf_rf,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues)