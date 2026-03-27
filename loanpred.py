import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
df=pd.read_csv("F:\\pythonproj\\loan_dataset\\train.csv")
df.drop(["Loan_ID","Dependents"],axis=1,inplace=True)
cate=["Gender","Married","Self_Employed"]
for cat in cate:
    df[cat]=df[cat].fillna(df[cat].mode()[0])
nume=["LoanAmount","Loan_Amount_Term"]
for num in nume:
    df[num]=df[num].fillna(df[num].median())
df["Credit_History"]=df["Credit_History"].fillna(df["Credit_History"].mean())
df=pd.get_dummies(df,columns=["Gender","Married","Self_Employed","Education","Property_Area","Loan_Status"],drop_first=True)
# print(df.head())
model=LogisticRegression()
x=df.drop("Loan_Status_Y",axis=1)
y=df["Loan_Status_Y"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))