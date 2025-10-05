import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

loan_data = pd.read_csv("loan_data.csv")

loan_data.head()

def show_loan_distributor(data):
    count = ""
    if isinstance(data, pd.DataFrame):
        count = data["not.fully.paid"].value_counts()
    else:
        count = data.value_counts()
    
    count.plot(kind='pie', explode=[0,0.1], figsize=(6,6), autopct='%1.1f%%', shadow=True)
    plt.ylabel("Loan:Fully paid vs not fully paid")
    plt.legend(["Fully paid", "Not fully paid"])
    plt.show()

#show_loan_distributor(loan_data)
encoded_loan_data = pd.get_dummies(loan_data, prefix="purpose", drop_first=True)

X = encoded_loan_data.drop('not.fully.paid', axis = 1)
y = encoded_loan_data['not.fully.paid']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify = y, random_state=2022)

X_train_cp = X_train.copy()
X_train_cp['not.fully.paid'] = y_train
y_0 = X_train_cp[X_train_cp['not.fully.paid'] == 0]
y_1 = X_train_cp[X_train_cp['not.fully.paid'] == 1]
y_0_undersample = y_0.sample(y_1.shape[0])
loan_data_undersample = pd.concat([y_0_undersample, y_1], axis = 0)


# Visualize the proportion of borrowers
#show_loan_distrib(loan_data_undersample)

#print(encoded_loan_data.dtypes)
