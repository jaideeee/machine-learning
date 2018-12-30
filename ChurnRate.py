import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(df) :
    df = df.drop(columns=['user','churn','housing','payment_type','zodiac_sign'])
    for i in range(df.shape[1]) :
        print(i)
        plt.subplot(6,5,i+1)
        fig = plt.gca()
        fig.set_title(df.columns.values[i])
        vals = np.size(df.iloc[:,i].unique())
        plt.hist(df.iloc[:,i],bins=vals)
    plt.show()

def plot_pie_char(dataset):
    ## Pie Plots
    dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                        'web_user', 'app_web_user', 'ios_user',
                        'android_user', 'registered_phones', 'payment_type',
                        'waiting_4_loan', 'cancelled_loan',
                        'received_loan', 'rejected_loan', 'zodiac_sign',
                        'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]
    # fig = plt.figure(figsize=(15, 12))
    # plt.suptitle('Pie Chart Distributions', fontsize=20)
    # for i in range(1, dataset2.shape[1] + 1):
    #     plt.subplot(4, 5, i)
    #     f = plt.gca()
    #     f.set_title(dataset2.columns.values[i - 1])
    #     values = dataset2.iloc[:, i - 1].value_counts(normalize=True).values
    #     index = dataset2.iloc[:, i - 1].value_counts(normalize=True).index
    #
    #     plt.pie(values, labels=index, autopct='%1.1f%%')
    #     plt.axis('equal')
    # plt.show()
    #from pie chart waiting_4_loan , cancelled_loan , received_loan, rejected_loan
    # very less %of 1's .
    w_for_loan = dataset[dataset2.waiting_4_loan ==1].churn.value_counts()
    cancel_loan = dataset[dataset2.cancelled_loan == 1].churn.value_counts()
    rec_loan = dataset[dataset2.received_loan == 1].churn.value_counts()
    rej_loan = dataset[dataset2.rejected_loan == 1].churn.value_counts()
    print("waiting_for_loan", w_for_loan)
    print("cancel_loan", cancel_loan)
    print("rec_loan", rec_loan)
    print("rej_loan", rej_loan)



data = pd.read_csv(r'C:\Users\lenovo\Downloads\Minimizing_Churn_data\churn_data.csv')
# missing values =>> age (4) , credit_score (8031), rewards_earned (3327)
# how to handle missing values ?
#age only 4 , can be removed , other columns can be removed
data = data.drop(columns=['credit_score','rewards_earned'])
data = data.dropna(subset=['age'])

#check correlation with output variable
#data.drop(columns=['user','churn','housing','payment_type','zodiac_sign']).corrwith(data.churn).plot.bar(rot=0)

#inter correlation
corr = data.drop(columns=['user','churn']).corr()
corr = corr >0.5
sns.heatmap(corr,annot=True)
plt.show()
data = data.drop(columns=['app_web_user'])
data.to_csv('churn_data_processed.csv',index=False)