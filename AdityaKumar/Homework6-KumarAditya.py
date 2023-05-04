import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Q1.
df = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"The first 5 rows of the dataset are as follows:\n{df.head().to_string()}")
print(df.info())
print(df.describe())
#3 numerical columns
print(df.shape)
# df[:5].plot()
# plt.show()

print(f"Numerical features are: {df.select_dtypes(include=['int','float']).columns.values}")
print(f"Categorical features are: {df.select_dtypes(exclude=['int','float']).columns.values}")

# Q2
df["TotalCharges"] = df["TotalCharges"].astype(float)
# Error due to empty string
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)
print(df.head())
print(df.TotalCharges.head())

# Q3
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

# Q4
df.isnull().sum()
#Missing only in TotalCharges
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Q5-6
durations = df['tenure']
event_observed = df['Churn']

km = KaplanMeierFitter()
km.fit(durations,event_observed, label = "# Users that didn't Churn")
km.plot_survival_function()
plt.grid()
plt.title("Survival function for all Customers")
plt.xlabel("Time")
plt.ylabel("Survival Function")
plt.show()

# Q7
dMOM = df[df['Contract'] == 'Month-to-month']
d1Y = df[df['Contract'] == 'One year']
d2Y = df[df['Contract'] == 'Two year']

durations_mom = dMOM['tenure']
event_mom = dMOM['Churn']
durations_1Y = d1Y['tenure']
event_1Y = d1Y['Churn']
durations_2Y = d2Y['tenure']
event_2Y = d2Y['Churn']

km_mom = KaplanMeierFitter()
km_1Y = KaplanMeierFitter()
km_2Y = KaplanMeierFitter()
km_mom.fit(durations_mom,event_mom, label = 'Month-to-Month')
km_1Y.fit(durations_1Y,event_1Y, label = 'One Year')
km_2Y.fit(durations_2Y,event_2Y, label = 'Two Year')

km_mom.plot_survival_function()
km_1Y.plot_survival_function()
km_2Y.plot_survival_function()

plt.grid()
plt.suptitle("Survival Function for Different Contract Types")
plt.xlabel("Time")
plt.ylabel("Survival Function")
plt.show()

# Q8
df_S = df[df['StreamingTV'] == 'Yes']
df_NS = df[df['StreamingTV'] == 'No']

durations_S = df_S['tenure']
event_S = df_S['Churn']
durations_NS = df_NS['tenure']
event_NS = df_NS['Churn']

km_S = KaplanMeierFitter()
km_NS = KaplanMeierFitter()
km_S.fit(durations_S,event_S, label = 'Subscribed TV Streaming')
km_NS.fit(durations_NS,event_NS, label = 'Not Subscribed TV Streaming')

km_S.plot_survival_function()
km_NS.plot_survival_function()
plt.grid()
plt.suptitle("Survival Function for Subscribed/Non-Subscribed Customers")
plt.xlabel("Time")
plt.ylabel("Survival Function")
plt.show()

# Q9
df1 = pd.read_csv('https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/dd.csv')

Asia = df1[df1['un_continent_name'] == 'Asia']
Europe = df1[df1['un_continent_name'] == 'Europe']
Africa = df1[df1['un_continent_name'] == 'Africa']
Americas = df1[df1['un_continent_name'] == 'Americas']
Oceania = df1[df1['un_continent_name'] == 'Oceania']

durationsAsia = Asia['duration']
eventAsia = Asia['observed']
durationsEu = Europe['duration']
eventEu = Europe['observed']
durationsAf = Africa['duration']
eventAf = Africa['observed']
durationsAmr = Americas['duration']
eventAmr = Americas['observed']
durationsOc = Oceania['duration']
eventOc = Oceania['observed']

km_Asia = KaplanMeierFitter()
km_Eu = KaplanMeierFitter()
km_Af = KaplanMeierFitter()
km_Amr = KaplanMeierFitter()
km_Oc = KaplanMeierFitter()

km_Asia.fit(durationsAsia,eventAsia, label = 'Asia')
km_Eu.fit(durationsEu,eventEu, label = 'Europe')
km_Af.fit(durationsAf,eventAf, label = 'Africa')
km_Amr.fit(durationsAmr,eventAmr, label = 'America')
km_Oc.fit(durationsOc,eventOc, label = 'Oceania')

km_Asia.plot_survival_function()
km_Eu.plot_survival_function()
km_Af.plot_survival_function()
km_Amr.plot_survival_function()
km_Oc.plot_survival_function()

plt.grid()
plt.suptitle("Survival Function for Political Regimes based on Regions")
plt.xlabel("Time")
plt.ylabel("Survival Function")
plt.show()
