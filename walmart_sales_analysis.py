import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

df = pd.read_csv("walmart_sales_dataset.csv")
print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

df.fillna(method='ffill', inplace=True)
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='Unemployment', y='Weekly_Sales', data=df, hue='Store')
plt.title("Weekly Sales vs Unemployment")
plt.show()



df['Month'] = df['Date'].dt.month
monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
plt.figure(figsize=(8,5))
monthly_sales.plot(kind='bar')
plt.title("Average Monthly Sales (Seasonality)")
plt.show()


plt.figure(figsize=(10,5))
sns.scatterplot(x='Temperature', y='Weekly_Sales', data=df)
plt.title("Temperature vs Weekly Sales")
plt.show()



plt.figure(figsize=(10,5))
sns.scatterplot(x='CPI', y='Weekly_Sales', data=df)
plt.title("CPI vs Weekly Sales")
plt.show()



store_sales = df.groupby('Store')['Weekly_Sales'].sum().sort_values(ascending=False)
print("Top Performing Stores:\n", store_sales.head())



print("Worst Performing Store:\n", store_sales.tail(1))


diff = store_sales.iloc[0] - store_sales.iloc[-1]
print(f"\nDifference between best and worst store: {diff}")




store1 = df[df['Store'] == 1][['Date', 'Weekly_Sales']]
store1.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}, inplace=True)


m = Prophet()
m.fit(store1)


future = m.make_future_dataframe(periods=12, freq='W')
forecast = m.predict(future)


m.plot(forecast)
plt.title("Store 1 - Weekly Sales Forecast (Next 12 Weeks)")
plt.show()
