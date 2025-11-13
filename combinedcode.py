import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
data = pd.read_csv("world.csv")
data.info()
data.describe()
data.isnull().sum()
# Replace commas with periods and convert to numeric
cols_to_convert = ['Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
                   'Net migration', 'Infant mortality (per 1000 births)',
                   'Literacy (%)', 'Phones (per 1000)', 'Arable (%)', 'Crops (%)',
                   'Other (%)', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

for col in cols_to_convert:
    data[col] = data[col].str.replace(',', '.').astype(float)

# Now try the groupby operation again
data.groupby('Region')[['GDP ($ per capita)', 'Literacy (%)', 'Agriculture']].median()
for col in data.columns.values:
    if data[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_values = data.groupby('Region')[col].median()
    for region in data['Region'].unique():
        data[col].loc[(data[col].isnull())&(data['Region']==region)] = guess_values[region]
fig, ax = plt.subplots(figsize=(16,6))
top_gdp_countries = data.sort_values('GDP ($ per capita)',ascending=False).head(20)
mean = pd.DataFrame({'Country':['World mean'], 'GDP ($ per capita)':[data['GDP ($ per capita)'].mean()]})
gdps = pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']],mean],ignore_index=True)

sns.barplot(x='Country',y='GDP ($ per capita)',data=gdps, palette='Set3')
ax.set_xlabel(ax.get_xlabel(),labelpad=15)
ax.set_ylabel(ax.get_ylabel(),labelpad=30)
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
plt.xticks(rotation=90)
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20,12))
plt.subplots_adjust(hspace=0.4)

corr_to_gdp = pd.Series()
for col in data.columns.values[2:]:
    if ((col!='GDP ($ per capita)')&(col!='Climate')):
        corr_to_gdp[col] = data['GDP ($ per capita)'].corr(data[col])
abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)
corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]

for i in range(2):
    for j in range(3):
        sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=data,
                   ax=axes[i,j], fit_reg=False, marker='.')
        title = 'correlation='+str(corr_to_gdp[i*3+j])
        axes[i,j].set_title(title)
axes[1,2].set_xlim(0,102)
plt.show()
data.loc[(data['Birthrate']<14)&(data['GDP ($ per capita)']<10000)]
LE = LabelEncoder()
data['Region_label'] = LE.fit_transform(data['Region'])
data['Climate_label'] = LE.fit_transform(data['Climate'])
data.sample()
train, test = train_test_split(data, test_size=0.3, shuffle=True)
training_features = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'Literacy (%)', 'Phones (per 1000)',
       'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',
       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Region_label',
       'Climate_label','Service']
train_X = train[training_features]
train_Y = train['GDP ($ per capita)']
test_X = test[training_features]
test_Y = test['GDP ($ per capita)']
model1 = LinearRegression()
model1.fit(train_X, train_Y)
train_pred_Y = model1.predict(train_X)
test_pred_Y = model1.predict(test_X)
train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)
test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))
msle_train = mean_squared_log_error(train_pred_Y, train_Y)
rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))
msle_test = mean_squared_log_error(test_pred_Y, test_Y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)
# XGBoost Regressor

# Convert to NumPy arrays for XGBoost compatibility
model3 = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model3.fit(train_X.values, train_Y.values)

train_pred3 = pd.Series(model3.predict(train_X.values).clip(0), index=train_Y.index)
test_pred3 = pd.Series(model3.predict(test_X.values).clip(0), index=test_Y.index)

print("\nXGBoost Regressor:")
print("Train RMSE:", np.sqrt(mean_squared_error(train_pred3, train_Y)))
print("Train MSLE:", mean_squared_log_error(train_pred3, train_Y))
print("Test RMSE:", np.sqrt(mean_squared_error(test_pred3, test_Y)))
print("Test MSLE:", mean_squared_log_error(test_pred3, test_Y))
# LSTM Model for GDP ($ per capita)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Define the target column
target = 'GDP ($ per capita)'

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[training_features])
Y_scaled = data[[target]].values

# Reshape for LSTM input: (samples, timesteps, features)
X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train/test split
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, Y_scaled, test_size=0.3, random_state=42)

# Build LSTM model
model4 = Sequential()
model4.add(LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=False))
model4.add(Dropout(0.2))
model4.add(Dense(1))

# Compile and fit
model4.compile(optimizer='adam', loss='mean_squared_error')
model4.fit(X_train_seq, y_train_seq, epochs=100, batch_size=16, verbose=0)

# Predictions
y_train_pred_seq = model4.predict(X_train_seq).flatten()
y_test_pred_seq = model4.predict(X_test_seq).flatten()

# Evaluation
print("\nLSTM Model:")
print("Train RMSE:", np.sqrt(mean_squared_error(y_train_pred_seq, y_train_seq)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test_pred_seq, y_test_seq)))
train_msle = mean_squared_log_error(y_train_seq, y_train_pred_seq)
test_msle = mean_squared_log_error(y_test_seq, y_test_pred_seq)

print("Train MSLE:", train_msle)
print("Test MSLE:", test_msle)
model2 = RandomForestRegressor(n_estimators = 50,
                             max_depth = 6,
                             min_weight_fraction_leaf = 0.05,
                             max_features = 0.8,
                             random_state = 42)
model2.fit(train_X, train_Y)
train_pred_Y = model2.predict(train_X)
test_pred_Y = model2.predict(test_X)
train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)
test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))
msle_train = mean_squared_log_error(train_pred_Y, train_Y)
rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))
msle_test = mean_squared_log_error(test_pred_Y, test_Y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)
plt.figure(figsize=(18,12))

train_test_Y = pd.concat([train_Y, test_Y])
train_test_pred_Y = pd.concat([train_pred_Y, test_pred_Y])

data_shuffled = data.loc[train_test_Y.index]
label = data_shuffled['Country']

colors = {'ASIA (EX. NEAR EAST)         ':'red',
          'EASTERN EUROPE                     ':'orange',
          'NORTHERN AFRICA                    ':'gold',
          'OCEANIA                            ':'green',
          'WESTERN EUROPE                     ':'blue',
          'SUB-SAHARAN AFRICA                 ':'purple',
          'LATIN AMER. & CARIB    ':'olive',
          'C.W. OF IND. STATES ':'cyan',
          'NEAR EAST                          ':'hotpink',
          'NORTHERN AMERICA                   ':'lightseagreen',
          'BALTICS                            ':'rosybrown'}

for region, color in colors.items():
    X = train_test_Y.loc[data_shuffled['Region']==region]
    Y = train_test_pred_Y.loc[data_shuffled['Region']==region]
    ax = sns.regplot(x=X, y=Y, marker='.', fit_reg=False, color=color, scatter_kws={'s':200, 'linewidths':0}, label=region)
plt.legend(loc=4,prop={'size': 12})

ax.set_xlabel('GDP ($ per capita) ground truth',labelpad=40)
ax.set_ylabel('GDP ($ per capita) predicted',labelpad=40)
ax.xaxis.label.set_fontsize(24)
ax.yaxis.label.set_fontsize(24)
ax.tick_params(labelsize=12)

x = np.linspace(-1000,50000,100) # 100 linearly spaced numbers
y = x
plt.plot(x,y,c='gray')

plt.xlim(-1000,60000)
plt.ylim(-1000,40000)

for i in range(0,train_test_Y.shape[0]):
    if((data_shuffled['Area (sq. mi.)'].iloc[i]>8e5) |
       (data_shuffled['Population'].iloc[i]>1e8) |
       (data_shuffled['GDP ($ per capita)'].iloc[i]>10000)):
        plt.text(train_test_Y.iloc[i]+200, train_test_pred_Y.iloc[i]-200, label.iloc[i], size='small')
data['Total_GDP ($)'] = data['GDP ($ per capita)'] * data['Population']
top_gdp_countries = data.sort_values('Total_GDP ($)',ascending=False).head(10)
other = pd.DataFrame({'Country':['Other'], 'Total_GDP ($)':[data['Total_GDP ($)'].sum() - top_gdp_countries['Total_GDP ($)'].sum()]})
gdps = pd.concat([top_gdp_countries[['Country','Total_GDP ($)']],other],ignore_index=True)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7),gridspec_kw = {'width_ratios':[2,1]})
sns.barplot(x='Country',y='Total_GDP ($)',data=gdps,ax=axes[0],palette='Set3')
axes[0].set_xlabel('Country',labelpad=30,fontsize=16)
axes[0].set_ylabel('Total_GDP',labelpad=30,fontsize=16)

colors = sns.color_palette("Set3", gdps.shape[0]).as_hex()
axes[1].pie(gdps['Total_GDP ($)'], labels=gdps['Country'],colors=colors,autopct='%1.1f%%',shadow=True)
axes[1].axis('equal')
plt.show()
Rank1 = data[['Country','Total_GDP ($)']].sort_values('Total_GDP ($)', ascending=False).reset_index()
Rank2 = data[['Country','GDP ($ per capita)']].sort_values('GDP ($ per capita)', ascending=False).reset_index()
Rank1 = pd.Series(Rank1.index.values+1, index=Rank1.Country)
Rank2 = pd.Series(Rank2.index.values+1, index=Rank2.Country)
Rank_change = (Rank2-Rank1).sort_values(ascending=False)
print('Rank of total GDP - Rank of GDP per capita:')
Rank_change.loc[top_gdp_countries.Country]
corr_to_gdp = pd.Series()
for col in data.columns.values[2:]:
    if ((col!='Total_GDP ($)')&(col!='Climate')&(col!='GDP ($ per capita)')):
        corr_to_gdp[col] = data['Total_GDP ($)'].corr(data[col])
abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)
corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]
print(corr_to_gdp)
plot_data = top_gdp_countries.head(10)[['Country','Agriculture', 'Industry', 'Service']]
plot_data = plot_data.set_index('Country')
ax = plot_data.plot.bar(stacked=True,figsize=(10,6))
ax.legend(bbox_to_anchor=(1, 1))
plt.show()
plot_data = top_gdp_countries[['Country','Arable (%)', 'Crops (%)', 'Other (%)']]
plot_data = plot_data.set_index('Country')
ax = plot_data.plot.bar(stacked=True,figsize=(10,6))
ax.legend(bbox_to_anchor=(1, 1))
plt.show()
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Assuming you have the train_gdp from training data
train_gdp = np.array([25000, 27000, 29000, 31000, 33000])  # Replace with actual GDP values

# Fit target scaler using the training GDP values
target_scaler = MinMaxScaler()
target_scaler.fit(train_gdp.reshape(-1, 1))

# Sample feature set for 2024 and 2025
raw_future_data = {
    'Population': [15000000, 15200000],
    'Area (sq. mi.)': [500000, 500000],
    'Pop. Density (per sq. mi.)': [30.0, 30.4],
    'Coastline (coast/area ratio)': [0.5, 0.5],
    'Net migration': [1.5, 1.6],
    'Infant mortality (per 1000 births)': [10.0, 9.5],
    'Literacy (%)': [95.0, 95.2],
    'Phones (per 1000)': [800, 820],
    'Arable (%)': [20.0, 20.2],
    'Crops (%)': [5.0, 5.0],
    'Other (%)': [75.0, 74.8],
    'Birthrate': [12.0, 11.8],
    'Deathrate': [8.0, 7.9],
    'Agriculture': [0.2, 0.19],
    'Industry': [0.3, 0.31],
    'Service': [0.5, 0.5],
    'Region_label': [3, 3],
    'Climate_label': [1, 1]
}

# Create DataFrame and align feature order for scaling
future_data = pd.DataFrame(raw_future_data)
future_data = future_data[scaler.feature_names_in_]  # Ensure correct order for scaler

# Scale features
future_scaled = scaler.transform(future_data)

# Reshape for LSTM
future_seq = future_scaled.reshape((future_scaled.shape[0], 1, future_scaled.shape[1]))

# Get predictions from each model
xgb_pred = model3.predict(future_data.values)
lstm_pred_scaled = model4.predict(future_seq)

# Inverse transform LSTM predictions
lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled).flatten()

# Final Output
print("Linear Regression Prediction:", model1.predict(future_data))
print("Random Forest Prediction:", model2.predict(future_data))
print("XGBoost Prediction:", xgb_pred)
print("LSTM Prediction (Inverse Transformed):", lstm_pred)
