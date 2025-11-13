# gdp-analysis

This project performs a comprehensive analysis of global socioeconomic indicators and builds multiple machine learning and deep learning models to predict **GDP per capita** and **Total GDP** of world countries.

The workflow includes:

* Data cleaning & preprocessing
* Exploratory data analysis
* Correlation studies
* Multiple regression model training
* Deep learning (LSTM) modeling
* Future GDP forecasting
* Comparative analysis of model performance



##  Project Overview

The dataset contains information about countries around the world, covering features such as:

* Population
* Area
* Literacy rate
* Birth & death rates
* Agriculture / Industry / Service contribution
* Climate
* Migration
* Phones per thousand
* GDP per capita

The main objective is to understand **which factors influence GDP** and to build predictive models that can estimate GDP for future years.

---

##  Data Preprocessing

The dataset required several preprocessing steps:

###  Conversion of comma-based numeric values

Columns like *Literacy (%)*, *Birthrate*, *Arable (%)*, etc. contained comma decimal formatting, which were converted to floating-point numbers.

###  Missing value imputation

Missing values across multiple columns were filled using **region-wise medians or most frequent values**, depending on the feature type.

###  Label encoding

Categorical variables such as:

* `Region`
* `Climate`

were converted into numeric labels to be used in machine learning models.

###  Feature scaling

Continuous variables were scaled using **MinMaxScaler** for models like LSTM.

---

##  Exploratory Data Analysis

The analysis includes:

* Identifying top GDP countries
* Visualizing GDP distribution
* Studying regional GDP differences
* Inspecting correlation between GDP and socioeconomic features
* Examining the impact of population, literacy, agriculture, and industry
* Stacked bar charts for sector contributions
* Pie charts for global GDP distribution

These visualizations help understand **key drivers of GDP**.

---

##  Models Implemented

Four machine learning/deep learning models were trained:

### **1. Linear Regression**

A baseline model to understand linear relationships between features and GDP.

### **2. Random Forest Regressor**

A tree-based ensemble model capable of capturing non-linear relationships and feature interactions.

### **3. XGBoost Regressor**

A boosting model that generally performs better on structured/tabular data.
This model delivered the most reliable and stable predictions among all traditional ML models.

### **4. LSTM (Long Short-Term Memory)**

A deep learning model designed for sequential data.
Due to the nature of the dataset (non-time-series), LSTM did not perform as effectively as the tree-based models.

---

##  **Model Comparison (Qualitative)**

* **XGBoost** provided the **best balance** between learning patterns and generalizing to unseen data.
* **Random Forest** also performed strongly, especially in capturing non-linear patterns.
* **Linear Regression** worked well as a baseline but lacks complexity for capturing deeper relationships.
* **LSTM** was **not suitable** for this dataset since the data is not sequential/time-series in nature.

---

##  Future GDP Forecasting

Forecasts were generated for future years (2024 and 2025) using:

* Linear Regression
* Random Forest
* XGBoost
* LSTM (scaled + inverse transformed)

Tree-based models gave **realistic and stable forecasts**, while LSTM produced unstable outputs due to the dataset not being temporal.

---


## This is the implementation of the entire project. Please Do refer the open link provided. 
### The Data I have used is proivded in this git.

https://colab.research.google.com/drive/1JnSEPSPiSh_UovNAJUiecPoOI-X1N6Hd?usp=sharing
