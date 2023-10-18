# EX-06 FEATURE TRANSFORMATION
### Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
### Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### Algorithm:
- Step1: Read the given Data.
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature Transformation techniques to all the features of the data set.
- Step4: Print the transformed features.
### Program:
```
Developed By: Shabreena Vincent
Register No: 212222230141
```
- Importing libraries and reading csv file:
  ```Python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm
  import scipy.stats as stats
  from sklearn.preprocessing import QuantileTransformer
  from sklearn.preprocessing import PowerTransformer
  df=pd.read_csv("Data_to_Transform.csv")
  ```
- Basic Information:
  ```Python
  df.head()
  df.info()
  df.info()
  ```
  <br>
  <img height=12% width=55% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/e1d70826-3e9e-4413-a1c3-ef9c3dc8747d">
  <img height=12% width=30% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/98818400-9ff3-4ad7-9ccf-ae9fa1129bb4">
  <img height=15% width=60% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/66652522-db31-409b-a238-ddb456185569">

  
- Before Transformation:
  ```Python
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()

  sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
  plt.title("Highly Negative Skew")
  plt.show()

  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  ```
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/aa5c908b-6571-4164-8e97-9bd5d6a7d22d">
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/86450d9c-b6cc-4402-8818-4794452776f6">
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/538b036b-d6ad-4c11-b96b-6612e9b019d3">
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/71b58850-8cea-4ae7-af4e-cee23a3597c1">  
- Log Transformation:
  ```Python
  df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  
  df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()
  ```
  <img height=17% width=43% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/40eb3b47-5fab-4bcc-b29b-166a14481d07">
  <img height=17% width=43% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/3db41bdf-7563-4192-bbc7-298ba9836c9e">
  
- Reciprocal Transformation:
  ```Python
  df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```
  <img height=17% width=43% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/ff507c37-5cdf-49f5-bdd8-26ad510444fb">

- SquareRoot Transformation:
  ```Python
  df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```
  <img height=17% width=43% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/4bbde658-4fb6-44c2-a810-256118b7986a">

- Power Transformation:
  ```Python
  df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  transformer=PowerTransformer("yeo-johnson")
  df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  ```
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/af6ffcb2-0549-41d4-a394-0447731adea0">
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/2a2955c9-5c2d-425d-b87a-1ed942cc772b">

  
- Quantile Transformation:
  ```Python
  qt = QuantileTransformer(output_distribution = 'normal')
  df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate  Negative Skew")
  plt.show()
  ```
  <img height=20% width=49% src="https://github.com/ROHITJAIND/EX-06-FEATURE-TRANSFORMATION/assets/118707073/9fd9482d-f704-453c-a0b7-1aa52561e39f">

### Result:  
Thus feature transformation is done for the given dataset.
