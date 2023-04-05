# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
1. Importing the libraries
2. Importing the dataset
3. Taking care of missing data
4. Encoding categorical data
5. Normalizing the data
6. Splitting the data into test and train

## PROGRAM:
```
Program Developed by: Vineesh.M
Register Number: 212221230122
```
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
```python
#read the dataset
df=pd.read_csv('Churn_Modelling data.csv')
df
```
```python
#drop unwanted columns
df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
```
```python
#checking for null, duplicates, outliers in lasrt column
df.isnull().sum()

df.duplicated()

df['Exited'].describe()
```

```python
#normalising data to normal distribution
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df),columns=['CreditScore','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited'])
df2
```

```python
#split dataset
x=df2.iloc[:,:-1].values #all rows from all except last column
x
```
```python
y=df2.iloc[:,-1].values #all rows from only last column
y
```
```python
##creating training and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train)
print("Size of X_train: ",len(X_train))
```
```python
print(X_test)
print("Size of X_test: ",len(X_test))
```

## OUTPUT:
### Dataset and Its Properties
![1](https://user-images.githubusercontent.com/93427254/230002649-9d967a13-cf95-4f05-b508-e9e29adcd01c.png)
![2](https://user-images.githubusercontent.com/93427254/230002650-3c19bc81-47ae-4856-ab79-47bbbadda151.png)
![3](https://user-images.githubusercontent.com/93427254/230002652-9d83de01-375f-496a-8533-39bd5599bfe4.png)
![4](https://user-images.githubusercontent.com/93427254/230002617-c47ee70b-107e-4abd-ae86-cfb34c562fc8.png)
![5](https://user-images.githubusercontent.com/93427254/230002629-04c32999-e79a-48f6-90cc-f81f25496ab8.png)

### Normalised Dataset
![6](https://user-images.githubusercontent.com/93427254/230002632-45b51e60-2f4f-491b-9b98-6a662675b4a7.png)

### X and Y Column Data
![7](https://user-images.githubusercontent.com/93427254/230002634-67a6129e-6de6-4d64-8114-80e5107c07f0.png)
![8](https://user-images.githubusercontent.com/93427254/230002636-ec9e678b-512f-492f-80ae-1aff81224e73.png)

### Training Data
![9](https://user-images.githubusercontent.com/93427254/230002640-e1c61467-ac2a-45ff-a193-93a8baa69793.png)

### Test Data
![10](https://user-images.githubusercontent.com/93427254/230002646-0677b9f7-437c-4568-82ef-4b0db8c6f9cc.png)


## RESULT
Thus, the Data preprocessing is performed over a data set successfully.
