import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import warnings
from sklearn.metrics import r2_score,mean_squared_error


warnings.filterwarnings("ignore")

applications = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2021/applications.csv")
print(applications.describe())
print(applications.dtypes)
print(applications.nunique())
print(applications.isnull().sum())
print(applications.duplicated().sum())


plt.figure(figsize=(10,6))
applications['homeownership'].value_counts().plot(kind='bar',rot=0)
plt.show()

applications['homeownership'] = [1 if X == "Own" else 0 for X in applications['homeownership']]

plt.figure(figsize=(11,6))
sns.heatmap(applications.corr(), fmt='f',annot=True,cmap="Blues")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='credit_limit',y='income',data=applications)
plt.show()


plt.figure(figsize=(10,6))
sns.scatterplot(x='purchases',y='income',data=applications)
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='purchases',y='credit_limit',data=applications)
plt.show()

X = applications.drop(['app_id','zip_code','purchases','ssn'],axis=1)
y = applications['purchases']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "LinearRegression":LinearRegression(),
    "lasso":Lasso(),
    "ridge":Ridge(),
    "RandomForestRegressor":RandomForestRegressor(),
    "DecisionTreeRegressor":DecisionTreeRegressor()
    }



def evaluate_models(X_train_scaled,X_test_scaled,y_train,y_test,model):
    model = model.fit(X_train_scaled,y_train)
    pred= model.predict(X_test_scaled)
    r2 = r2_score(y_test,pred)
    mse = mean_squared_error(y_test, pred)
    print(f'{model.__class__.__name__}; --R2 Score-- {r2*100:.2f}; --MSE-- {mse}')
    return pred

for model_name,model in models.items():
    pred = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, model)
    print(f'results: {model_name}')





#with polynomial features

poly_x = PolynomialFeatures(degree=3)
X_train_poly = poly_x.fit_transform(X_train)
X_test_poly = poly_x.transform(X_test)


for model_name,model in models.items():
    pred = evaluate_models(X_train_poly, X_test_poly, y_train, y_test, model)
    print(f'results using polynomial features: {model_name}')



credit = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2021/credit_bureau.csv")

print(credit.isnull().sum())
print(credit.duplicated().sum())
print(credit.dtypes)
print(credit.nunique())

plt.figure(figsize=(10,6))
sns.heatmap(credit.corr(), fmt='f',annot=True,cmap="coolwarm")
plt.show()


purch_app_bureau = pd.concat([applications,credit],join='inner',axis=1)
purch_app_bureau = purch_app_bureau.loc[:,~purch_app_bureau.columns.duplicated()].copy()
purch_app_bureau.isnull().sum()
purch_app_bureau.duplicated().sum()
purch_app_bureau.dtypes

plt.figure(figsize=(12,6))
sns.heatmap(purch_app_bureau.corr(),fmt='f',cmap="Blues",annot=True)
plt.show()

X = purch_app_bureau.drop(['app_id','ssn','zip_code','purchases'],axis=1)
y = purch_app_bureau['purchases']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for model_name, model in models.items():
    pred = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, model)
    print("Results from Purch App models: {model_name}")
    
    
    
poly_x = PolynomialFeatures(degree=2)
X_train_poly = poly_x.fit_transform(X_train)
X_test_poly = poly_x.fit_transform(X_test)

    

for model_name,model in models.items():
    print(f'results using polynomial features for purch_app: {model_name}')
    pred = evaluate_models(X_train_poly, X_test_poly, y_train, y_test, model)



demographic = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2021/demographic.csv")

demographic.describe()
demographic.isnull().sum()
demographic.dtypes
demographic.nunique()
demographic.duplicated().sum()


plt.figure(figsize=(10,6))
sns.heatmap(demographic.corr(), fmt='f',annot=True,cmap='coolwarm')
plt.show()


purchase_full = pd.concat([applications,demographic,credit],join='inner',axis=1)
purchase_full = purchase_full.loc[:,~purchase_full.columns.duplicated()].copy()
purchase_full.isnull().sum()
purchase_full.duplicated().sum()
purchase_full.describe()
purchase_full.dtypes

plt.figure(figsize=(12,6))
sns.heatmap(purchase_full.corr(),fmt='f',annot=True,cmap="Blues")
plt.show()


X = purchase_full.drop(['app_id','ssn','zip_code','purchases'],axis=1)
y = purchase_full['purchases']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




for model_name,model in models.items():
    pred = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, model)
    print(f'results from purchase full: {model_name}')




poly_x = PolynomialFeatures(degree=3)
X_train_poly = poly_x.fit_transform(X_train)
X_test_poly = poly_x.transform(X_test)

    

for model_name,model in models.items():
    print(f'results from polynomial features for purchase full: {model_name}')
    pred = evaluate_models(X_train_poly, X_test_poly, y_train, y_test, model)





utilization = purchase_full['purchases'] / purchase_full['credit_limit']


utilization.describe
max_utils = np.max(utilization)
min_utils = np.min(utilization)
print('Maxium Utils\n')
print(max_utils)
print('minimum utils')
print(min_utils)

X = purch_app_bureau.drop(['app_id','ssn','zip_code'],axis=1)
y=  utilization

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



for model_name,model in models.items():
    pred = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, model)
    print(f'results from utility model: {model_name}')




    

for model_name,model in models.items():
    pred = evaluate_models(X_train_poly, X_test_poly, y_train, y_test, model)
    print(f'results for utilization: {model_name}')



log_odds_utils = abs(np.log(utilization)) / (1-(utilization))

log_odds_utils.describe()

X = purch_app_bureau.drop(['app_id','ssn','zip_code'],axis=1)
y = log_odds_utils

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



for model_name,model in models.items():
    pred = evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, model)
    print(f'results from log odds utils: {model_name}')
    






#just hyperparameter tune like everyone else
lr_params = {
    "copy_X": [True, False],
    "fit_intercept": [True, False],
    "n_jobs": [-1, 1],
}

lr_model = GridSearchCV(LinearRegression(), param_grid=lr_params,cv=5,scoring='neg_mean_squared_error').fit(X_train_scaled,y_train)
print(lr_model.best_estimator_)
print(lr_model.best_params_)
print(lr_model.best_score_)
lr = LinearRegression(copy_X=True,fit_intercept=True,n_jobs=-1).fit(X_train_scaled,y_train)
lr_pred = lr.predict(X_test_scaled)
print('Best R2 Score For Linear Regression model')
print(r2_score(y_test, lr_pred))



lasso_params = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_model = GridSearchCV(Lasso(),lasso_params,cv=5,scoring='neg_mean_squared_error').fit(X_train_scaled,y_train)
print(lasso_model.best_params_)
print(lasso_model.best_score_)

lasso = Lasso(alpha=0.01).fit(X_train_scaled,y_train)
lasso_pred = lasso.predict(X_test_scaled)
r2_lasso = r2_score(y_test, lasso_pred)
print(f'r2 score for best lasso params: {r2_lasso*100:.2f}')


ridge_params = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_model = GridSearchCV(Ridge(),ridge_params,cv=10,scoring='neg_mean_squared_error').fit(X_train_scaled,y_train)
print(ridge_model.best_params_)
print(ridge_model.best_score_)
ridge = Ridge(alpha=1).fit(X_train_scaled,y_train)
ridge_pred = ridge.predict(X_test_scaled)
print('best r2 score from ridge regressor')
print(r2_score(y_test, ridge_pred))


