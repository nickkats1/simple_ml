import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,f1_score,confusion_matrix
import numpy as np

warnings.filterwarnings('ignore')

df = pd.read_csv('https://raw.githubusercontent.com/LeeMorinUCF/QMB6358S22/main/demo_24_Classification/credit.csv')
df.isna().sum()
df.duplicated().sum()
df.dtypes
df.nunique()

plt.figure(figsize=(10,6))
df['Default'].value_counts().plot(kind='bar',rot=0)
plt.show()



plt.figure(figsize=(10,6))
df['checkingstatus1'].value_counts().plot(kind='pie',autopct="%1.1f%%")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(df['history'])
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(df['installment'])
plt.show()



plt.figure(figsize=(10,6))
sns.barplot(df['job'])
sns.barplot(df['foreign'])
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(df['residence'])
plt.show()


plt.figure(figsize=(10,6))
sns.boxenplot(df['duration'])
plt.show()



df.dtypes



df1 = df.copy()


le = LabelEncoder()

for i in df1:
    df1[i] = le.fit_transform(df1[i])




def sub(df1):
    f,axs = plt.subplots(2,2,figsize=(10,6))
    sns.barplot(x='job',y='Default',ax=axs[0,0],data=df1)
    sns.barplot(x='installment',y='Default',ax=axs[0,1],data=df1)
    sns.scatterplot(x='amount',y='age',ax=axs[1,0],data=df1)
    sns.barplot(x='employ',y='Default',ax=axs[1,1],data=df1)
    plt.show()
    
sub(df1)


X = df.drop('Default',axis=1)
y = df['Default']




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)


ohe = OneHotEncoder()
sc = StandardScaler()



ct = make_column_transformer(
    (ohe,X.select_dtypes(include='object').columns),
    (sc,X.select_dtypes(include=['int64','float64']).columns),remainder='passthrough')



ct.fit_transform(X)



models = {
    "LogisticRegression":LogisticRegression(),
    "RandomForestClassifier":RandomForestClassifier(),
    "GrdientBoostingClassifier":GradientBoostingClassifier(),
    "BaggingClassifier":BaggingClassifier(),
    "Knneighbors":KNeighborsClassifier()
}


def evaluate_models(model, X_train, X_test, y_train, y_test):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    
    return model.__class__.__name__, acc * 100, roc * 100, f1 * 100


for name, model in models.items():
    model_name, acc, roc, f1 = evaluate_models(model, X_train, X_test, y_train, y_test)
    print(f'{model_name} -- Accuracy: {acc:.2f}%, ROC AUC: {roc:.2f}%, F1: {f1:.2f}%')




def plot_roc_curve(models,X_test,y_test):
    plt.figure(figsize=(12,6))
    
    for name, model in models.items():
        pipe = make_pipeline(ct,model).fit(X_train, y_train)
        pred_prob = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, pred_prob)
        plt.plot(fpr, tpr, label=name)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


plot_roc_curve(models, X_test, y_test)


def plot_confusion_matrix(y_test, y_pred, models):
    conmap = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,6))
    sns.heatmap(conmap, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {models}')
    plt.show()




for model_name, model in models.items():

    pipe = make_pipeline(ct,model).fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    

    plot_confusion_matrix(y_test, y_pred, model)






def cross_validation(X,y,models):
    pipe = make_pipeline(ct,models).fit(X,y)
    cv_scores = cross_val_score(pipe, X,y,cv=10,scoring='roc_auc').max()
    print(f'{model.__class__.__name__}, The cv scores using 10-fold-cross validation: {cv_scores*100:.2f}')
    return cv_scores


for model_name,model in models.items():
    cv_scores = cross_validation(X, y,model)
    



lr_random_param_grid = {
    'logisticregression__C': np.logspace(-4,4,9),
    'logisticregression__penalty': ['l1','l2']
}

lr_random_search = RandomizedSearchCV(
    make_pipeline(ct,LogisticRegression()),
    lr_random_param_grid,
    scoring='roc_auc',
    cv=10
).fit(X_train, y_train)


# Best Logistic Regression model
best_lr_estimator = lr_random_search.best_estimator_
lr_random_pred = best_lr_estimator.predict(X_test)
lr_random_pred_prob = best_lr_estimator.predict_proba(X_test)[:,1]
print(f'Logistic Regression - Best Parameters: {lr_random_search.best_params_}')
print(f'Logistic Regression - Best ROC-AUC Score: {lr_random_search.best_score_ * 100:.2f}%')




rfc_param_grid = {
    'randomforestclassifier__n_estimators': [50,100,200],
    'randomforestclassifier__max_depth': [None,10,20],
    'randomforestclassifier__min_samples_split': [2,5,10],
    'randomforestclassifier__min_samples_leaf': [1,2,4]
}

rfc_random_search = RandomizedSearchCV(
    make_pipeline(ct,RandomForestClassifier()),
    rfc_param_grid,
    n_iter=10,
    scoring='roc_auc',
    cv=10
).fit(X_train, y_train)

best_rfc_estimator = rfc_random_search.best_estimator_
rfc_random_pred = best_rfc_estimator.predict(X_test)
rfc_random_pred_prob = best_rfc_estimator.predict_proba(X_test)[:, 1]
print(f'Random Forest - Best Parameters: {rfc_random_search.best_params_}')
print(f'Random Forest - Best ROC-AUC Score: {rfc_random_search.best_score_ * 100:.2f}')




gbc_random_param_grid = {
    'gradientboostingclassifier__n_estimators': [50,100,200],
    'gradientboostingclassifier__learning_rate': [0.01,0.1,0.2],
    'gradientboostingclassifier__max_depth': [3,4,5],
    'gradientboostingclassifier__min_samples_split': [2,5,10],
}
gbc_random_search = RandomizedSearchCV(
    make_pipeline(ct,GradientBoostingClassifier()),
    gbc_random_param_grid,
    n_iter=10,
    scoring='roc_auc',
    cv=10,
).fit(X_train,y_train)



knn_param_grid = {
    'kneighborsclassifier__n_neighbors': [3,5,7,10],
    'kneighborsclassifier__weights': ['uniform','distance'],
    'kneighborsclassifier__p': [1,2]
}

knn_grid_search = RandomizedSearchCV(
    make_pipeline(ct,KNeighborsClassifier()),
    knn_param_grid,
    scoring='roc_auc',
    cv=10).fit(X_train, y_train)


best_knn_estimator = knn_grid_search.best_estimator_
knn_grid_pred = best_knn_estimator.predict(X_test)
knn_grid_pred_prob = best_knn_estimator.predict_proba(X_test)[:,1]
print(f'K-Nearest Neighbors - Best Parameters: {knn_grid_search.best_params_}')
print(f'K-Nearest Neighbors - Best ROC-AUC Score: {knn_grid_search.best_score_ * 100:.2f}%')



BC_param_grid = {
    'baggingclassifier__n_estimators': [50,100,200],
    'baggingclassifier__max_samples': [1.0,0.8,0.6],
    'baggingclassifier__max_features': [1.0,0.8,0.6],
}




BC_rand_search = RandomizedSearchCV(
    make_pipeline(ct,BaggingClassifier()),
    BC_param_grid,
    n_iter=10,
    scoring='roc_auc',
    cv=10
).fit(X_train,y_train)

best_BC_estimator = BC_rand_search.best_estimator_
BC_rand_pred = best_BC_estimator.predict(X_test)
BC_rand_pred_prob = best_BC_estimator.predict_proba(X_test)[::,1]
print(f'Bagging Classifier - Best Parameters: {BC_rand_search.best_params_}')
print(f'Bagging Classifier - Best ROC-AUC Score: {BC_rand_search.best_score_ * 100:.2f}%')






