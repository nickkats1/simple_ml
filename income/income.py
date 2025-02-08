import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("C:/regular/income/income.csv",delimiter=',')
df.describe()

df.columns = df.columns.str.replace(".","_")

print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.dtypes)
print(df.nunique())
df.head(10)



plt.figure(figsize=(10,6))
sns.countplot(df['workclass'])
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(df['income'])
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(df['race'])
plt.yticks(rotation=45)
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(df['relationship'])
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(df['marital_status'])
plt.show()


df1 = df.copy()
le = LabelEncoder()

for i in df1:
    df1[i] = le.fit_transform(df1[i])


plt.figure(figsize=(12,5))
sns.heatmap(df1.corr(),fmt='f',annot=True,cmap="coolwarm")
plt.show()


df['income'] = [1 if X == ">50K" else 0 for X in df['income']]


fig,axs = plt.subplots(2,2,figsize=(10,5))
sns.barplot(x='sex',y='hours_per_week',ax=axs[0,0],data=df)
sns.countplot(x='race',ax=axs[0,1],data=df1,hue='income')
sns.countplot(x='income',ax=axs[1,0],data=df1,hue='sex')
sns.countplot(x='occupation',ax=axs[1,1],data=df1,hue='income')
plt.show()





X = df.drop("income",axis=1)
y = df['income']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



ct = make_column_transformer(
    (OneHotEncoder(sparse_output=False),X.select_dtypes(include='object').columns),
    (StandardScaler(),X.select_dtypes(include=['float32','int64']).columns),remainder='passthrough')

ct.fit_transform(X)

models = {
    "LogisticRegression":LogisticRegression(),
    "RandomForestClassifier":RandomForestClassifier(),
    "GradientBosstingClassifier":GradientBoostingClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "KNneighbors":KNeighborsClassifier()
    }

def evaluate_models(X_train,X_test,y_train,y_test,model):
    pipe = make_pipeline(ct,model).fit(X_train,y_train)
    pred = pipe.predict(X_test)
    pred_prob = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test,pred)
    roc = roc_auc_score(y_test,pred_prob)
    print(f'{model.__class__.__name__}; --Accuracy Score-- {acc*100:.2f}%; --ROC-- {roc*100:.2f}%')
    return pred,pred_prob


for model_name,model in models.items():
    print(f'results: {model_name}')
    pred,pred_prob = evaluate_models(X_train, X_test, y_train, y_test, model)




def plot_roc_curve(models, X_test, y_test):
    plt.figure(figsize=(12,6))
    
    for model_name, model in models.items():
        pipe = make_pipeline(ct,model).fit(X,y)
        pred_prob = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, pred_prob)
        plt.plot(fpr, tpr, label=model_name)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


plot_roc_curve(models, X_test, y_test)



def plot_confusion_matrix(y_test, y_pred, models):
    conmap = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,6))
    sns.heatmap(conmap,annot=True,fmt='d',cmap='coolwarm')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {models}')
    plt.show()




for model_name, model in models.items():

    pipe = make_pipeline(ct,model).fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    

    plot_confusion_matrix(y_test, y_pred, model)



def cv_scores(X,y,model):
    pipe = make_pipeline(ct,model).fit(X,y)
    cv_scores = cross_val_score(pipe, X,y,cv=10,scoring='roc_auc').max()
    print(f'{model.__class__.__name__}, --Ten Fold Cross-validation scores-- {cv_scores*100:.2f}%')
    return cv_scores


for model_name,model in models.items():
    print("Results from 10-fold cross-validation: ")
    scores = cv_scores(X, y, model)

















