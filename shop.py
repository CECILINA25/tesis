import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
#algortimos
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#importar matriz de confusion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV


shop=pd.read_csv('intention.csv')


font = {'size': 12}
plt.rc('font', **font)

# sumammos filas
shop.isna().sum()
#sumamos las columnas
shop.isna().sum(axis=1)
#total de set de datos
shop.isna().sum().sum()

print(shop['VisitorType'].unique())
shop.info()
shop.isna().sum()
shop.describe()
m=shop.describe()
m2=m.rename(columns={'Administrative_Duration':'A Duration','Informational_Duration':'I Duration','ProductRelated_Duration':'P Duration' })
                               
shop.isnull().sum()

x=shop.groupby(['VisitorType', 'Revenue'])['Revenue'].count()
ax=sns.countplot(x='VisitorType', hue='Revenue', palette='Set1', data=shop)
ax.set(title='Estado del pasajero (compro/no compro) dado a la clase a la que pertenecia', 
       xlabel='Nuevo visitador', ylabel='Total')
plt.show()

shop.groupby(['Month', 'Revenue'])['Revenue'].count()


shop.groupby(['Month', 'Revenue'])['Revenue'].count()
ax=sns.countplot(x='Month', hue='Revenue', palette='Set1', data=shop)
ax.set(title='Mes de compras efectivas', 
       xlabel='Mes', ylabel='Total')
plt.show()
shop['Revenue'].value_counts(normalize=True)


# ANALISIS GRAFICO DE VARIABLES
columnas=['Administrative', 'Informational', 'SpecialDay','OperatingSystems', 'Browser', 'Region',
             'TrafficType', 'Weekend', 'VisitorType']

df=shop[columnas]
for i, col in enumerate(df.columns):
    plt.figure(i)
    sns.countplot(x=col, data=df)

# analisis variable week
shop['Weekend'].value_counts()
plt.rcParams['figure.figsize'] = (18, 7)
size = [9462, 3868 ]
colors = ['pink', 'magenta', 'yelow']
labels = "False", "True"
explode = [0, 0]
plt.subplot(1, 2, 1)
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Weekend', fontsize = 30)
plt.axis('off')
plt.legend()

#ANALISIS DE VARIABLE REGION
print(shop['Region'].unique())
shop['Region'].value_counts()


size = [4780, 2403, 1182, 1136 ,805, 761, 511, 434,318]
colors = ['pink','orange', 'yellow', 'pink', 'crimson', 'lightgreen', 'cyan', 'blue','magenta']
labels = "1", "3","4","2","6","7","9","8","5"

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle = 90)
plt.title('Region', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()

#analissi traffic type
df2 = pd.crosstab(shop['TrafficType'], shop['Revenue'])
df2.div(df2.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['blue', 'red'])
plt.title('Traffic Type vs Revenue', fontsize = 30)
plt.show()

#revenue
df3= pd.crosstab(shop['VisitorType'], shop['Revenue'])
df3.div(df3.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['blue', 'red'])
plt.title('Visitor Type vs Revenue', fontsize = 30)
plt.show()


df4= pd.crosstab(shop['Region'], shop['Revenue'])
df4.div(df4.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['blue', 'red'])
plt.title('Region vs Revenue', fontsize = 30)
plt.show()




# VARIABLE SPECIALDAY
shop['SpecialDay'].value_counts()
size = [11079, 351, 325, 243 ,178, 154]
colors = ['pink','orange', 'yellow', 'crimson', 'lightgreen', 'cyan','magenta']
labels = "0.0", "0.6","0.8","0.4","0.2","1.0"

plt.subplot(1, 2, 2)
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle = 90)
plt.title('SpecialDay', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()





#CONVERTIR A NUMERICOS VARIABLES BOOLEANAS
shop_numeric=shop.copy()
#CONTAR VALORES FALSOS Y VERDADEROS
shop['Weekend'].value_counts()

#contamos los valores por mes
shop.groupby(['Month']).count()
#remplazamos los valores por verdadesp y falto 
shop_numeric['Weekend'].replace([True,False],[1,0], inplace=True)
shop_numeric['Revenue'].replace([True,False],[1,0], inplace=True)

#CONVERTIR MESES A DATOS NUMERICOS
shop_numeric['Month'].replace([ 'Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
,[2,3,5,6,7,8,9,10,11,12], inplace=True)
# verificamos la cantidad por mes
shop_numeric.groupby(['Month']).count()
shop_numeric.info()
# convertir varible tipo de visitador
date_numeric=shop_numeric.drop(["VisitorType"], axis=1)
# cambiamos nombre para poder visaulizar mejor la tabla
#date_numeric=date_numeric.rename(columns={'Administrative_Duration':'A Duration','Informational_Duration':'I Duration','ProductRelated_Duration':'P Duration' })
date_categorica=shop_numeric.filter(["VisitorType"])
# vamos a usa la variable dummie
cat_numerical=pd.get_dummies(date_categorica.iloc[:,0], drop_first=False)
date=pd.concat([cat_numerical, date_numeric], axis=1)
#guardamos el archivo de lso datos transformados
date.to_csv('shop_total.csv', sep=';')
#leemos el archivo guardado
shop_total=pd.read_csv('shop_total.csv', sep=';')
shop_total=shop_total.drop(['Unnamed: 0'], axis=1)
shop_total.info()
#graficar todas las varibles
shop_total.hist(figsize=(15,13))
corr_s=shop_total.corr(method="spearman")
corr_k=shop_total.corr(method="kendall")
corr_s["Revenue"].sort_values(ascending=False)
corr_k["Revenue"].sort_values(ascending=False)

plt.figure(figsize=(20,10))
sns.heatmap(corr_s,annot=True,center=1,robust=True)

#Grafica de correlaciones

scatter_matrix(shop_numeric, figsize=(14,10))
plt.show()

#scaling

y = shop_total['Revenue'].copy()
X = shop_total.drop('Revenue', axis=1)



# DIVIDIR EN ENTRENAR Y TESTEO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
# scalador 
scaler = StandardScaler()
X_train_escalado= scaler.fit_transform(X_train)

# instaciar
forest_clf = RandomForestClassifier()
knn_clf = KNeighborsClassifier()
lr_clf= LogisticRegression()
nv_clf= GaussianNB()

# entrenamiento
forest_clf.fit(X_train,y_train)
knn_clf.fit(X_train,y_train)
lr_clf.fit(X_train,y_train)
nv_clf.fit(X_train,y_train)

#calcular las predicciones de cada modelo
y_train_prediccion_forest = cross_val_predict(forest_clf, X_train, y_train, cv=5)
y_train_prediccion_knn = cross_val_predict(knn_clf, X_train, y_train, cv=5)
y_train_prediccion_lr = cross_val_predict(lr_clf, X_train, y_train, cv=5)
y_train_prediccion_nv = cross_val_predict(nv_clf, X_train, y_train, cv=5)


#calculamos la matriz de confusion para cada modelo

confusion_matrix(y_train, y_train_prediccion_forest)
confusion_matrix(y_train, y_train_prediccion_knn)
confusion_matrix(y_train, y_train_prediccion_lr)
confusion_matrix(y_train, y_train_prediccion_nv)

recall_score(y_train, y_train_prediccion_forest)
#RESULTADO Out[13]:  0.5710577547047372
recall_score(y_train, y_train_prediccion_knn)
#RESULTADO 0.2809863724853991
recall_score(y_train, y_train_prediccion_lr)
#RESULTADAO0. 0.3711875405580792
recall_score(y_train, y_train_prediccion_nv)
#RESULTADO  0.5412070084360805


param_grid ={'n_estimators': [1, 10, 100, 1000], 'criterion': ['gini', 'entropy'], 'max_depth':[None,2,5,50,200],'min_samples_split':[0.1,2,3,4]}

cuadricula = GridSearchCV(forest_clf, param_grid, return_train_score=True, scoring='recall', cv=5)

cuadricula.fit(X_train, y_train) 
cuadricula.best_params_
#{'criterion': 'entropy',
#'max_depth': 200,
#'min_samples_split': 4,
# 'n_estimators': 1000}

cuadricula.best_score_
#0.5788593283738914



lr_clf.get_params()


