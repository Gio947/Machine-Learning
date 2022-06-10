import matplotlib.pyplot as plt
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#metodo per la regressione lineare
def linearRegression(x_train, x_test, y_train, y_test):
    print("\n")
    print("-----------------------------Regressione lineare-----------------------")

    regr_lin = linear_model.LinearRegression()
    regr_lin.fit(x_train, y_train)

    y_pred_lin_train = regr_lin.predict(x_train)
    y_pred_lin_test = regr_lin.predict(x_test)

    # valutiamo il training set
    rmse_lin_train = sqrt(mean_squared_error(y_train, y_pred_lin_train))
    r2_lin_train = r2_score(y_train, y_pred_lin_train)
    print("rmse in fase di training : " , rmse_lin_train)
    print("score in fase di training : " ,r2_lin_train)

    # valutiamo il test set
    rmse_lin_test = sqrt(mean_squared_error(y_test, y_pred_lin_test))
    r2_lin_test = r2_score(y_test, y_pred_lin_test)

    print("rmse in fase di testing : ",rmse_lin_test)
    print("score in fase di testing : " ,r2_lin_test)

    print("Valore di intercept : " , regr_lin.intercept_)
    print("Coefficienti : " , regr_lin.coef_)

#metodo per la regressione polinomiale
def polynomialRegression(grado , x_train, x_test, y_train, y_test):
    print("\n")
    print("------------------Regressione polinomiale di grado " , grado)

    poly = PolynomialFeatures(degree=grado, include_bias=True)

    x_train_trans = poly.fit_transform(x_train)
    x_test_trans = poly.fit_transform(x_test)

    lr = LinearRegression()
    lr.fit(x_train_trans, y_train)

    y_pred_train = lr.predict(x_train_trans)
    y_pred_test = lr.predict(x_test_trans)
    # lr = linear_model.Ridge(alpha=5.)

    # valutiamo il modello sul training set
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    print("rmse in fase di training : " , rmse_train)
    print("score in fase di training : " , r2_train)

    # valutiamo il modello sul test set
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = sqrt(mean_squared_error(y_test, y_pred_test))
    print("rmse in fase di testing : ", rmse_train)
    print("score in fase di testing: ", r2_test)

    print("valore di intercept : " , lr.intercept_)

    print("Valori predetti in fase di testing :")
    print(y_pred_test)
    print("Differenza tra valori corretti e valori predetti :")
    print(y_test - y_pred_test)

#metodo per il Random Forest
def randomForestRegression( x_train, x_test, y_train, y_test) :
    print("\n")
    print("----------------------------Random forest-----------------------")

    regressor = RandomForestRegressor(random_state=0)
    #griglia di parametri per la ricerca
    param_grid = {
        'n_estimators': [50, 70, 100],
        'max_depth': [4, 6, 8],
        'min_samples_leaf' : [1 , 2 , 3 , 6]

    }
    CV_rfc = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=4)
    CV_rfc.fit(x_train, y_train)

    print("parametri migliori : " , CV_rfc.best_params_)
    print("score migliore : " ,CV_rfc.best_score_)

    #istanzio un nuovo modello a cui passo i parametri migliori
    regressor1 = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=6, random_state=0)
    regressor1.fit(x_train, y_train)

    y_pred_train = regressor1.predict(x_train)
    y_pred_test = regressor1.predict(x_test)

    #valutiamo il modello in fase di training
    rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
    score_train = r2_score(y_train, y_pred_train)

    print("rmse in fase di training :", rmse_train)
    print("score in fase di training : ", score_train)

    # valutiamo il modello in fase di testing
    rmse_test = sqrt(mean_squared_error(y_test, y_pred_test))
    score_test = r2_score(y_test, y_pred_test)

    print("rmse in fase di test :", rmse_test)
    print("score in fase di test : ", score_test)

    #creo il png con l'albero decisionale dell'estimatore numero zero
    featureNames = ['court_surface', 'opponent_name', 'player_name', 'prize_money', 'round', 'tournament']
    targetName = ['total_points']
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
    tree.plot_tree(regressor1.estimators_[0], feature_names=featureNames, class_names=targetName, filled=True)
    fig.savefig('treeEstimatore0.png')

#metodo a cui passo le colonne per fare il grafico a dispersione per visualizzare la loro correlazione
def plotChartColumns(column1 , column2, titleColumn2) :
    plt.scatter(column1, column2, color='red')
    plt.title(titleColumn2 + ' vs total_points', fontsize=14)
    plt.xlabel('total_points', fontsize=14)
    plt.ylabel(titleColumn2, fontsize=14)
    plt.grid(True)
    plt.show()

def mainFunction():

    dataset = pd.read_excel('Dataset2.xlsx', sheet_name="Foglio2", engine='openpyxl')

    #trasformo le varibili da stringhe in numeri
    categorical_columns = ['court_surface', 'opponent_name', 'player_name', 'round', 'tournament']

    for column in categorical_columns:
        dataset[column] = pd.factorize(dataset[column])[0]

    #lista delle features
    x_column = ['court_surface','opponent_name' ,'player_name','prize_money' ,'round', 'tournament']

    x_train, x_test, y_train, y_test = train_test_split(dataset[x_column], dataset['total_points'], test_size=0.30)

    #correlazione variabili con metodo di Pearson
    print("Indice di correlazione tra total_points e tournament : " , dataset['total_points'].corr(dataset['tournament'],method='pearson',))
    print("Indice di correlazione tra total_points e court_surface : " ,dataset['total_points'].corr(dataset['court_surface'],method='pearson',))
    print("Indice di correlazione tra total_points e prize_money : " ,dataset['total_points'].corr(dataset['prize_money'],method='pearson',))
    print("Indice di correlazione tra total_points e round : " ,dataset['total_points'].corr(dataset['round'],method='pearson',))
    print("Indice di correlazione tra total_points e player_name : " ,dataset['total_points'].corr(dataset['player_name'],method='pearson',))
    print("Indice di correlazione tra total_points e opponent_name : " ,dataset['total_points'].corr(dataset['opponent_name'],method='pearson',))

    plotChartColumns(dataset['total_points'], dataset['tournament'], 'tournament')
    plotChartColumns(dataset['total_points'], dataset['court_surface'], 'court_surface')
    plotChartColumns(dataset['total_points'], dataset['prize_money'], 'prize_money')
    plotChartColumns(dataset['total_points'], dataset['round'], 'round')
    plotChartColumns(dataset['total_points'], dataset['player_name'], 'player_name')
    plotChartColumns(dataset['total_points'], dataset['opponent_name'], 'opponent_name')

    #stampo delle informazioni sul dataset
    print(dataset.sample(20))
    print(dataset.sample(20).corr(method='pearson'))
    print(dataset.describe())

    #chiamo le funzioni dei miei modelli
    linearRegression(x_train, x_test, y_train, y_test)
    polynomialRegression(7, x_train, x_test, y_train, y_test)
    randomForestRegression(x_train, x_test, y_train, y_test)



if __name__ == '__main__':
    mainFunction()


