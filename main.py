import pandas as pd
import classify_conv
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


if __name__ == '__main__':

    df = pd.read_csv('COsnowtotals.csv')
    df = df.iloc[:, 1:]
    df = df[df.snow > 30]
    print(df.snow.max())
    print(df.snow.min())
    # 0-300 every 60 == class 5 classes 1-6 6 = op
    df['snow'] = df.snow.apply(lambda x: classify_conv.classify_convert(x))

    df_train = df.iloc[:, :-1]
    df_test = df.iloc[:, -1]

    scaler = StandardScaler()
    df_train = scaler.fit(df_train)

    x_train, x_test, y_train, y_test = train_test_split(df_train, df_test, test_size=.20)

    clf = OneVsRestClassifier(SVC(kernel='rbf', cache_size=1000))

    C_range = [.001, .01, .1, 1, 10, 100]
    param_grid = {'estimator__C': C_range}
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    grid_search.fit(x_train, y_train)
    print('Best C Params: ', grid_search.best_params_)


    print('check')