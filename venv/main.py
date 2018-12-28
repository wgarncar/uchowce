import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    data1 = pd.read_csv("data1")
    X = data1.iloc[:, [1, 2, 3, 4, 5, 6, 7]].values
    Y = data1.iloc[:, [8]].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = Sequential()
    model.add(Dense(output_dim=10, init='uniform', activation='relu', input_dim=7))
    model.add(Dense(output_dim=6, init='uniform', activation='relu'))
    model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X_train, Y_train, batch_size=10, epochs=100)

    scores = model.evaluate(X_test, Y_test)
    print("Wiek oczekiwany: {}, wiek otrzymany: {}".format(Y_test, scores))


