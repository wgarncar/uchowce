import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    data1 = pd.read_csv("data1")
    data1.head()

    X = data1.iloc[:, [1, 2, 3, 4, 5, 6, 7]].values #pominąłem płeć i ilość ringow (płeć, bo char pierdolił trochę zabawe, trzeba będzie zamienić chara na inta później)
    Y = data1.iloc[:, [8]].values #ilość ringów to nasz oczekiwany wynik

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #ostanie 0.2 calych danych to dane do testów

    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    model = Sequential() #stworzenie sieci neuronowej
    #in
    model.add(Dense(50, init='uniform', activation='relu', input_dim=7)) #dodanie warstwy wejsiowej - 7 elementów, oraz na wyjsciu (1 ukryta) - 10 elementów
    #extra
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(50, init='uniform', activation='relu')) #dodatkowa warstwa
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(50, init='uniform', activation='relu')) #dodatkowa warstwa
    #Out
    model.add(Dense(1, init='uniform', activation='sigmoid')) # warstwa wyjściowa 1 neuron, funkcja sigmoidalna
    model.summary()

    #ustawienia sieci
    model.compile(loss='mean_squared_error', # binary_crossentropy ?
              optimizer='adam')

    #after 3 epochs in a row in which the model doesn’t improve, training will stop
    early_stopping_monitor = EarlyStopping(patience=3)

    #validation split at 0.4, which means that 40% of the training data we provide in the model will be set aside for testing model performance
    model.fit(X, Y/100, batch_size=5, epochs=150, validation_split=0.2, callbacks=[early_stopping_monitor])

    #scores = model.evaluate(X, Y/100, batch_size=10)
    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    #scores = model.predict(X_test, batch_size=None, verbose=0, steps=None) #przewidywanie wyniku dla danych testowych
    #scores = model.evaluate(X,Y);
    #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    predictions = model.predict(X)
    rounded = [round(x[0]*100) for x in predictions]
    #print(rounded)
    #print("Wiek oczekiwany: \n{} \nWiek otrzymany: \n{}".format(Y, scores*100))

    licz = 0
    for i in range(len(rounded)):
        n = Y[i]
        p = rounded[i]
        print("{}, {}".format(n, predictions[i]*100))

        #if n == p:
        if n == math.floor(predictions[i]*100) or n == math.ceil(predictions[i]*100): #hehe wiecej procent
            licz += 1

    print("Trafnych wyników: %.2f%%" % (licz/len(rounded)*100))
