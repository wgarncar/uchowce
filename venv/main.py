import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    data1 = pd.read_csv("data1")
    X = data1.iloc[:, [1, 2, 3, 4, 5, 6, 7]].values #pominąłem płeć i ilość ringow (płeć, bo char pierdolił trochę zabawe, trzeba będzie zamienić chara na inta później)
    Y = data1.iloc[:, [8]].values #ilość ringów to nasz oczekiwany wynik

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #ostanie 0.2 calych danych to dane do testów

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = Sequential() #stworzenie sieci neuronowej
    model.add(Dense(output_dim=10, init='uniform', activation='relu', input_dim=7)) #dodanie warstwy wejsiowej - 7 elementów, oraz na wyjsciu (1 ukryta) - 10 elementów
    #model.add(Dropout(0.5))
    #model.add(Dense(output_dim=5, init='uniform', activation='relu')) #dodatkowa warstwa
    #model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, init='uniform', activation='sigmoid')) # warstwa wyjściowa 1 neuron, funkcja sigmoidalna

    #ustawienia sieci
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(X_train, Y_train/100, batch_size=10, epochs=100) #uczenie sieci na danych treningowych

    scores = model.predict(X_test, batch_size=None, verbose=0, steps=None) #przewidywanie wyniku dla danych testowych
    #print("Wiek oczekiwany: \n{} \nWiek otrzymany: \n{}".format(Y_test, scores*100))

    licz = 0
    for i in range(len(X_test)):
        #n = scores[i]*100
        #p = round(n)
        #print("{}, {}".format(n, p))

        if(Y_test[i] == int(scores[i]*100)):
            licz += 1

    print("Trafnych wyników: ", licz)
