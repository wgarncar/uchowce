import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def maxf(tab):
    m=tab[0]
    for i in tab: #pętla for w Pythonie to bardziej "foreach"
        if i>m:   #i przyjmuje wartości kolejnych komórek tablicy tab
            m=i
    return m

if __name__ == "__main__":

    data1 = pd.read_csv("data1")
    data1.head()

    X = data1.iloc[:, [0, 1, 2, 3, 4, 5, 6,
                       7]].values  # pominąłem płeć i ilość ringow (płeć, bo char pierdolił trochę zabawe, trzeba będzie zamienić chara na inta później)
    Y = data1.iloc[:, [8]].values  # ilość ringów to nasz oczekiwany wynik

    for i in range(4177):
        print(X[i][0])
        if X[i][0] == "M":
            X[i][0] = 1
        elif X[i][0] == 'F':
            X[i][0] = 2
        elif X[i][0] == 'I':
            X[i][0] = 3
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #ostanie 0.2 calych danych to dane do testów

    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.transform(X_test)

    model = Sequential() #stworzenie sieci neuronowej
    #in
    model.add(Dense(100, init='uniform', activation='relu', input_dim=8)) #dodanie warstwy wejsiowej - 7 elementów, oraz na wyjsciu (1 ukryta) - 10 elementów
    #extra
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(100, init='uniform', activation='relu')) #dodatkowa warstwa
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    #model.add(Dense(50, init='uniform', activation='relu')) #dodatkowa warstwa
    #Out
    model.add(Dense(1, init='uniform', activation='sigmoid')) # warstwa wyjściowa 1 neuron, funkcja sigmoidalna
    model.summary()

    #ustawienia sieci
    model.compile(loss='mean_squared_error', # binary_crossentropy ?
              optimizer='adam')

    #after 3 epochs in a row in which the model doesn’t improve, training will stop
    early_stopping_monitor = EarlyStopping(patience=3)

    #validation split at 0.4, which means that 40% of the training data we provide in the model will be set aside for testing model performance
    model.fit(X, Y/100, batch_size=5, epochs=150, validation_split=0.3, callbacks=[early_stopping_monitor])

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
    max = 35
    tab1 = [0] * max
    tab2 = [0] * max
    licznik = [0] * max
    bladsr = 0
    for i in range(max):
        licznik[i] = i
    for i in range(len(rounded)):
        n = Y[i]
        p = rounded[i]
        print("{}, {}".format(n, p))
        blad = n - p
        bladsr += abs(blad)

        if n == p:
        #if n == math.floor(predictions[i]*100) or n == math.ceil(predictions[i]*100): #hehe wiecej procent
            licz += 1
        tmpn = (int)(n)
        tmpp = (int)(p)
        tab1[tmpn] += 1
        tab2[tmpp] += 1

    bladsr = bladsr/len(rounded)

    print("Trafnych wyników: %.2f%%" % (licz/len(rounded)*100))
    print("Błąd średni: ", bladsr)

    plt.figure(figsize=(25, 25))
    plt.subplot(211)
    plt.bar(licznik, tab1)
    plt.title('Oczekiwane wyniki')
    plt.subplot(212)
    plt.bar(licznik, tab2)
    plt.title('Otrzymane wyniki')
    plt.savefig('plotY+P.pdf')
    plt.close()


    plt.figure(figsize=(25, 25))
    plt.plot(licznik, tab1)
    plt.plot(licznik, tab2)
    plt.title('Porównanie wyników')
    plt.savefig('plotBoth.pdf')
    plt.close()


