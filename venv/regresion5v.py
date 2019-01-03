import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LinearRegression
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

    X = data1.iloc[:, [0, 1, 2, 3, 4]].values  # pominąłem płeć i ilość ringow (płeć, bo char pierdolił trochę zabawe, trzeba będzie zamienić chara na inta później)
    Y = data1.iloc[:, [8]].values  # ilość ringów to nasz oczekiwany wynik

    for i in range(4177):
        print(X[i][0])
        if X[i][0] == "M":
            X[i][0] = 1
        elif X[i][0] == 'F':
            X[i][0] = 2
        elif X[i][0] == 'I':
            X[i][0] = 3

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) #ostanie 0.3 calych danych to dane do testów

    feature_train = pd.DataFrame(data=X_train)
    feature_test = pd.DataFrame(data=X_test)

    regresja = LinearRegression()
    regresja.fit(feature_train, Y_train)

    predykcja_na_treningowym = regresja.predict(feature_train)
    predykcja_na_testowym = regresja.predict(feature_test)

    print(predykcja_na_testowym)
    rounded = [round(x[0]) for x in predykcja_na_testowym]
    #print(rounded)
    #print("Wiek oczekiwany: \n{} \nWiek otrzymany: \n{}".format(Y, scores*100))

    licz = 0
    max = 40
    tab1 = [0] * max
    tab2 = [0] * max
    licznik = [0] * max
    bladsr = 0
    for i in range(max):
        licznik[i] = i
    for i in range(len(rounded)):
        n = Y_test[i]
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
    plt.savefig('regresion5v_plotY+P.pdf')
    plt.close()


    plt.figure(figsize=(25, 25))
    plt.plot(licznik, tab1)
    plt.plot(licznik, tab2)
    plt.title('Porównanie wyników')
    plt.savefig('regresion5v_plotBoth.pdf')
    plt.close()


