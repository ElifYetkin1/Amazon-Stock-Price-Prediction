import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv(r'C:\Users\eliff\Desktop\stoktahmin\AMZN.csv')

df.columns = ['Date','Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']



#Close ve Adj Close sütunlarındaki verilerin hepsi aynı ve fazladan veri makine öğrenmesi algoritmamızda işe yaramayacak bu yüzden siliyoruz.

df = df.drop(['Adj Close'], axis=1)

#veri setimizde null yani boş veirler var mı kontrol edelim

df.isnull().sum()

#grafikler

features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20, 10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sn.distplot(df[col])
plt.show()

#veri setini detaylandırmak için hangi gün ay ve yılın verisi olduğunu da sütun olarak ekliyorum

splitted = df['Date'].str.split('-', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

#Çeyrek, üç aylık bir grup olarak tanımlanır. İnsanların şirketin performansını analiz edebilmesi için her şirket
# üç aylık sonuçlarını hazırlar ve kamuoyuna açıklar. Bu üç aylık sonuçlar hisse senedi fiyatlarını büyük ölçüde
# etkiler, bu nedenle öğrenme modeli için yararlı bir özellik olabileceğinden bu özelliği ekledik.

df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
print(df.head())

df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

#veriyi eğitim ve test olarak ayırıyoruz

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


#Ana öğrenme algoritmasını burada kuruyoruz. Lojistik regresyon ve SVC kullanacağız.
#İki modelin uygulanmasını da tek seferde yapmak için bir for döngüsü kullanıyoruz
#ardından sonuçları görmüş için formatlanmış bir text ile ekrana yazdırıyoruz.

models = [LogisticRegression(), SVC(
    kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, models[i].predict_proba(X_valid)[:, 1]))
    print()

# elimizdeki veriler binary tarzında olmadığı için yani devamlı veriler olduğu için aslında başka bir model deneyerek daha başarılı sonuçlar elde edebiliriz

models = [LogisticRegression(), SVC(
    kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, models[i].predict_proba(X_train)[:, 1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, models[i].predict_proba(X_valid)[:, 1]))
    print()

metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()