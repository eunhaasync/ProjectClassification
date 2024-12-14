import numpy as np
import pandas as pd
X = np.array([[10, 7, 2], [6, 5, 4], [2, 3, 6], [8, 0, 10]])
df=pd.DataFrame(X, columns=['Fitur 1', 'Fitur 2', 'Fitur 3'])
display(df)

X_mean = np.mean(X)
X_std = np.std(X)
X_norm = (X - X_mean) / X_std
df_norm = pd.DataFrame(X_norm, columns=['Fitur 1', 'Fitur 2', 'Fitur 3'])
display(df_norm)

N = X.shape[0]
C = np.dot(X_norm.T, X_norm) / (N - 1)
df_C = pd.DataFrame(C)
display(df_C)
nilai, vektor = np.linalg.eig(C)
print("Nilai eigen, lambda = ")
print(nilai)
print("\nVektor eigen, v = ")
print(vektor)

indeks_nilai = np.argsort(nilai)[::-1]
nilai_urut = nilai[indeks_nilai]
vektor_urut = vektor[:, indeks_nilai]

print("\nNilai eigen, lambda = ")
print(nilai_urut)
print("\nVektor eigen, v = ")
print(vektor_urut)
t = 0.6
k = int(np.floor(t*N))
v = vektor_urut[:,0:k]
print("Matriks komponen utama, v = ")
df_v = pd.DataFrame(v, columns=['KU_'+str(i) for i in range(1,k+1)])
display(df_v)

X_pca = np.dot(X_norm, v)
print("Dataset yang telah direduksi menjadi {} komponen utama adalah = ".format(k))
df_X_pca = pd.DataFrame(X_pca, columns=['KU_'+str(i) for i in range(1,k+1)])
display(df_X_pca)