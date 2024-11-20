import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

# Definir uma seed para garantir reprodutibilidade
SEED = 4
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Carregar o CSV
data = pd.read_csv('df_posicoes.csv')

# Extrair os recursos (features) e os alvos (targets)
X = data.drop(['Best_X', 'Best_Y'], axis=1).values
y = data[['Best_X', 'Best_Y']].values

X_new = X[1:800, :]
y_new = y[1:800, :]

df_saida = pd.read_csv("dados_saida.csv")
y_new1 = np.array(df_saida)

# Dados simulados (substituir pelos seus)
dados_saida = np.random.rand(100, 2)  # Best_X e Best_Y
dados_entrada = np.random.rand(100, 20)  # Pos_X e Pos_Y

# Salvar os dados em arquivos CSV
pd.DataFrame(dados_saida, columns=["Best_X", "Best_Y"]).to_csv("dados1.csv", index=False)
pd.DataFrame(dados_entrada).to_csv("dados2.csv", index=False, header=[f"Pos_{i+1}" for i in range(20)])

print("Dados salvos em arquivos")

# Convertendo a lista de listas para uma matriz numpy 100x20
df_entrada = pd.read_csv("dados_entrada.csv")
X_new1 = np.array(df_entrada)

# Normalizar os dados
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_new_scaled = scaler_X.fit_transform(X_new)
y_new_scaled = scaler_y.fit_transform(y_new)

X_new1_scaled = scaler_X.fit_transform(X_new1)
y_new1_scaled = scaler_y.fit_transform(y_new1)

# Definir o número de instantes de tempo (K)
timesteps = 10

# Ajustar o formato dos dados para (amostras, timesteps, características)
n_samples = X_new_scaled.shape[0] // timesteps
n_samples1 = X_new1_scaled.shape[0] // timesteps

X_new_reshaped = X_new_scaled[:n_samples * timesteps].reshape(n_samples, timesteps, X_new_scaled.shape[1])
X_new1_reshaped = X_new1_scaled[:n_samples1 * timesteps].reshape(n_samples1, timesteps, X_new1_scaled.shape[1])

X_train = X_new_reshaped
y_train = y_new_scaled[:n_samples]

X_test = X_new1_reshaped
y_test = y_new1_scaled[:n_samples1]

# Definir o modelo
def build_model():
    model = Sequential()
    model.add(Input(shape=(timesteps, X_train.shape[2])))  # timesteps x características
    model.add(LSTM(units=128, activation='relu'))  # Apenas uma LSTM
    model.add(Dropout(0.25))  # Apenas uma camada de Dropout
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Configuração da validação cruzada
n_splits = 5  # Número de folds
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Variável para armazenar a perda de cada fold
fold_losses = []

# Loop para validação cruzada
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"\nTreinando fold {fold + 1}/{n_splits}...")

    # Dividir os dados de treino e validação para o fold atual
    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    # Construir e compilar o modelo para cada fold
    model = build_model()

    # Treinar o modelo no fold atual
    history = model.fit(
        X_fold_train, y_fold_train,
        epochs=100,
        validation_data=(X_fold_val, y_fold_val),
        verbose=1
    )

    # Avaliar a perda no conjunto de validação para o fold atual
    val_loss = model.evaluate(X_fold_val, y_fold_val, verbose=0)
    print(f"Perda de validação para fold {fold + 1}: {val_loss}")
    fold_losses.append(val_loss)

# Calcular a média da perda entre os folds
mean_loss = np.mean(fold_losses)
print(f"\nPerda média de validação cruzada: {mean_loss}")

# Avaliar o modelo final no conjunto de teste
test_loss = model.evaluate(X_test, y_test)
print(f'\nPerda no teste: {test_loss}')

# Plotar a perda de treino e validação para o último fold
plt.figure(figsize=(10, 6))  # Aumentar o tamanho da figura
plt.plot(history.history['loss'], label='Perda no treinamento')
plt.plot(history.history['val_loss'], label='Perda na validação')

# Ajuste dos tamanhos das fontes
plt.xlabel('Épocas', fontsize=16)
plt.ylabel('Perda', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

y_pred = model.predict(X_test)

# Comparar previsões e valores reais
print("Primeiras previsões comparadas com valores reais")
for i in range(5):
    print(f"Previsão: {y_pred[i]} | Verdadeiro: {y_test[i]}")
print("...")

print("Últimas previsões comparadas com valores reais")
for i in range(len(y_pred) - 5, len(y_pred)):
    print(f"Previsão: {y_pred[i]} | Verdadeiro: {y_test[i]}")
