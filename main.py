import numpy as np
import pymssql
import pandas as pd
import tensorflow as tf
from docplex.mp.model import Model

server = 'server'
database = 'AdventureWorks2014'
username = 'your_username'
password = 'your_password'



# Połączenie z bazą danych MS Azure
conn = pymssql.connect(server=server, user=username, password=password, database=database)

# Pobranie danych z bazy danych
query = '''
        SELECT h.OrderDate, d.OrderQty, d.ProductID, d.UnitPrice, i.InventoryQty
        FROM Sales.SalesOrderHeader AS h
        INNER JOIN Sales.SalesOrderDetail AS d ON h.SalesOrderID = d.SalesOrderID
        INNER JOIN Production.ProductInventory AS i ON d.ProductID = i.ProductID
        '''
df = pd.read_sql(query, conn)

# Zamknięcie połączenia z bazą danych
conn.close()

# Konwersja kolumny OrderDate na typ daty
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

# Sortowanie danych według daty
df.sort_values(by='OrderDate', inplace=True)

# Podział danych na treningowe i testowe (70% treningowe, 30% testowe)
train_size = int(len(df) * 0.7)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Normalizacja danych treningowych
train_mean = train_data['OrderQty'].mean()
train_std = train_data['OrderQty'].std()
train_data['OrderQty'] = (train_data['OrderQty'] - train_mean) / train_std

# Normalizacja danych testowych (z wykorzystaniem statystyk danych treningowych)
test_data['OrderQty'] = (test_data['OrderQty'] - train_mean) / train_std

# Funkcja do generowania sekwencji dla sieci neuronowej
def generate_sequences(data, window_size):
    inputs = []
    outputs = []
    for i in range(len(data) - window_size):
        inputs.append(data[i:i+window_size])
        outputs.append(data[i+window_size])
    return np.array(inputs), np.array(outputs)

# Generowanie sekwencji dla danych treningowych i testowych
window_size = 7
train_inputs, train_outputs = generate_sequences(train_data['OrderQty'].values, window_size)
test_inputs, test_outputs = generate_sequences(test_data['OrderQty'].values, window_size)

# Implementacja sieci neuronowej NNAR


class NNAR(tf.keras.Model):
    def __init__(self, window_size):
        super(NNAR, self).__init__()
        self.window_size = window_size
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, self.window_size])
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

# Inicjalizacja i kompilacja modelu NNAR
model = NNAR(window_size)
model.compile(optimizer='adam', loss='mse')

# Trening modelu
model.fit(train_inputs, train_outputs, epochs=50, batch_size=32)

# Predykcja na danych testowych
predicted_values = model.predict(test_inputs)

# Odtworzenie oryginalnej skali danych
predicted_values = (predicted_values * train_std) + train_mean

# Przygotowanie wyników
results = test_data.iloc[window_size:].copy()
results['PredictedOrderQty'] = predicted_values
results["OrderQty"] = df['OrderQty ']

# Wyświetlenie wyników
print(results[['OrderDate', 'ProductID', 'OrderQty', 'PredictedOrderQty']])


#ZPD
MAX_COST = 10000

# Tworzenie instancji modelu
model = Model(name="Minimalny koszt zakupów")

# Tworzenie zmiennych decyzyjnych
variables = {}
for idx, row in df.iterrows():
    inventory_qty = row['InventoryQty']
    order_qty = row['OrderQty']
    product_id = row['ProductID']
    unit_price = row['UnitPrice']
    variables[product_id] = model.integer_var(name="x{}".format(product_id))

for idx, row in df.iterrows():
    inventory_qty = row['InventoryQty']
    order_qty = row['OrderQty']
    product_id = row['ProductID']
    unit_price = row['UnitPrice']
    model.add_constraint(variables[product_id] * (results['PredictedOrderQty'][product_id] - inventory_qty[product_id]) >= 0,
                         ctname="min_order_quantity_{}".format(product_id))

# Definiowanie funkcji celu
cost_expression = model.sum(variables[product_id] * (results['PredictedOrderQty'][product_id] - inventory_qty[product_id]) * unit_price[product_id]
                            for product_id, variable in variables.items())
model.add_kpi(cost_expression, "Koszt")

# Dodawanie ograniczeń dotyczących minimalnego i maksymalnego kosztu
model.add_constraint(cost_expression <= MAX_COST, ctname="max_cost")

# Minimalizacja kosztu
model.minimize(cost_expression)

# Rozwiązywanie modelu
solution = model.solve()

# Wyświetlanie wyników
print("Status: ", solution.solve_status)
print("Minimalny koszt: ", solution.get_value("Koszt"))
print("Wartości zmiennych decyzyjnych:")
for product_id, variable in variables.items():
    print("x{}: {}".format(product_id, solution.get_value(variable)))
