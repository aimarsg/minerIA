import pandas as pd
from sklearn.model_selection import train_test_split

# Lee el archivo CSV
df = pd.read_csv('./datos_train2.csv')

# Aplica la división de entrenamiento y prueba
df_train, df_test = train_test_split(df, test_size=0.30, stratify=df['label'])

# Guarda df_train en un nuevo archivo CSV
df_train.to_csv('datos_train3.csv', index=False)

print("Datos de entrenamiento guardados en 'datos_train.csv'")
