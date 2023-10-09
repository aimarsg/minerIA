import pandas as pd
from twitter_scraper import get_tweets
import requests
from bs4 import BeautifulSoup
import httpx
from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.common.keys import Keys

# Leer el archivo CSV original
csv_file = "./datos_train3.csv"
df = pd.read_csv(csv_file)


driver = webdriver.Chrome()
# Función para obtener el texto de una URL de Twitter
def obtener_texto_twitter(id):

    url = f"https://twiiit.com/anyuser/status/{id}"
    response = requests.get(url, allow_redirects=True)
    print(response.content)
    print(url)
    if response.status_code == 200:
        cuerpo = response.content
        soup = BeautifulSoup(cuerpo, 'html.parser')
        tweets = soup.find_all('div', {'class': "tweet-content media-body"})
        print(tweets)
        texto = ' '.join([tweet.get_text() for tweet in tweets])
        print(texto)
        return texto
    else:
        return None


# Aplicar la función a cada fila del DataFrame y agregar los resultados a una nueva columna
df['texto_twitter'] = df['id'].apply(obtener_texto_twitter)

# Guardar el DataFrame modificado en un nuevo archivo CSV
csv_file_con_tweets = "./data/dataset_con_tweets.csv"
df.to_csv(csv_file_con_tweets, index=False)

print(f"Se ha creado el archivo '{csv_file_con_tweets}' con los tweets.")