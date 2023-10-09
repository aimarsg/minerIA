import csv
import sys
import pandas as pd  # Importa Pandas

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


STOPWORDS = set(stopwords.words('english'))


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words if word not in cachedStopWords]

    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))

    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
    return filtered_tokens

def main():
    if len(sys.argv) != 3:
        print("Uso: python limpiarDatos.py entrada.csv salida.csv")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        # Cargar el archivo CSV en un DataFrame de Pandas
        df = pd.read_csv(input_file, header=None)


        ####### PREPROCESADO #######

        # lowercasing
        df[""] = df[""].str.lower()

        # eliminar stopwords
        df[""] = df[""].apply(lambda text: remove_stopwords(text))

        # stemming
        df[""] = df[""].apply(lambda text: stem_words(text))

        ############################

        # Guardar el DataFrame en un archivo CSV
        df.to_csv(output_file, index=False, header=False)

        print(f"Tokens procesados y guardados en '{output_file}'")
    except FileNotFoundError:
        print(f"No se encontró el archivo '{input_file}'")

if __name__ == "__main__":
    main()
