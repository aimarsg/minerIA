import csv
import sys

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words
             if word not in cachedStopWords]

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
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='',
                                                                     encoding='utf-8') as outfile:
            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)

            for row in csv_reader:
                if len(row) > 0:
                    text = ' '.join(row)
                    tokens = tokenize(text)
                    csv_writer.writerow(tokens)

        print(f"Tokens procesados y guardados en '{output_file}'")
    except FileNotFoundError:
        print(f"No se encontró el archivo '{input_file}'")


if __name__ == "__main__":
    main()