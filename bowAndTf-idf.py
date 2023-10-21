import getopt
import os
import sys
import pandas as pd

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.corpora import Dictionary
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.matutils import sparse2full

def main():

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                # He cambiado unicode por str
                return str(x, 'utf-8')
            else:
                return str(x)
        else:
            return str(x)

    ml_dataset = pd.read_csv('output.csv', header=None, names=['User', 'Text', 'Label'])

    print(ml_dataset.head(5))

    ml_dataset = ml_dataset[['User', 'Text', 'Label']]

    text_features = ['Text']

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)


    documentos = ml_dataset[ml_dataset['Text'].notna()]['Text'].tolist()


    tokenizer = RegexpTokenizer(r'\w+')

    def tokenize_documents(documents):
        tokenized_documents = []
        for idx in range(len(documents)):
            # Convert to lowercase and tokenize
            doc = documents[idx].lower()
            tokens = tokenizer.tokenize(doc)
            tokenized_documents.append(tokens)
        return tokenized_documents

    documentos = tokenize_documents(documentos)
    lemmatizer = WordNetLemmatizer()
    documents = [[lemmatizer.lemmatize(token) for token in document] for document in documentos]

    # Crear documentos etiquetados para Doc2Vec
    tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documents)]

    # Doc2Vec
    model = Doc2Vec(vector_size=500, window=5, min_count=1, workers=4, epochs=10)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Obtener vectores para cada documento
    vectors = [model[i] for i in range(len(tagged_data))]
    print(len(vectors))
    print("Longitud del vector:", len(vectors[0]))

    # Crear un DataFrame con los vectores y las columnas correspondientes
    df_doc2vec = pd.DataFrame(vectors)
    df_doc2vec['User'] = ml_dataset['User']
    df_doc2vec['Label'] = ml_dataset['Label']

    # Guardar en un archivo CSV
    df_doc2vec.to_csv('doc2vec_results.csv', index=False)

    def bow_vectorize(document, dictionary):
        bow_vector = dictionary.doc2bow(document)
        dense_vector = sparse2full(bow_vector, len(dictionary))
        return dense_vector

    #BOW
    dictionary = Dictionary(documents)
    bow = [bow_vectorize(doc, dictionary) for doc in documents]
    print('BOW --> ')
    print('Supportive')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(bow))

    # Agregar columnas de usuario y etiqueta
    df_bow = pd.DataFrame(bow)

    df_bow['User'] = ml_dataset['User']
    df_bow['Label'] = ml_dataset['Label']

    # Guardar en un archivo CSV
    df_bow.to_csv('bow_results2.csv', index=False)
"""
    #TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in documentsSupportive])
    print('TF-IDF --> ')
    print('Supportive')
    print('Number of unique tokens: %d' % len(tfidf_vectorizer.get_feature_names()))
    print('Number of documents: %d' % len(tfidf_matrix.toarray()))

    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in documentsIdeation])
    print('TF-IDF --> ')
    print('Ideation')
    print('Number of unique tokens: %d' % len(tfidf_vectorizer.get_feature_names()))
    print('Number of documents: %d' % len(tfidf_matrix.toarray()))

    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in documentsBehavior])
    print('TF-IDF --> ')
    print('Behavior')
    print('Number of unique tokens: %d' % len(tfidf_vectorizer.get_feature_names()))
    print('Number of documents: %d' % len(tfidf_matrix.toarray()))

    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in documentsAttempt])
    print('TF-IDF --> ')
    print('Attempt')
    print('Number of unique tokens: %d' % len(tfidf_vectorizer.get_feature_names()))
    print('Number of documents: %d' % len(tfidf_matrix.toarray()))
"""
if __name__ == "__main__":
    main()
