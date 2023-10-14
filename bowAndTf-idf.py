import getopt
import os
import sys
import pandas as pd

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

    """
    dfSupportive = ml_dataset[ml_dataset['Label'] == 'Supportive']
    dfIdeation = ml_dataset[ml_dataset['Label'] == 'Ideation']
    dfBehavior = ml_dataset[ml_dataset['Label'] == 'Behavior']
    dfAttempt = ml_dataset[ml_dataset['Label'] == 'Attempt']
    """

    documentos = ml_dataset[ml_dataset['Text'].notna()]['Text'].tolist()

    """# GENERAR DOCUMENTOS
    documentsSupportive = dfSupportive[dfSupportive['Text'].notna()]['Text'].tolist()
    documentsIdeation = dfIdeation[dfIdeation['Text'].notna()]['Text'].tolist()
    documentsBehavior = dfBehavior[dfBehavior['Text'].notna()]['Text'].tolist()
    documentsAttempt = dfAttempt[dfAttempt['Text'].notna()]['Text'].tolist()
    """

    tokenizer = RegexpTokenizer(r'\w+')

    def tokenize_documents(documents):
        tokenized_documents = []
        for idx in range(len(documents)):
            # Convert to lowercase and tokenize
            doc = documents[idx].lower()
            tokens = tokenizer.tokenize(doc)
            tokenized_documents.append(tokens)
        return tokenized_documents

    """documentsSupportive = tokenize_documents(documentsSupportive)
    documentsIdeation = tokenize_documents(documentsIdeation)
    documentsBehavior = tokenize_documents(documentsBehavior)
    documentsAttempt = tokenize_documents(documentsAttempt)"""
    documentos = tokenize_documents(documentos)
    lemmatizer = WordNetLemmatizer()
    documents = [[lemmatizer.lemmatize(token) for token in document] for document in documentos]

    """
    documentsSupportive = [[lemmatizer.lemmatize(token) for token in document] for document in documentsSupportive]
    documentsIdeation = [[lemmatizer.lemmatize(token) for token in document] for document in documentsIdeation]
    documentsBehavior = [[lemmatizer.lemmatize(token) for token in document] for document in documentsBehavior]
    documentsAttempt = [[lemmatizer.lemmatize(token) for token in document] for document in documentsAttempt]
"""
    def bow_vectorize(document, dictionary):
        bow_vector = dictionary.doc2bow(document)
        dense_vector = sparse2full(bow_vector, len(dictionary))
        return dense_vector

    #BOW
    dictionary = Dictionary(documentos)
    bow = [bow_vectorize(doc, dictionary) for doc in documentos]
    print('BOW --> ')
    print('Supportive')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(bow))
    """
    dictionary_ideation = Dictionary(documentsIdeation)
    bow_ideation = [bow_vectorize(doc, dictionary_ideation) for doc in documentsIdeation]
    print('Ideation:')
    print('Number of unique tokens: %d' % len(dictionary_ideation))
    print('Number of documents: %d' % len(bow_ideation))

    dictionary_behavior = Dictionary(documentsBehavior)
    bow_behavior = [bow_vectorize(doc, dictionary_behavior) for doc in documentsBehavior]
    print('Behavior:')
    print('Number of unique tokens: %d' % len(dictionary_behavior))
    print('Number of documents: %d' % len(bow_behavior))

    dictionary_attempt = Dictionary(documentsAttempt)
    bow_attempt = [bow_vectorize(doc, dictionary_attempt) for doc in documentsAttempt]
    print('Attempt:')
    print('Number of unique tokens: %d' % len(dictionary_attempt))
    print('Number of documents: %d' % len(bow_attempt))

    df_bow_supportive = pd.DataFrame(bow_supportive)
    df_bow_ideation = pd.DataFrame(bow_ideation)
    df_bow_behavior = pd.DataFrame(bow_behavior)
    df_bow_attempt = pd.DataFrame(bow_attempt)
    """
    # Agregar columnas de usuario y etiqueta
    df_bow = pd.DataFrame(bow)

    df_bow['User'] = ml_dataset['User']
    df_bow['Label'] = ml_dataset['Label']

    """    
    df_bow_ideation['User'] = dfIdeation['User']
    df_bow_ideation['Label'] = 'Ideation'

    df_bow_behavior['User'] = dfBehavior['User']
    df_bow_behavior['Label'] = 'Behavior'

    df_bow_attempt['User'] = dfAttempt['User']
    df_bow_attempt['Label'] = 'Attempt'

    # Concatenar todos los DataFrames
    df_bow_all = pd.concat([df_bow_supportive, df_bow_ideation, df_bow_behavior, df_bow_attempt])
    """
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
