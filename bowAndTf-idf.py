import getopt
import os
import sys
import pandas as pd

from gensim.corpora import Dictionary
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    ml_dataset = pd.read_csv('output.csv')

    print(ml_dataset.head(5))

    ml_dataset = ml_dataset[
        ['User', 'Text', 'Label']]

    text_features = ['Text']
    
    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    dfSupportive = ml_dataset[ml_dataset['Label'] == 'Supportive']
    dfIdeation = ml_dataset[ml_dataset['Label'] == 'Ideation']
    dfBehavior = ml_dataset[ml_dataset['Label'] == 'Behavior']
    dfAttempt = ml_dataset[ml_dataset['Label'] == 'Attempt']

    # GENERAR DOCUMENTOS
    documentsSupportive = dfSupportive[dfSupportive['Text'].notna()]['Text'].tolist()
    documentsIdeation = dfIdeation[dfIdeation['Text'].notna()]['Text'].tolist()
    documentsBehavior = dfBehavior[dfBehavior['Text'].notna()]['Text'].tolist()
    documentsAttempt = dfAttempt[dfAttempt['Text'].notna()]['Text'].tolist()

    tokenizer = RegexpTokenizer(r'\w+')

    def tokenize_documents(documents):
        tokenized_documents = []
        for idx in range(len(documents)):
            # Convert to lowercase and tokenize
            doc = documents[idx].lower()
            tokens = tokenizer.tokenize(doc)
            tokenized_documents.append(tokens)
        return tokenized_documents

    documentsSupportive = tokenize_documents(documentsSupportive)
    documentsIdeation = tokenize_documents(documentsIdeation)
    documentsBehavior = tokenize_documents(documentsBehavior)
    documentsAttempt = tokenize_documents(documentsAttempt)

    lemmatizer = WordNetLemmatizer()

    documentsSupportive = [[lemmatizer.lemmatize(token) for token in document] for document in documentsSupportive]
    documentsIdeation = [[lemmatizer.lemmatize(token) for token in document] for document in documentsIdeation]
    documentsBehavior = [[lemmatizer.lemmatize(token) for token in document] for document in documentsBehavior]
    documentsAttempt = [[lemmatizer.lemmatize(token) for token in document] for document in documentsAttempt]

    #BOW
    dictionary = Dictionary(documentsSupportive)
    corpus = [dictionary.doc2bow(doc) for doc in documentsSupportive]
    print('BOW --> ')
    print('Supportive')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    dictionary = Dictionary(documentsIdeation)
    corpus = [dictionary.doc2bow(doc) for doc in documentsIdeation]
    print('Ideation:')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    dictionary = Dictionary(documentsBehavior)
    corpus = [dictionary.doc2bow(doc) for doc in documentsBehavior]
    print('Behavior:')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    dictionary = Dictionary(documentsAttempt)
    corpus = [dictionary.doc2bow(doc) for doc in documentsAttempt]
    print('Attempt:')
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

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

if __name__ == "__main__":
    main()
