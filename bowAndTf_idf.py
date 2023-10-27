import sys
import pandas as pd
import argparse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.matutils import sparse2full

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            # He cambiado unicode por str
            return str(x, 'utf-8')
        else:
            return str(x)
    else:
        return str(x)

def read_csv(file_path):
    ml_dataset = pd.read_csv(file_path, header=None, names=['User', 'Text', 'Label'])
    ml_dataset = ml_dataset[['User', 'Text', 'Label']]

    text_features = ['Text']

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    documentos = ml_dataset[ml_dataset['Text'].notna()]['Text'].tolist()

    return ml_dataset, documentos


def preprocess_text(documents):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    processed_documents = []

    for idx in range(len(documents)):
        # Convertir a minúsculas y tokenizar
        doc = documents[idx].lower()
        tokens = tokenizer.tokenize(doc)

        # Lematizar los tokens
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

        processed_documents.append(lemmatized_tokens)

    return processed_documents

def train_doc2vec_model(documents):
    # Prepare tagged data
    tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documents)]

    # Instantiate Doc2Vec model
    model = Doc2Vec(vector_size=2, window=5, min_count=1, workers=4, epochs=10)

    # Build vocabulary
    model.build_vocab(tagged_data)

    # Train the model on the corpus
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    model.save("d2v.model")

    return model, tagged_data


def add_new_instance(model, new_instance):
    # Inicializar una lista para almacenar los vectores inferidos
    inferred_vectors = []
    # Iterar sobre cada valor en la lista new_instance
    for instance_value in new_instance:
        # Obtener el vector inferido para la instancia actual
        inferred_vector = model.infer_vector(instance_value)

        # Agregar el vector inferido a la lista
        inferred_vectors.append(inferred_vector)

    return inferred_vectors

def save_doc2vec_results(model, tagged_data, ml_dataset):
    # Obtener vectores para cada documento
    vectors = [model[i] for i in range(len(tagged_data))]
    print(len(vectors))
    print("Longitud del vector:", len(vectors[0]))

    # Crear un DataFrame con los vectores y las columnas correspondientes
    df_doc2vec = pd.DataFrame(vectors)
    df_doc2vec['User'] = ml_dataset['User']
    df_doc2vec['Label'] = ml_dataset['Label']

    # Guardar en un archivo CSV
    df_doc2vec.to_csv('doc2vec_results2.csv', index=False)

def bow_vectorize(document, dictionary):
    bow_vector = dictionary.doc2bow(document)
    dense_vector = sparse2full(bow_vector, len(dictionary))
    return dense_vector

def find_text_by_vector(model, vector, ml_dataset):
    # Calcular similitudes de coseno entre el vector dado y los vectores del modelo
    similarities = model.docvecs.most_similar([vector])
    print(similarities)

    # Encontrar el índice del documento más similar
    most_similar_index = int(similarities[0][0])
    # Obtener el texto original asociado al índice
    most_similar_text = ml_dataset.iloc[most_similar_index]['Post']
    print(most_similar_text)

"""
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
    parser = argparse.ArgumentParser(description='Procesar nuevas instancias y actualizar modelo Doc2Vec')
    parser.add_argument('--csv_file', nargs="?", type=str, help='Ruta al archivo CSV de instancias')
    parser.add_argument('--n_instancias', type=str, help='Ruta al archivo CSV de nuevas instancias')
    parser.add_argument('--model', type=str, help='Ruta al archivo CSV de nuevas instancias')
    parser.add_argument('--t_original', type=str, help='Ruta al archivo CSV de nuevas instancias')


    args = parser.parse_args()

    if args.csv_file and not args.model:
        # Leer y preprocesar el archivo CSV
        ml_dataset, documentos = read_csv(args.csv_file)

        documentos = preprocess_text(documentos)

        # Entrenar el modelo Doc2Vec
        model, tagged_data = train_doc2vec_model(documentos)

        save_doc2vec_results(model, tagged_data, ml_dataset)

    if args.n_instancias and args.model:
        #pdb.set_trace()
        # Añadir nueva instancia al modelo Doc2Vec
        ml_dataset, documentos = read_csv(args.n_instancias)
        documentos = preprocess_text(documentos)
        model = Doc2Vec.load(args.model)
        new_vector = add_new_instance(model, documentos)

        print("Nuevo Vector:", new_vector)

        # Guardar el vector en un archivo CSV
        df_new_vector = pd.DataFrame(new_vector)
        df_new_vector['User'] = ml_dataset['User']
        df_new_vector['Label'] = ml_dataset['Label']

        # Guardar en un archivo CSV
        df_new_vector.to_csv('new_vector_result.csv', index=False)

    if args.t_original and args.model and args.csv_file:
        # Cargar el modelo Doc2Vec
        model = Doc2Vec.load(args.model)

        df_ml_dataset = pd.read_csv(args.csv_file)

        # Leer el archivo CSV con vectores y textos originales
        df = pd.read_csv(args.t_original)
        vector = df.columns[:-2]

        # Obtener el vector correspondiente al texto original
        original_vector = df[vector].values[0]

        find_text_by_vector(model, original_vector, df_ml_dataset)
