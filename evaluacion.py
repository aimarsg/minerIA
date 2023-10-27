import argparse
import pickle
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from mpl_toolkits.mplot3d import Axes3D

num_instancias = 500

def obtenerLabels(fichero):
    global num_instancias
    with open(fichero, 'r') as file:
        lines = file.readlines()
    # inicializar los labels
    num_instancias = int(lines[0].split(':')[1])
    labels = np.arange(num_instancias)
    i = 0
    for line in lines[1:]:
        cols = line.split(';')
        cluster_n = int(cols[0].split(':')[1])
        cluster = eval(cols[2].split(':')[1])
        for instancia in cluster:
            labels[instancia] = i
        i += 1

    return labels


def leer_etiquetas_reales(fich_etiquetas):
    etiquetas_reales = []

    # Leer el fichero de texto y guardar cada elemento en el array
    with open(fich_etiquetas, 'r') as file:
        for line in file:
            line = line.strip()
            if line == 'Supportive':  # 0
                etiquetas_reales.append(0)
            elif line == 'Ideation':  # 1
                etiquetas_reales.append(1)
            elif line == 'Behavior':  # 2
                etiquetas_reales.append(2)
            elif line == 'Attempt':  # 3
                etiquetas_reales.append(3)
            else:  # indicator #4
                etiquetas_reales.append(4)

    return etiquetas_reales


def matriz_de_confusion(predicted_labels, train_y):
    to_string = lambda x: str(x)
    # Obtener matriz de confusión Class to clustering eval
    cm = confusion_matrix(np.vectorize(to_string)(predicted_labels), np.vectorize(to_string)(train_y))
    # Dibujar el heatmap
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Etiquetas")
    plt.ylabel("Clusters")
    plt.show()

    # Encontrar la asignación óptima utilizando el algoritmo de asignación óptima (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Reasignar los clusters según la asignación óptima
    reassigned_labels = [col_ind[labels[i]] for i in range(len(labels))]

    # Dibujar el heatmap de la nueva matriz de confusión
    new_cm = confusion_matrix(np.vectorize(to_string)(reassigned_labels), np.vectorize(to_string)(train_y))
    ax = sns.heatmap(new_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Etiquetas")
    plt.ylabel("Clusters")
    plt.show()


    # Calcula las métricas
    accuracy = accuracy_score(train_y, labels)
    precision = precision_score(train_y, labels, average='weighted')
    recall = recall_score(train_y, labels, average='weighted')
    f1 = f1_score(train_y, labels, average='weighted')

    print("FIGURAS DE MERITO")
    # Imprime las métricas
    print(f"Accuracy: \t {accuracy}")
    print(f"Precision: \t {precision}")
    print(f"Recall: \t {recall}")
    print(f"F1 Score: \t {f1}")

    return reassigned_labels


def dibujar_instancias(labels, etiquetas_reales, dim):
    valores = list(instancias.values())
    instancias_v = []

    # Recorre el array y almacena los primeros elementos en el nuevo array
    for elemento in valores:
        primer_elemento = elemento[0]
        instancias_v.append(primer_elemento)

    if len(instancias_v[0])==2 and dim>2:
        print("se van a utilizar 2 dimensiones")

    n_components = dim  # número de componentes que va a tener el PCA
    pca = PCA(n_components)
    components = pca.fit_transform(instancias_v)
    instancias_pca = pca.transform(instancias_v)

    samples = num_instancias
    # Dibujar los puntos en el espacio, color: cluster, etiqueta-numérica: clase
    # Color del punto: cluster
    if dim==2:
        sc = plt.scatter(instancias_pca[:samples, 0], instancias_pca[:samples, 1],
                         cmap=plt.cm.get_cmap('nipy_spectral', 5), c=labels[:samples])
        plt.colorbar()
        # Etiqueta numérica: clase
        for i in range(samples):
            plt.text(instancias_pca[i, 0], instancias_pca[i, 1], etiquetas_reales[i])
        plt.show()
    elif dim==3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(instancias_pca[:samples, 0], instancias_pca[:samples, 1], instancias_pca[:samples, 2],
                         cmap=plt.cm.get_cmap('nipy_spectral', 5), c=labels[:samples])
        plt.colorbar(sc)
        # Etiqueta numérica: clase
        for i in range(samples):
            ax.text(instancias_pca[i, 0], instancias_pca[i, 1], instancias_pca[i, 2], etiquetas_reales[i])
        plt.show()
    else:
        raise ValueError




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_file', type=str, help='Ruta al archivo de entrada')
    parser.add_argument('etiquetas', type=str, help='Ruta al archivo que contiene las etiquetas')
    parser.add_argument('--dim', type=int, choices=[2, 3], help='Elija una opción:  2 para dibujar 2D, 3 para dibujar 3D')

    args = parser.parse_args()

    labels = obtenerLabels(args.input_file)


    #print(labels)
    conteo = np.bincount(labels)

    # Imprimir el resultado del conteo
    for i, count in enumerate(conteo):
        print(f"El número {i} aparece {count} veces en el array.")

    etiquetas_reales = leer_etiquetas_reales(args.etiquetas)

    #print(etiquetas_reales)

    labels_reasignados = matriz_de_confusion(labels, etiquetas_reales)

    with open('instancias.pickle', 'rb') as f:
        instancias = pickle.load(f)

    if args.dim is not None:
        dim = args.dim
    else:
        dim = 2

    dibujar_instancias(labels_reasignados, etiquetas_reales, dim)


