import argparse
import copy
import pdb
import time
import pickle
import sys
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

# VARIABLES GLOBALES #
distancia_entre_instancias = {}
distancia_entre_clusters = {}
lista_clusters = {}
t_distancia = 1  # default: single link
t_mink = 2  # default: distancia euclidea
fich_salida = 'salida.txt'  # default


def leer_datos(input_file):
    """
    input file: dataset en formato csv que contiene las columnas user, label y post
    devuelte: una lista de los posts
    """
    df = pd.read_csv(input_file, sep=",")
    df = df.drop('User', axis=1)
    df['Label'].to_csv("real_labels.txt", index=False, header=False)
    df = df.drop('Label', axis=1)
    #df = df.head(50)
    with open(sys.argv[2], 'w') as archivo:
        archivo.write(f"numero de instancias: {len(df.index)} \n")
    return df.values.tolist()

def leer_datos2(input_file):
    """
    input file: dataset en formato csv que contiene las columnas user, label y post
    devuelte: una lista de los posts
    """
    df = pd.read_csv(input_file, sep=",")
    df = df.drop('User', axis=1)
    df['Label'].to_csv("real_labels.txt", index=False, header=False)
    df = df.drop('Label', axis=1)
    #df = df.head(50)
    return df.values.tolist()


def reducir_dimensionalidad_pca(data, dim):
    """
    data: lista de instancias, dim: la nueva dimension
    devuelve: la lista con el numero de dimensiones indicado
    """
    pca = PCA(n_components=dim)
    df_reducido = pca.fit_transform(data)
    return df_reducido


def inicializar_distancias(clusters):
    """
    clusters: un diccionario de los clusters iniciales (un cluster por cada instancia)
    calcula la distancia entre todas las instancias y se almacena en
                                                                - distancia_entre_instancias (diccionario)
                                                                - distancia_entre_clusters   (dicccionario)
    """
    global distancia_entre_clusters
    global distancia_entre_instancias
    distancia_minima = float('inf')

    for cluster1 in range(len(clusters)):
        for cluster2 in range(cluster1 + 1, len(clusters)):
            instancia1 = clusters[cluster1][0]
            instancia2 = clusters[cluster2][0]
            distancia = distancia_minkowski(instancia1, instancia2)
            distancia_entre_instancias[(cluster1, cluster2)] = distancia
            if distancia < distancia_minima:
                distancia_minima = distancia

    distancia_entre_clusters = copy.copy(distancia_entre_instancias)


def fusionar_clusters(clusters_cercanos):
    """
    clusters_cercanos: una tupla con los indices de los clusters con distancia menor (i1, i2)
    devuelve: el id del nuevo cluster creado que contiene los dos id de los clusters_cercanos
    """
    global lista_clusters
    i1, i2 = clusters_cercanos
    clusters_fusionados = [i1, i2]
    id_nuevo_cluster = len(lista_clusters)
    lista_clusters[id_nuevo_cluster] = clusters_fusionados
    return id_nuevo_cluster


def distancia_minkowski(point1, point2):
    """
    point1 y point2: dos vectores de igual longitud que representan dos instancias distintas
    devuelve: la distancia minkowski de orden t_mink entre los dos vectores
    """
    res = (np.sum((abs(np.array(point1) - np.array(point2))) ** t_mink)) ** (1 / t_mink)
    # res = np.power(np.sum((np.array(point1) - np.array(point2)) ** t_mink), (1/t_mink))
    return res


def calcular_distancia_entre_clusters(idxcluster1, idxcluster2):
    """
    idxcluster1 e idxcluster2: id de clusters
    devuelve: la distancia entre esos dos clusters, a partir de los valores almacenados en el diccionario de distancias,
        en funcion del valor t_distancia:
        1> SINGLE LINK (minima)
        2> COMPLETE LINK (maxima)
        3> MEAN LINK (media)
    """
    global distancia_entre_instancias
    global distancia_entre_clusters
    global lista_clusters

    distancias = []

    cluster1 = lista_clusters[idxcluster1]
    cluster2 = lista_clusters[idxcluster2]

    if not isinstance(cluster1[0], int):
        cluster1 = [idxcluster1]

    if not isinstance(cluster2[0], int):
        cluster2 = [idxcluster2]

    distancia_minima = float('inf')
    dist = 0
    for idxinstancia1 in cluster1:
        for idxinstancia2 in cluster2:
            tupla1 = (idxinstancia1, idxinstancia2)
            tupla2 = (idxinstancia2, idxinstancia1)
            # comprobar si ya existe la distancia entre las instancias de un cluster y el otro cluster directamente
            if (idxcluster1, idxinstancia2) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxcluster1, idxinstancia2)]
                distancias.append(dist)
            elif (idxinstancia2, idxcluster1) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxinstancia2, idxcluster1)]
                distancias.append(dist)
            elif (idxinstancia1, idxcluster2) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxinstancia1, idxcluster2)]
                distancias.append(dist)
            elif (idxcluster2, idxinstancia1) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxcluster2, idxinstancia1)]
                distancias.append(dist)
            # comprobar distancias entre pares de instancias de ambos clusters
            elif tupla1 in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[tupla1]
                distancias.append(dist)
            elif tupla2 in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[tupla2]
                distancias.append(dist)
            # si no estan en la distancia entre clusters podria estar en la distancia entre instancias,
            # ya que podrian ser clusters aun no fusionados
            elif tupla1 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla1]
                distancias.append(dist)
            elif tupla2 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla2]
                distancias.append(dist)

            else:
                print(
                    f"Error : tupla1 {tupla1} \n clusters (d) {idxcluster1}: {cluster1} y {idxcluster2} :{cluster2} \n")
                input("continuar...")

    # LOGICA PARA SACAR LA DISTANCIA ENTRE LOS CLUSTERS

    if t_distancia == 1:
        return min(distancias)

    elif t_distancia == 2:
        return max(distancias)

    elif t_distancia == 3:
        return sum(distancias) / len(distancias)


def actualizar_distancias(clusters_cercanos, id_nuevo_cluster):
    """
    clusters_cercanos: tupla con los clusters que se unen
    id_nuevo_cluster: id del cluster nuevo creado que contiene los clusters que se unen

    elimina del diccionario distancia_entre_clusters las distancias de los clusters_cercanos a los demas
    añade al diccionario distancia_entre_clusters las distancias del nuevo cluster a los demas
    """
    global distancia_entre_clusters

    i, j = clusters_cercanos

    if (i, j) in distancia_entre_clusters:
        distancia_entre_clusters.pop((i, j))
    elif (j, i) in distancia_entre_clusters:
        distancia_entre_clusters.pop((j, i))
    else:
        print("error: no se ha encontrado en la lista de distancias entre clusters los dos clusters a fusionar")

    nueva_distancia_entre_clusters = copy.copy(distancia_entre_clusters)

    for cluster in lista_clusters.keys():
        if cluster != i and cluster != id_nuevo_cluster:
            if cluster != j:

                if (i, cluster) in distancia_entre_clusters:
                    distancia = calcular_distancia_entre_clusters(id_nuevo_cluster, cluster)
                    nueva_distancia_entre_clusters[(id_nuevo_cluster, cluster)] = distancia
                    nueva_distancia_entre_clusters.pop((i, cluster))

                elif (cluster, i) in distancia_entre_clusters:
                    distancia = calcular_distancia_entre_clusters(cluster, id_nuevo_cluster)
                    nueva_distancia_entre_clusters[(cluster, id_nuevo_cluster)] = distancia

                    nueva_distancia_entre_clusters.pop((cluster, i))

        if cluster != j and cluster != id_nuevo_cluster:
            if cluster != i:

                if (j, cluster) in distancia_entre_clusters:
                    nueva_distancia_entre_clusters.pop((j, cluster))
                elif (cluster, j) in distancia_entre_clusters:
                    nueva_distancia_entre_clusters.pop((cluster, j))

    distancia_entre_clusters = nueva_distancia_entre_clusters


def asignar_labels(num_instancias):
    """
    num_instancias: el numero de instancias inicial
    calcula la etiqueta correspondiente a cada instancia, yendo hacia abajo en el arbol
    devuelve: una lista con la etiqueta correspondiente a cada instancia y el numero de etiquetas distintas
    """
    labels = np.arange(num_instancias)
    lista = list(lista_clusters.keys())
    while True and len(lista) > 0:
        elemento = lista.pop()
        if elemento < num_instancias:
            break
        marcar_labels(lista_clusters[elemento], labels, elemento, lista)  # llamada al metodo recursivo
    unique_labels = np.unique(labels)
    return labels, len(unique_labels)


def marcar_labels(cluster, labels, label, lista):
    """
    metodo recursivo para profundizar en el arbol y obtener la etiqueta de cada instancia
    """
    for elemento in cluster:
        idElemento = lista.index(elemento)
        lista.pop(idElemento)
        if not isinstance(lista_clusters[elemento][0], int):  # es decir, no es un cluster
            labels[elemento] = label
        else:
            marcar_labels(lista_clusters[elemento], labels, label, lista)


def cluster_jerarquico(data):
    """
    data: las instancias vectorizadas sobre las que se quiere aplicar el clustering jerarquico
    devuelve:
            - lista_clusters: un diccionario que representa el arbol generado en el algoritmo,
            que contiene las instancias originales y todas las uniones que se han realizado en las sucesivas iteraciones
            - un array con la información necesaria para hacer la representación gráfica a traves de un dendograma
    """
    global distancia_entre_clusters
    global distancia_entre_instancias
    global lista_clusters

    j = 0
    for p in data:  # inicializar la lista de clusters
        lista_clusters[j] = [p]
        j += 1

    with open('instancias.pickle', 'wb') as f:
        pickle.dump(lista_clusters, f)

    inicializar_distancias(lista_clusters)  # inicializar las distancias

    i = 1
    Z = []  # Matriz de enlace (para la representacion del dendograma)
    silhouette_scores = []  # lista de puntuaciones silhouette
    lista_cluster_fus = {}
    with open(sys.argv[2], 'a') as archivo:
        while j > i:
            # obtener clusters con la distancia minima
            clusters_cercanos, distancia_minima = min(distancia_entre_clusters.items(), key=lambda x: x[1])

            # unir los clusters mas cercanos y actualizar el diccionario de distancias
            id_cluster_nuevo = fusionar_clusters(clusters_cercanos)
            lista_cluster_fus[id_cluster_nuevo] = clusters_cercanos
            actualizar_distancias(clusters_cercanos, id_cluster_nuevo)

            archivo.write(
                f"iteración: {i}, distancia: {distancia_minima}, clusters fusionados: {id_cluster_nuevo}:{lista_clusters[id_cluster_nuevo]}, ")

            labels, unique = asignar_labels(num_instancias=j)
            # Calcular Silhouette para cada iteración
            if unique > 1:
                silhouette = silhouette_score(data, labels)
                silhouette_scores.append(silhouette)
                archivo.write(f"Silhouette Score: {silhouette}\n")
            for key, values in lista_cluster_fus.items():
                instancias_fusionadas = []
                for v in values:
                    lista_instancias = conseguir_instancias_fusionadas(v, lista_clusters, lista_cluster_fus,
                                                                       instancias_fusionadas)

            # añadir la informacion para el dendograma
            Z.append(
                [clusters_cercanos[0], clusters_cercanos[1], distancia_minima, len(lista_instancias)])

            i += 1
    return lista_clusters, np.array(Z)


def conseguir_instancias_fusionadas(cluster, instancias, lista, instancias_fusionadas):
    """
    Método recursivo para devolver las instancias de los clusters
    """
    if cluster in lista.keys():  # Si es un índice de instancia
        a = lista[cluster]
        for v in a:
            conseguir_instancias_fusionadas(v, instancias, lista, instancias_fusionadas)
    else:
        instancias_fusionadas.append(instancias[cluster][0])
    return instancias_fusionadas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering jerarquico")
    parser.add_argument('fich_entrada', type=str, help="Ruta al fichero de entrada")
    parser.add_argument('salida', type=str, help="Nombre del archivo de salida")
    parser.add_argument('--PCA', type=int, help="opcional, indica la dimension del PCA")
    parser.add_argument('--distancia', type=int, choices=[1, 2, 3],
                        help="opcional, indica la distancia entre clusters a utilizar\n"
                             "1 para single link\n"
                             "2 para complete link\n"
                             "3 para mean link")
    parser.add_argument('--mink', type=int, help='Distancia minkowski', required=False)
    start_time = time.time()
    args = parser.parse_args()

    fich_salida = args.salida

    datos = leer_datos(args.fich_entrada)

    if args.PCA is not None:
        datos = reducir_dimensionalidad_pca(datos, args.PCA)
        print("Dimensionalidad reducida con PCA:", datos.shape)

    if args.distancia is not None:
        t_distancia = args.distancia

    if args.mink is not None:
        if args.mink >= 1:
            t_mink = args.mink
        else:
            print("distancia no valida, se va a utilizar la euclidea")
            input("pulsa para continuar... ")

    print(f"Numero de instancias: {len(datos)}")

    # Obtener clusters jerárquicos
    clusters, Z = cluster_jerarquico(datos)

    #Z = linkage(datos)
    # Mostrar dendrograma y guardarlo
    dendrogram(Z)
    #plt.gcf().set_size_inches(38.4, 21.6)
    #plt.savefig(args.salida + ".png", dpi=500, bbox_inches='tight')
    plt.show()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time} segundos")
