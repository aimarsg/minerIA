import copy
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt

distancia_entre_instancias = {}
distancia_entre_clusters = {}
lista_clusters = {}
idx_instancias = {}


def obtenerIdxInstancia(instancia):
    global idx_instancias
    for key, value in idx_instancias.items():
        if id(instancia) == id(value):
            return key


def leer_datos():
    input_file = sys.argv[1]
    df = pd.read_csv(input_file)
    df = df.drop('User', axis=1)
    df = df.drop('Label', axis=1)
    return df.values.tolist()


def reducir_dimensionalidad_pca(data):
    # Realizar PCA para reducir la dimensionalidad a 500
    pca = PCA(n_components=500)
    df_reducido = pca.fit_transform(data)

    return df_reducido


def inicializar_distancias(clusters):
    global distancia_entre_clusters
    global distancia_entre_instancias
    global lista_clusters

    distancia_minima = float('inf')
    clusters_cercanos = (None, None)

    for cluster1 in range(len(clusters)):
        for cluster2 in range(cluster1 + 1, len(clusters)):
            instancia1 = clusters[cluster1]
            instancia2 = clusters[cluster2]
            distancia = euclidean_distance(instancia1, instancia2)
            distancia_entre_instancias[(cluster1, cluster2)] = distancia
            if distancia < distancia_minima:
                distancia_minima = distancia
                clusters_cercanos = (cluster1, cluster2)

    distancia_entre_clusters = copy.copy(distancia_entre_instancias)
    return clusters_cercanos


def fusionar_clusters(clusters_cercanos):
    print("fusionando clusters...")
    global lista_clusters
    i1, i2 = clusters_cercanos
    cluster1 = lista_clusters[i1]
    cluster2 = lista_clusters[i2]
    clusters_fusionados = cluster1 + cluster2
    lista_clusters[i1] = clusters_fusionados
    lista_clusters.pop(i2)


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def calcular_distancia_entre_clusters(idxcluster1, idxcluster2):
    global distancia_entre_instancias
    global distancia_entre_clusters
    global lista_clusters

    cluster1 = lista_clusters[idxcluster1]
    cluster2 = lista_clusters[idxcluster2]

    distancia_minima = float('inf')
    dist = 0
    for instancia1 in cluster1:
        for instancia2 in cluster2:
            idxinstancia1 = obtenerIdxInstancia(instancia1)
            idxinstancia2 = obtenerIdxInstancia(instancia2)
            tupla1 = (idxinstancia1, idxinstancia2)
            tupla2 = (idxinstancia2, idxinstancia1)
            if tupla1 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla1]
            elif tupla2 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla2]
            else:
                print("Error catastofico")
            if dist < distancia_minima:
                distancia_minima = dist

    return distancia_minima


def actualizar_distancias(clusters_cercanos):
    global distancia_entre_clusters
    i, j = clusters_cercanos
    print("actualizando distancias")

    # suponiendo que i y j se fusionan en i
    if (i, j) in distancia_entre_clusters:
        distancia_entre_clusters.pop((i, j))
    elif (j, i) in distancia_entre_clusters:
        distancia_entre_clusters.pop((j, i))
    else:
        print("algo pasa")

    # actualizar todas las distancias respecto al cluster i
    nueva_distancia_entre_clusters = copy.copy(distancia_entre_clusters)

    for cluster in lista_clusters.keys():
        if cluster != i:
            if (i, cluster) in distancia_entre_clusters:
                distancia = calcular_distancia_entre_clusters(i, cluster)
                nueva_distancia_entre_clusters[(i, cluster)] = distancia
            elif (cluster, i) in distancia_entre_clusters:
                distancia = calcular_distancia_entre_clusters(cluster, i)
                nueva_distancia_entre_clusters[(cluster, i)] = distancia
            else:
                print("error")
        if cluster != j and cluster != i:
            if (j, cluster) in distancia_entre_clusters:
                nueva_distancia_entre_clusters.pop((j, cluster))
            elif (cluster, j) in distancia_entre_clusters:
                nueva_distancia_entre_clusters.pop((cluster, j))
            else:
                print("error2")

    """for tupla, distancia in distancia_entre_clusters.items():
        cluster1, cluster2 = tupla
        if cluster1 == j or cluster2 == j:
            nueva_distancia_entre_clusters.pop(tupla)
        elif cluster1 == i or cluster2 == i:
            distancia = calcular_distancia_entre_clusters(cluster1, cluster2)
            nueva_distancia_entre_clusters[tupla] = distancia"""

    distancia_entre_clusters = nueva_distancia_entre_clusters


def cluster_jerarquico(data, umbral):
    global distancia_entre_clusters
    global distancia_entre_instancias
    global lista_clusters
    global idx_instancias

    j = 0
    for p in data:
        idx_instancias[j] = p
        lista_clusters[j] = [p]
        j += 1

    clusters_cercanos = inicializar_distancias(lista_clusters)

    i = 1
    while len(lista_clusters) > 1:

        print("-------------------ITERACION " + str(i) + "-------------------------")

        fusionar_clusters(clusters_cercanos)
        actualizar_distancias(clusters_cercanos)

        clusters_cercanos, distancia_minima = min(distancia_entre_clusters.items(), key=lambda x: x[1])

        if distancia_minima > umbral:
            break
        i += 1

    return lista_clusters


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python clustering.py entrada.csv")
        exit(0)
    else:
        datos = leer_datos()
        clusters_dimensionados = reducir_dimensionalidad_pca(datos)
        print("Dimensionalidad reducida con PCA:", clusters_dimensionados.shape)
        umbral_clusters = 20  # Ajusta este valor según tus necesidades

        # Obtener clusters jerárquicos
        clusters = cluster_jerarquico(clusters_dimensionados, umbral_clusters)
        print(clusters)
        print(len(clusters))
