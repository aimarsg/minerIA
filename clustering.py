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
lista_clusters={}
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
    ###################################################QUITAR ESTO
    df = df.head(10)
    return df.values.tolist()


def reducir_dimensionalidad_pca(data):
    # Realizar PCA para reducir la dimensionalidad a 500
    pca = PCA(n_components=10)
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
            #distancia_entre_instancias[(instancia1, instancia2)] = distancia
            distancia_entre_instancias[(cluster1, cluster2)] = distancia
            if distancia < distancia_minima:
                distancia_minima = distancia
                clusters_cercanos = (cluster1, cluster2)

    distancia_entre_clusters = copy.copy(distancia_entre_instancias)
    return clusters_cercanos


def fusionar_clusters(clusters, clusters_cercanos):
    print("fusionando clusters...")
    '''cluster1, cluster2 = clusters_cercanos
    print(str(id(cluster1)) +', '+ str(id(cluster2)))
    clusters_fusionados = cluster1 + cluster2
    clusters.remove(cluster1)
    clusters.remove(cluster2)
    clusters.append(clusters_fusionados)'''

    global lista_clusters
    i1, i2 = clusters_cercanos
    print(str(i1) + ', ' + str(i2))
    cluster1 = lista_clusters[i1]
    cluster2 = lista_clusters[i2]
    clusters_fusionados = cluster1 + cluster2
    #print(clusters_fusionados)
    lista_clusters[i1] = clusters_fusionados
    lista_clusters.pop(i2)

    # print(clusters_cercanos)
    # nuevo_cluster_data = clusters[clusters_cercanos[0]] + clusters[clusters_cercanos[1]]
    # clusters.pop(clusters_cercanos[0])
    # clusters.pop(clusters_cercanos[1] - 1)
    # clusters.append(nuevo_cluster_data)

    #return clusters_fusionados


def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


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
            print(str(idxinstancia1)+', '+str(idxinstancia2))
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


def actualizar_distancias(clusters, clusters_cercanos):
    global distancia_entre_clusters

    i, j = clusters_cercanos
    print("actualizando distancias")
    print("eliminar: " + str(i)+', '+str(j))



    # suponiendo que i y j se fusionan en i
    if (i, j) in distancia_entre_clusters:
        distancia_entre_clusters.pop((i, j))
    elif (j, i) in distancia_entre_clusters:
        distancia_entre_clusters.pop((j, i))
    else:
        print("algo pasa")

    # actualizar todas las distancias respecto al cluster i
    nueva_distancia_entre_clusters = copy.copy(distancia_entre_clusters)
    for tupla, distancia in distancia_entre_clusters.items():
        cluster1, cluster2 = tupla
        if cluster1 == j or cluster2 == j:
            nueva_distancia_entre_clusters.pop(tupla)
        elif cluster1 == i or cluster2 == i:
            distancia = calcular_distancia_entre_clusters(cluster1, cluster2)
            nueva_distancia_entre_clusters[tupla] = distancia

    distancia_entre_clusters = nueva_distancia_entre_clusters

    '''
    # Calcular la distancia mínima entre el nuevo cluster y los clusters restantes
    distancias_actualizadas = []

    for k in range(len(clusters)):
        if k != i and k != j:
            distancia_minima = min(
                distancia_entre_clusters[(i, k)] if (i, k) in distancia_entre_clusters else float('inf'),
                distancia_entre_clusters[(j, k)] if (j, k) in distancia_entre_clusters else float('inf')
            )

            distancias_actualizadas.append(((i, k), distancia_minima))
            distancias_actualizadas.append(((j, k), distancia_minima))

    # Eliminar las distancias relacionadas con los clusters fusionados
    for k in range(len(clusters)):
        if (i, k) in distancias:
            del distancias[(i, k)]
        if (j, k) in distancias:
            del distancias[(j, k)]

    # Actualizar las distancias con las nuevas distancias mínimas
    for distancia, valor in distancias_actualizadas:
        distancias[distancia] = valor
    '''


def cluster_jerarquico(data, umbral):
    global distancia_entre_clusters
    global distancia_entre_instancias
    global lista_clusters
    global idx_instancias

    # clusters = [tuple(p) for p in data]
    clusters = [tuple(p) for p in data]

    ####
    j=0
    for p in data:
        print(type(p))
        print(type([p]))
        idx_instancias[j] = p
        lista_clusters[j] = [p]
        j+=1
    ####


    clusters_cercanos = inicializar_distancias(clusters)
    # print(distancia_entre_clusters[(0,1)])
    i = 1
    while len(clusters) > 1:
        #########################################################PARA DEBUGGING
        print("-------------------ITERACION "+str(i)+"-------------------------")
        '''for key in distancia_entre_clusters.keys():
            id1, id2 = key
            print(f"({id(id1)}, {id(id2)})")
        for cluster in clusters:
            print(id(cluster))'''
        for key in idx_instancias.keys():
            print(key)

        fusionar_clusters(clusters, clusters_cercanos)
        actualizar_distancias(clusters, clusters_cercanos)

        clusters_cercanos, distancia_minima = min(distancia_entre_clusters.items(), key=lambda x: x[1])

        if distancia_minima > umbral:
            break
        i+=1

    return clusters


def calcular_distancia(instancia1, instancia2):
    # Calcula la distancia de Manhattan entre dos instancias
    distancia = float("inf")
    for i in range(len(instancia1)):
        distancia += abs(instancia1[i] - instancia2[i])
    return distancia


def calcular_distancia_average(cluster1, cluster2):
    # Calcula la distancia promedio entre dos clusters
    total_distancia = 0
    for instancia1 in cluster1:
        for instancia2 in cluster2:
            total_distancia += calcular_distancia(instancia1, instancia2)

    return total_distancia / (len(cluster1) * len(cluster2))


def calcular_distancias_clusters(instancias, clusters, distancias):
    # Calcular las distancias entre todos los clusters
    for i, cluster1 in clusters.items():
        for j, cluster2 in clusters.items():
            if i < j:
                distancia = calcular_distancia_average(cluster1, cluster2)
                distancias[(i, j)] = distancia


def agrupar(clusters, distancias):
    # Encontrar las distancias mínimas y fusionar los clusters
    min_distancia = min(distancias.values())
    pares_cercanos = [par for par, distancia in distancias.items() if distancia == min_distancia]
    par_cercano = pares_cercanos[0]

    i, j = par_cercano
    clusters[i].update(clusters[j])
    del clusters[j]


"""
def actualizar_distancias(clusters, distancias):
    # Eliminar las distancias que involucran al cluster fusionado y recalcular las distancias con el nuevo cluster
    cluster_indices = list(clusters.keys())
    cluster_indices.sort()  # Ordenar los índices para evitar conflictos de eliminación

    for i in cluster_indices:
        if i != max(cluster_indices):  # Evitar recálculo para el último cluster fusionado
            del distancias[(i, max(cluster_indices))]  # Eliminar distancias sobrantes
            distancia = calcular_distancia_average(clusters[i], clusters[max(cluster_indices)])
            distancias[(i, max(cluster_indices))] = distancia
"""


def main():
    # global clusters
    # 1. Limpieza de datos
    # 2. Vectorización
    # 3. Clustering jerárquico

    instancias = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    umbral = 2.0  # Umbral de distancia aquí

    # clusters = {i: {i} for i in range(len(instancias)}
    distancias = {}

    while len(clusters) > 1:
        calcular_distancias_clusters(instancias, clusters, distancias)
        agrupar(clusters, distancias)
        actualizar_distancias(clusters, distancias)

        if min(distancias.values()) >= umbral:
            break

    print("Clusters finales:", list(clusters.values()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python clustering.py entrada.csv")
        exit(0)
    else:
        clusters = leer_datos()
        clusters_dimensionados = reducir_dimensionalidad_pca(clusters)
        print("Dimensionalidad reducida con PCA:", clusters_dimensionados.shape)
        umbral_clusters = 24  # Ajusta este valor según tus necesidades

        # Obtener clusters jerárquicos
        clusters = cluster_jerarquico(clusters_dimensionados, umbral_clusters)
        print(clusters)
