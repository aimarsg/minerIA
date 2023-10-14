import csv
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt

ClustersList = {}
distancias = {}

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

def cluster_cercano(clusters):
    distancia_minima = float('inf')
    clusters_cercanos = (None, None)

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            for punto1 in clusters[i]:
                for punto2 in clusters[j]:
                    distancia = euclidean_distance(punto1, punto2)
                    if distancia < distancia_minima:
                        distancia_minima = distancia
                        clusters_cercanos = (i, j)

    return clusters_cercanos

def fusionar_clusters(clusters, clusters_cercanos):
    nuevo_cluster_data = clusters[clusters_cercanos[0]] + clusters[clusters_cercanos[1]]
    clusters.pop(clusters_cercanos[0])
    clusters.pop(clusters_cercanos[1] - 1)
    clusters.append(nuevo_cluster_data)

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def actualizar_distancias(clusters, clusters_cercanos):
    i, j = clusters_cercanos

    # Calcular la distancia mínima entre el nuevo cluster y los clusters restantes
    distancias_actualizadas = []

    for k in range(len(clusters)):
        if k != i and k != j:
            distancia_minima = min(
                distancias[(i, k)] if (i, k) in distancias else float('inf'),
                distancias[(j, k)] if (j, k) in distancias else float('inf')
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


def cluster_jerarquico(data, umbral):
    clusters = [[p] for p in data]
    distancias_historia = []

    while len(clusters) > 1:
        clusters_cercanos = cluster_cercano(clusters)
        distancia_minima = euclidean_distance(
            clusters[clusters_cercanos[0]][0], clusters[clusters_cercanos[1]][0]
        )

        if distancia_minima > umbral:
            break

        distancias_historia.append((clusters_cercanos, distancia_minima))
        fusionar_clusters(clusters, clusters_cercanos)
        actualizar_distancias(clusters, clusters_cercanos)

    return clusters, distancias_historia

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
    global clusters
    # 1. Limpieza de datos
    # 2. Vectorización
    # 3. Clustering jerárquico

    instancias = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    umbral = 2.0  # Umbral de distancia aquí

    #clusters = {i: {i} for i in range(len(instancias)}
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
        clusters_dimensionados=reducir_dimensionalidad_pca(clusters)
        print("Dimensionalidad reducida con PCA:", clusters_dimensionados.shape)
        umbral_clusters = 5  # Ajusta este valor según tus necesidades

        # Obtener clusters jerárquicos
        clusters, resultado_clusters = cluster_jerarquico(clusters_dimensionados, umbral_clusters)
