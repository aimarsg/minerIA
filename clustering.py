def calcular_distancia(instancia1, instancia2):
    # Calcula la distancia de Manhattan entre dos instancias
    distancia = 0
    for i in range(len(instancia1)):
        distancia += abs(instancia1[i] - instancia2[i])
    return distancia

def calcular_distancia_average(cluster1, cluster2):
    # Calcula la distancia promedio entre dos clusters
    total_distancia = 0
    for instancia1 in cluster1:
        for instancia2 in cluster2:
            total_distancia += calcular_distancia(instancia1, instancia2)

    return total_distancia / (len(cluster1) * len(cluster2)

                              
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

def actualizar_distancias(clusters, distancias):
    # Eliminar las distancias que involucran al cluster fusionado y recalcular las distancias con el nuevo cluster
    cluster_indices = list(clusters.keys())
    cluster_indices.sort()  # Ordenar los índices para evitar conflictos de eliminación

    for i in cluster_indices:
        if i != max(cluster_indices):  # Evitar recálculo para el último cluster fusionado
            del distancias[(i, max(cluster_indices))]  # Eliminar distancias sobrantes
            distancia = calcular_distancia_average(clusters[i], clusters[max(cluster_indices)])
            distancias[(i, max(cluster_indices))] = distancia

def main():
    # 1. Limpieza de datos
    # 2. Vectorización
    # 3. Clustering jerárquico

    instancias = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    umbral = 2.0  # Umbral de distancia aquí

    clusters = {i: {i} for i in range(len(instancias)}
    distancias = {}

    while len(clusters) > 1:
        calcular_distancias_clusters(instancias, clusters, distancias)
        agrupar(clusters, distancias)
        actualizar_distancias(clusters, distancias)

        if min(distancias.values()) >= umbral:
            break

    print("Clusters finales:", list(clusters.values()))

if __name__ == "__main__":
    main()
