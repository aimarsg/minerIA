# Clustering Jerárquico

Este proyecto consiste en la implementación de un algoritmo de clustering jerárquico para la asignatura de Minería de Datos.

## 🛠️ Instalación

Para instalar todos los paquetes necesarios en el entorno virual y poder ejecutar el codigo, hay que ejecutar el siguiente comando [pip](https://pip.pypa.io/en/stable/) :

```bash
pip install -r requirements.txt
```

## Contenido
El paquete de software contiene diferentes ejecutables:
*   asignar_clusters.py
*   bowAndTF-idf.py
*   clustering.py
*   evaluación.py
*   limpiarDatos.py


## Uso

El comando para hacer el clustering es el siguiente: 
```python
python clustering.py ficheroDeEntrada ficheroDeSalida 
```
Esto dará como salida una estructura de árbol que representa cómo se unen los clusters.

Tiene los siguientes parametros opcionales:
* -- PCA: dimensión de los atributos que se desea utilizar
* -- distancia: distancia entre clusters: 1 single link, 2 complete link y 3 mean link
* -- mink: grado de la distancia minkowski

Ejemplo de uso:

```python
python clustering.py doc2vec_results.csv resultsdoc2vec.txt --PCA 3 --distancia 2 --mink 3
```

Para obtener los clusters a partir del árbol resultante, hay que llamar a asignar_clusters.py:
```python
python 
```

Tiene los siguientes parametros opcionales:
* -- 

Ejemplo de uso:

```python
python 
```

Por último, para poder hacer una evaluación class to cluster y visualizar las instancias hay que hacer uso de evaluacion.py

```
python evaluacion.py ficheroDeEntrada etiquetas
```
Tiene los siguientes parametros opcionales:
* -- matriz: para visualizar la matriz de confusion y obtener las figuras de merito
* -- instancias: para visualizar las instancias
* -- dim: 2 para visualizar las instancias en 2D o 3 para visualizarlas en 3D

Ejemplo de uso
```
python evaluacion.py instancias_clusters2.txt real_labels.txt --matriz --dim 3 --instancias

```
