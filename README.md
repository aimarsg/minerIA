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
*   bowAndTF_idf.py
*   clustering.py
*   evaluación.py
*   limpiarDatos.py


## Uso

Primero hay que utilizar bowAndTF_idf.py para preprocesar el conjunto de datos. Con el parametro --csv_file se indica el fichero de entrada.

```python
python bowAndTf_idf.py --csv_file 500_Reddit_users_posts_labels.csv
```

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
python clustering.py datos_preprocesados.csv resultadosClustering.txt --PCA 3 --distancia 2 --mink 3
```

Para obtener los clusters a partir del árbol resultante, hay que llamar a asignar_clusters.py con diferentes opciones, ssegún la opción deseada:
* Obtener los clusters en función de una distancia dada:
```python
python asignar_clusters.py 1 resultadosClustering.txt --distancia 3 
```
* Obtener los clusters en función del número de clusters deseado:
```python
python asignar_clusters.py 2 resultadosClustering.txt --num 5
```
* Dados un número de clusters y unas instancias, asignar esas instancias a los clusters a los que pertenecen:

```python
python asignar_clusters.py 3 resultadosClustering.txt --nuevas_instancias instancias.csv --num 5 
```

Por último, para poder hacer una evaluación class to cluster y visualizar las instancias hay que hacer uso de evaluacion.py

```python
python evaluacion.py ficheroDeEntrada etiquetas
```
Tiene los siguientes parametros opcionales:
* -- matriz: para visualizar la matriz de confusion y obtener las figuras de merito
* -- instancias: para visualizar las instancias
* -- dim: 2 para visualizar las instancias en 2D o 3 para visualizarlas en 3D

Ejemplo de uso
```python
python evaluacion.py instancias_clusters2.txt real_labels.txt --matriz --dim 3 --instancias

```
