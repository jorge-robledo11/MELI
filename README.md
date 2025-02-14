# Clasificación de Artículos Nuevos/Usados para MercadoLibre

## Tabla de Contenido
- [Descripción del Problema](#descripción-del-problema)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Distribución de Clases](#distribución-de-clases)
- [Análisis Exploratorio de Datos](#análisis-exploratorio-de-datos)
  - [Estructura del Dataset](#estructura-del-dataset)
  - [Variables Relevantes](#variables-relevantes)
- [Criterios de Selección de Variables](#criterios-de-selección-de-variables)
  - [Feature Selection](#feature-selection)
  - [Feature Engineering](#feature-engineering)
  - [Pipeline](#pipeline)
- [Modelado](#modelado)
  - [Estrategia de Modelado](#estrategia-de-modelado)
  - [Hiperparámetros Optimizados](#hiperparámetros-optimizados)
- [Resultados](#resultados)
  - [Métricas Alcanzadas](#métricas-alcanzadas)
  - [Curvas de Aprendizaje](#curvas-de-aprendizaje)
  - [Importancia de Variables](#importancia-de-variables)
- [Conclusiones](#conclusiones)

## Descripción del Problema
El proyecto consiste en desarrollar un modelo de clasificación binaria que determine si un artículo publicado en MercadoLibre es nuevo o usado, basado en características del producto y de su publicación. Se busca obtener un modelo robusto que permita distinguir correctamente entre ambos estados, mejorando la precisión en comparación con análisis previos realizados en notebooks.

## Estructura del Proyecto
```
├── LICENSE                
├── README.md
├── main.py
├── poetry.lock
├── pyproject.toml
├── artifacts/
│   ├── models/
│   │   └── final_model.json
│   └── pipelines/
│       ├── pipeline_feature_engineering.pkl
│       ├── pipeline_feature_selection.pkl
│       └── pipeline_preprocessing.pkl
├── config/  
│   ├── config.py
│   ├── config.toml
│   └── features_names.yaml
├── data/
├── docs/
├── mlruns/
├── notebooks/
│   ├── experiments/
│   │   ├── 01_preprocessing.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   ├── 03_hyperparameter_tunning.ipynb
│   │   └── 04_interpretability_and_results.ipynb
│   └── exploration/ 
│       ├── 01_exp_inicial.ipynb
│       ├── 02_analisis_negocio.ipynb
│       └── 03_eda.ipynb
├── reports/ 
│   ├── clases.png
│   ├── correlation_matrix_categoricals.png
│   ├── correlation_matrix_continuous.png
│   ├── feature-importance.png
│   ├── insights.png
│   ├── insights2.png
│   ├── insights3.png
│   ├── insights4.png
│   ├── insights5.png
│   ├── learning-curve-auc.png
│   ├── learning-curve-logloss.png
│   ├── results-fe.png
│   └── roc-curve.png
└── src/   
    ├── dataset/ 
    │   └── fetch_data.py
    ├── features/   
    │   ├── engineering.py
    │   ├── preprocessing.py
    │   ├── selection.py
    │   └── split.py
    ├── models/    
    │   ├── evaluate.py
    │   └── train.py
    └── utils/ 
        ├── preprocessors.py
        └── utils_fn.py
```

## Distribución de Clases

La gráfica muestra la proporción de artículos en dos categorías de condición ("new" y "used"). Se observa que la mayoría de los artículos se clasifica como nuevos (alrededor de un 55%), mientras que el resto corresponde a artículos usados (en torno al 45%).

![Diagrama de Clases](reports/clases.png)

## Análisis Exploratorio de Datos

### Estructura del Dataset
- Se utilizó un conjunto de datos que contiene diversas características de los productos, tales como información de precios, vendedores, envíos y atributos específicos.
- Se realizaron procesos de limpieza y transformación para manejar valores faltantes y codificar variables categóricas.

### Variables Relevantes
Se identificaron los siguientes grupos de variables:

1. **Variables de Precio y Cantidad**:
   - Precio unitario
   - Cantidad disponible
   - Cantidad vendida
   - Stock inicial

2. **Variables de Envío**:
   - Método de envío
   - Envío gratuito
   - Retiro local

3. **Variables de Ubicación**:
   - Estado/Provincia
   - Ciudad

4. **Variables de la Publicación**:
   - Tipo de listado
   - Modo de compra
   - Aceptación de MercadoPago
   - Atributos del producto

## Criterios de Selección de Variables

### Feature Selection
Se implementó una pipeline de selección de características basado en:

1. **Análisis de Correlación**:
   - Eliminación de variables altamente correlacionadas.

2. **Análisis de Varianza**:
   - Eliminación de variables con nula o cuasi nula varianza.

### Feature Engineering
Se realizaron los siguientes procesos de ingeniería de características:

1. **Transformaciones Temporales**:
   - Cálculo de la diferencia entre la fecha de última actualización y la de creación.
   - Diferencia entre el tiempo de inicio y fin de la publicación.

2. **Variables Derivadas**:
   - Monto total (precio × cantidad).
   - Ratios entre stock y ventas.

### Pipeline
Se implementó una pipeline de muchas transformaciones basado en la experimentación, donde se probaron ciertas combinaciones de parámetros y buscando maximizar una métrica en cuestión.

![Pipeline](reports/results-fe.png)
*Nota: La imagen se muestra para ilustrar el proceso de creación de variables, aunque puede no reflejar numéricamente los valores exactos obtenidos en los notebooks.*

## Modelado

### Estrategia de Modelado
- Se utilizó XGBoost como algoritmo principal.
- División del dataset: 60% entrenamiento, 20% validación y 20% prueba.
- Optimización de hiperparámetros mediante Optuna en 50 pruebas.
- Validación cruzada estratificada.
- Ajuste del learning rate de manera adaptativa según el número de iteraciones.

### Hiperparámetros Optimizados
Los mejores hiperparámetros encontrados fueron:
- Learning rate con scheduler adaptativo.
- Profundidad máxima del árbol.
- Peso mínimo por nodo.
- Regularización alpha.
- Regularización lambda.
- Subsample.
- Colsample_bytree.

## Resultados

### Métricas Alcanzadas
- **AUC en Entrenamiento**: 0.87
- **AUC en Validación**: 0.87
- **AUC en Prueba**: 0.87

![Curva ROC](reports/roc-curve.png)

*La curva ROC ilustra el rendimiento general del clasificador.*

### Curvas de Aprendizaje
Las curvas de aprendizaje indican una convergencia estable sin signos de overfitting:

![Curva de Aprendizaje AUC](reports/learning-curve-auc.png)
![Curva de Aprendizaje Log-loss](reports/learning-curve-logloss.png)

### Importancia de Variables
el **modelo se apoya principalmente en el tipo de publicación, el inventario y el valor total** para identificar la condición del producto (nuevo vs. usado), mientras que **otras variables** (envío, tiempo, entrega local y ubicación) **aportan señales adicionales** que refuerzan la clasificación.

![Importancia de Variables](reports/feature-importance.png)

*Nota: Se han notado discrepancias entre algunos resultados expuestos en los notebooks y los del modelo final. Se recomienda revisar los notebooks para obtener detalles experimentales y comprender los ajustes realizados en la etapa final.*

## Conclusiones
- El modelo obtiene un desempeño sobresaliente con AUC superior a 0.87 en todos los conjuntos.
- La mínima diferencia entre las métricas de entrenamiento y prueba refuerza la capacidad de generalización del modelo.
- Las variables relacionadas con el `listing_type`, `available_quantity` y `total_amount` ser predictores claves.
- A pesar de algunas diferencias con los resultados experimentales de los notebooks, los ajustes finales aseguran un balance adecuado entre precisión y recall.