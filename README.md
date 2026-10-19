# Predicción de eventos de epilepsia con Deep Learning y Redes Neuronales Convolucionales (RNC) a partir de espectrogramas de EEG

**Autor:** Alexis Vázquez Martínez · **Asesor:** Dr. José Antonio Cantoral Ceballos · **Coordinadores:** Dr. José Antonio Cantoral Ceballos, Dr. Jesús Arturo Pérez Díaz

> **Estado del documento:** investigación en revisión. Este README separa **(A) hechos verificables** (lo que hace el código y las salidas guardadas en los notebooks) de **(B) resultados y conclusiones pendientes de validación**. Las conclusiones del trabajo original **no deben tomarse como evidencia clínica** hasta completar el [plan de acción](PLAN_DE_ACCION.md).

---

## Resumen

El estudio de las señales cerebrales mediante electroencefalogramas (EEG) se ha usado desde hace tiempo para estudiar ataques epilépticos y trastornos del sueño, porque son señales relativamente sencillas de obtener y no invasivas. Ya existen algoritmos que intentan predecir crisis epilépticas, pero la mayoría se ha centrado en el análisis de señales crudas y no tiene suficiente sustento para un uso terapéutico.

Esta investigación propone otro enfoque: transformar los EEG en **espectrogramas** (representaciones tiempo-frecuencia tratadas como imágenes) y clasificarlos con una **red neuronal convolucional 2D**. El objetivo es distinguir el periodo **preictal** (antes de una crisis) del periodo **interictal** (entre crisis). La meta inicial que se propuso fue una efectividad mayor al 60 %.

## 1. Introducción

La epilepsia es una enfermedad cerebral crónica que afecta a cerca de 50 millones de personas en el mundo, con mayor concentración en países en desarrollo. Se caracteriza por convulsiones parciales o generalizadas, que pueden ir acompañadas de pérdida de conciencia o del control de esfínteres. Las crisis pueden causar problemas emocionales y físicos; estos últimos ocurren sobre todo durante el ataque y pueden ser graves según el contexto (ahogamientos, accidentes automovilísticos). Las crisis son cambios temporales en la actividad bioeléctrica del cerebro, por lo que pueden observarse mediante EEG: electrodos colocados sobre el cuero cabelludo que registran los impulsos eléctricos de las neuronas.

En la última década, la disponibilidad de grandes conjuntos de datos de EEG impulsó el uso de deep learning para clasificarlos. La literatura reciente se ha alejado del análisis de señal cruda y prefiere representaciones como imágenes de ventanas cortas de tiempo, que son más robustas a la variabilidad de la señal. Las RNC son una opción natural para esas representaciones.

### 1.1 Problemática

- Trabajos previos reportan buenos resultados, pero la obtención de muestras y el preprocesamiento suelen describirse de forma ambigua.
- Varios trabajos con RNC usan bases de datos privadas o son anteriores a 2020, lo que dificulta reproducirlos.

### 1.2 Estado del arte considerado

- Gao & Wang, *EEG Signal Processing and Feature Extraction* (Springer, 2019): el análisis de EEG como imágenes es un enfoque prometedor, aunque sin una comparación profunda entre modelos.
- Moctezuma & Molinas, *EEG Channel-Selection Method for Epileptic-Seizure Classification Based on Multi-Objective Optimization* (Frontiers in Neuroscience, 2020): usa la misma base de datos (CHB-MIT) y sirvió de referencia para elegir pacientes y canales.
- Alotaibi et al., *A review of channel selection algorithms for EEG signal processing* (2015).
- Michel & Brunet, *EEG Source Imaging: A Practical Review of the Analysis Steps* (2019).
- Roy et al., *Deep learning-based electroencephalography analysis: a systematic review* (2019).
- Gong et al., *Deep Learning in EEG: Advance of the Last Ten-Year Critical Period* (2021).
- Truong et al., *Convolutional neural networks for seizure prediction using intracranial and scalp EEG* (Neural Networks, 2018). Usa un enfoque muy similar (STFT + CNN sobre CHB-MIT) y se agregó en esta revisión como **punto de comparación**.

## 2. Datos

**CHB-MIT Scalp EEG Database v1.0.0** (PhysioNet, acceso abierto; Shoeb, 2009; Goldberger et al., 2000).

- Registros del Boston Children's Hospital: 22 sujetos (5 hombres, 17 mujeres, de 1.5 a 22 años), agrupados en 23 casos.
- Muestreo de 256 Hz, resolución de 16 bits, montaje bipolar del sistema 10-20.
- Archivos `.edf` de aproximadamente 1 h, con un archivo `chbXX-summary.txt` por paciente que indica el inicio y fin de cada crisis.
- 198 crisis anotadas en total.

Los datos **no se incluyen en este repositorio**. Se descargan desde https://physionet.org/content/chbmit/1.0.0/ (≈ 43 GB).

## 3. Metodología (tal como está implementada)

### 3.1 Herramientas

Python con `mne` (lectura de EDF), `scipy` (espectrogramas), `numpy`, `torch` (modelo) y `os`.

### 3.2 Selección de pacientes y canales

- **Diseño descrito en el paper:** 19 pacientes (chb01–chb11, chb14, chb17–chb23) y 18 canales bipolares (`FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2, FP2-F8, F8-T8, T8-P8-0, P8-O2, FZ-CZ, CZ-PZ`). Todos los pacientes tienen al menos 3 crisis registradas.
- **Lo que ejecutaron los notebooks guardados:** 3 pacientes (`chb01, chb02, chb03`) y 3 canales (`print(n_channels)` → `3`). La lista de 12 canales que aparece hoy en el código se editó después de esa ejecución.
- Solo se procesan los archivos `.edf` que contienen todos los canales seleccionados.

### 3.3 Espectrogramas (`Epilepsy_spectograms.ipynb`)

- `scipy.signal.spectrogram(x, fs=256, nfft=256, noverlap=128)`: segmentos de 256 muestras con 50 % de traslape. Esto da una **resolución de 1 Hz** (0–128 Hz, 129 bins) y **2 columnas por segundo**.
- Se eliminan los bins **57–63 Hz** (interferencia de la línea eléctrica de 60 Hz) y **100–128 Hz**. Quedan **93 bins de frecuencia**.
- Escala logarítmica (`10·log10`) y normalización **min-max por canal y por archivo completo**.
- Los espectrogramas de todos los archivos de un paciente se concatenan en el eje del tiempo. Los tiempos de crisis se reajustan a ese eje concatenado.

### 3.4 Ventanas y etiquetado

- Ventana de **60 s** (120 columnas). La elección se basó en Panayu et al. (2021) sobre tamaño de ventana en RNC para EEG.
- **Preictal (clase 1):** los **20 minutos previos** al inicio de cada crisis, con ventanas traslapadas que avanzan cada 10 s.
- **Interictal (clase 0):** el resto del tiempo entre el fin de una crisis y el inicio del preictal siguiente. Se toman ventanas espaciadas uniformemente, en la misma cantidad que las preictales, de modo que el conjunto queda balanceado.
- **Partición:** la **última crisis de cada paciente** va a prueba y las anteriores a entrenamiento. No hay conjunto de validación.

### 3.5 Modelo (`Epilepsy_model.ipynb`)

- Entrada: tensor `canales × 93 × 120`.
- 5 capas `Conv2d` (kernel 3×3, padding 1, 64 filtros, inicialización Kaiming) con activación ReLU y **sin pooling**.
- Capa totalmente conectada de 64·93·120 = 714 240 → 2 clases.
- `BatchNorm2d` está definido en el código pero **no se aplica** (está comentado en `forward`).
- Entrenamiento: Adam, *learning rate* 3.5e-7, 5 épocas, batch de 16, `CrossEntropyLoss`.
- Métrica: exactitud (*accuracy*) sobre los conjuntos balanceados.
- Hardware: Intel i5 de 8.ª generación sin GPU. El pipeline completo tardó hasta 18 h y ocupó unos 256 GB de disco.

## 4. Resultados

### 4.1 Hechos (salidas guardadas en los notebooks)

| Conjunto | Interictal (0) | Preictal (1) |
|---|---|---|
| Entrenamiento | 1512 ventanas | 1512 ventanas |
| Prueba | 342 ventanas | 342 ventanas |

| Época | Loss (último batch) | Accuracy en prueba |
|---|---|---|
| 0 | 0.643 | 71.93 % |
| 1 | 0.643 | 71.49 % |
| 2 | 0.566 | 72.08 % |
| 3 | 0.787 | 72.95 % |
| 4 | 0.671 | 72.22 % |

Accuracy final en entrenamiento: **74.11 %**. Configuración: 3 pacientes, 3 canales.

### 4.2 Resultados reportados en el paper y en la presentación (sin evidencia reproducible en el repositorio)

| Experimento | Train | Test | Fuente |
|---|---|---|---|
| 19 pacientes, 18 canales | 65 % | 56 % | Paper y slides |
| 10 pacientes, 10 canales | 72 % | 74 % | Paper (los valores parecen invertidos respecto al notebook) |
| 19 × 18 + capa de max-pooling | 65 % | 55 % | Paper |

No hay en el repositorio salidas ni configuraciones que permitan reproducir estas tres corridas.

### 4.3 Discrepancias detectadas entre paper y código

| Aspecto | Paper | Código |
|---|---|---|
| Pacientes / canales de la corrida guardada | 19 × 18 o 10 × 10 | 3 × 3 |
| Frecuencias eliminadas | 57–64 Hz y 117–124 Hz | 57–63 Hz y 100–128 Hz |
| Batch normalization | Aplicada tras cada convolución | No aplicada |

## 5. Discusión y conclusiones — **PENDIENTES DE VALIDACIÓN**

> Las conclusiones del trabajo original quedan en suspenso. A continuación se listan las afirmaciones originales y la razón por la que **todavía no pueden sostenerse**.

| Afirmación original | Estado | Motivo |
|---|---|---|
| "El modelo logró resultados significativos" (~72–74 %) | ⏳ Pendiente | Solo hay una corrida guardada, con 3 pacientes y sin conjunto de validación. El conjunto de prueba se evaluó en cada época. |
| El modelo distingue el periodo preictal del interictal | ⏳ Pendiente | La accuracy ya es 72 % en la época 0 y no mejora, y el loss se queda cerca de ln 2 ≈ 0.69. Con un learning rate de 3.5e-7 la red apenas se actualiza, así que la separación podría venir de artefactos del preprocesamiento y no de un patrón preictal. |
| Usar menos pacientes y canales da mejores resultados | ⏳ Pendiente | Las corridas no son comparables (distintos datos y ninguna evaluación con varias semillas ni intervalos de confianza). |
| Un dispositivo en tiempo real podría usar este modelo | ⏳ Pendiente | La normalización usa estadísticas de la hora completa, incluida la crisis futura, lo que no es posible en tiempo real. Tampoco hay métricas clínicas (sensibilidad por crisis, falsas alarmas por hora). |

**Factores metodológicos que deben corregirse antes de concluir** (se detallan en el [plan de acción](PLAN_DE_ACCION.md)):

1. No hay horizonte de predicción (SPH) ni margen postictal: el interictal empieza justo al terminar la crisis anterior.
2. La normalización no es causal (usa información del futuro).
3. Los huecos entre grabaciones se ignoran al concatenar archivos.
4. La accuracy balanceada no es una métrica clínica; faltan sensibilidad, FPR/h, AUC y la comparación contra un predictor aleatorio.
5. No hay conjunto de validación ni validación cruzada (por ejemplo, *leave-one-seizure-out*).
6. Faltan modelos base simples (potencia por bandas) para saber si la RNC aporta algo.
7. No se evalúa por separado el desempeño por paciente frente al desempeño en pacientes nuevos.

**Lo que sí se mantiene:**

- La representación espectrograma + RNC 2D es un enfoque válido y usado en la literatura (p. ej., Truong et al. 2018).
- CHB-MIT es un dataset público y adecuado para reproducir y compararse con otros trabajos.
- El trabajo requiere la guía de un especialista médico (neurólogo o epileptólogo) para definir el periodo preictal, interpretar señales y evaluar su aplicabilidad clínica.
- La infraestructura fue una limitante real (sin GPU, 18 h de ejecución, 256 GB de disco), lo que restringió la exploración de parámetros.

## 6. Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `Epilepsy_spectograms.ipynb` | Lectura de EDF, generación de espectrogramas por paciente, ventanas y partición train/test |
| `Epilepsy_model.ipynb` | Dataset de PyTorch, modelo RNC, entrenamiento y evaluación |
| `Prediccion de Eventos de Epilepsia ... .pdf` | Paper original (versión 2021) |
| `Deep learning en la predicción de epilepsia Final.pptx` | Presentación original |
| `PLAN_DE_ACCION.md` | Pasos para corregir y validar la investigación |

## 7. Referencias principales

1. Organización Mundial de la Salud, *Epilepsia*, 2016.
2. Shoeb A., *Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment*, PhD Thesis, MIT, 2009.
3. Goldberger A. et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23), 2000.
4. Moctezuma L. & Molinas M., *EEG Channel-Selection Method for Epileptic-Seizure Classification Based on Multi-Objective Optimization*, Front. Neurosci., 2020.
5. Gong S. et al., *Deep Learning in EEG: Advance of the Last Ten-Year Critical Period*, IEEE, 2021.
6. Craik A., He Y. & Contreras-Vidal J., *Deep learning for EEG classification tasks: a review*, J. Neural Eng., 2019.
7. Roy Y. et al., *Deep learning-based electroencephalography analysis: a systematic review*, J. Neural Eng., 2019.
8. Panayu K. et al., *A Comparative Study of Window Size and Channel Arrangement on EEG-Emotion Recognition Using Deep CNN*, 2021.
9. Truong N. D. et al., *Convolutional neural networks for seizure prediction using intracranial and scalp EEG*, Neural Networks 105, 2018.
10. Mormann F. et al., *Seizure prediction: the long and winding road*, Brain 130(2), 2007.
11. Snyder D. et al., *The statistics of a practical seizure warning system*, J. Neural Eng., 2008.

La lista completa de referencias del trabajo original está en el PDF.
