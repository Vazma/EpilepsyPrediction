# Plan de acción: llevar la investigación a un estado listo para revisión médica

**Objetivo:** tener resultados **reproducibles, con métricas clínicamente interpretables y comparables con la literatura** para presentarlos a un neurólogo o epileptólogo. Cada fase termina con un entregable concreto y un criterio de "listo".

> Las conclusiones del trabajo original están en pausa (ver la sección 5 del [README](README.md)) hasta completar las fases 1–3.

---

## Fase 0 — Definir el alcance

| # | Paso | Entregable |
|---|---|---|
| 0.1 | Formular la pregunta clínica: *¿se puede alertar con X minutos de anticipación con menos de Y falsas alarmas por hora?* | Párrafo de objetivo en el README |
| 0.2 | Fijar los parámetros iniciales con base en la literatura: preictal de 30 min, SPH de 5 min e interictal a ≥ 4 h de cualquier crisis | Tabla de parámetros en `config/` |
| 0.3 | Elegir el escenario principal: modelo **por paciente** (más factible) y, después, **entre pacientes** | Decisión documentada |

**Listo cuando:** la pregunta, los parámetros y las métricas están escritos y no cambian a mitad de los experimentos.

## Fase 1 — Reproducibilidad

| # | Paso | Entregable |
|---|---|---|
| 1.1 | Pasar los notebooks a módulos (`src/data`, `src/features`, `src/models`, `src/eval`) y dejar los notebooks solo para visualización | Código modular |
| 1.2 | Mover rutas, canales, pacientes y tamaños de ventana a un archivo de configuración (YAML) | `config/*.yaml` |
| 1.3 | Crear `requirements.txt` o `environment.yml` con versiones fijas, semillas fijas y un script de descarga de CHB-MIT desde PhysioNet | Entorno reproducible |
| 1.4 | Leer los tiempos de crisis y los horarios de los archivos con un parser robusto del `summary.txt`, respetando los **huecos entre grabaciones** | Tabla `patients/seizures/files` |
| 1.5 | Reproducir la corrida original (chb01–03, 3 canales) para confirmar el ~72 % como línea base | Reporte "baseline original" |
| 1.6 | Registrar cada corrida (configuración, commit, métricas) con MLflow, W&B o un CSV | Bitácora de experimentos |

**Listo cuando:** cualquier persona puede ejecutar un comando y obtener los mismos números.

## Fase 2 — Corrección metodológica de los datos

| # | Paso | Entregable |
|---|---|---|
| 2.1 | **Etiquetado:** preictal = [onset − 35 min, onset − 5 min]; interictal = segmentos a ≥ 4 h de cualquier crisis; descartar ictal y postictal | Función de etiquetado con pruebas |
| 2.2 | Usar solo **crisis "líderes"** (sin otra crisis en las horas previas) para el preictal | Lista de crisis usadas por paciente |
| 2.3 | No dejar que una ventana cruce un hueco entre archivos | Validación automática |
| 2.4 | **Normalización causal:** z-score con estadísticas del entrenamiento o de una ventana pasada, nunca de la hora completa | Preprocesamiento aplicable en tiempo real |
| 2.5 | Revisar los filtros: notch a 60 Hz y armónicos, y pasa-banda de 0.5–50 Hz (o justificar otro rango) | Documentado en la metodología |
| 2.6 | Inspeccionar visualmente ventanas preictales e interictales por paciente para detectar artefactos | Figuras para la reunión médica |

**Listo cuando:** las etiquetas no tienen fuga de información y cada decisión está justificada con una referencia.

## Fase 3 — Evaluación correcta

| # | Paso | Entregable |
|---|---|---|
| 3.1 | Validación **leave-one-seizure-out** por paciente; para pacientes nuevos, **leave-one-patient-out** | Esquema de validación |
| 3.2 | Separar entrenamiento, validación (para *early stopping* e hiperparámetros) y prueba; **nunca** elegir con el conjunto de prueba | Splits versionados |
| 3.3 | Métricas por ventana: AUC-ROC, sensibilidad y especificidad | Tabla por paciente |
| 3.4 | Métricas por evento (clínicas): **sensibilidad por crisis**, **falsas alarmas por hora (FPR/h)** y porcentaje del tiempo en alarma, con suavizado de alarmas (*k de n* ventanas positivas y periodo refractario) | Tabla por paciente y global |
| 3.5 | Comparar contra un **predictor aleatorio** (Snyder 2008) o con pruebas de surrogados, para demostrar que el resultado supera al azar | Prueba estadística |
| 3.6 | Reportar intervalos de confianza (varias semillas o bootstrap) | Media ± IC |

**Listo cuando:** existe una tabla de resultados comparable con Truong et al. 2018 (sensibilidad por crisis y FPR/h en CHB-MIT).

## Fase 4 — Modelos

| # | Paso | Entregable |
|---|---|---|
| 4.1 | **Modelos base:** potencia por bandas (δ, θ, α, β, γ) por canal con regresión logística, Random Forest o LightGBM | Referencia mínima a superar |
| 4.2 | **RNC corregida:** conv + BatchNorm + pooling + dropout + global average pooling, learning rate entre 1e-3 y 1e-4, más épocas y *early stopping* | Modelo de < 500 k parámetros |
| 4.3 | Alternativas: EEGNet o CNN-1D sobre señal cruda, y CNN + LSTM para contexto temporal | Comparación |
| 4.4 | Búsqueda de hiperparámetros solo con el conjunto de validación (ventana, preictal, canales) | Tabla de sensibilidad |
| 4.5 | **Ablación de canales:** desempeño con 18 canales frente a 4–6, más cercano a un dispositivo portátil | Curva desempeño vs. canales |
| 4.6 | Interpretabilidad: importancia de bandas y canales, y mapas Grad-CAM sobre los espectrogramas | Figuras para la reunión médica |

**Listo cuando:** el mejor modelo supera al modelo base y al predictor aleatorio con significancia estadística.

## Fase 5 — Validación externa y robustez

| # | Paso | Entregable |
|---|---|---|
| 5.1 | Probar en otro dataset público: Siena Scalp EEG (adultos) o TUH EEG Seizure Corpus | Resultados externos |
| 5.2 | Analizar fallos: qué crisis no se anticipan y qué provoca falsas alarmas (sueño, artefactos) | Reporte cualitativo |
| 5.3 | Simulación en *streaming*: inferencia con ventana deslizante sobre grabaciones continuas | Demo reproducible |

## Fase 6 — Paquete para el médico

| # | Paso | Entregable |
|---|---|---|
| 6.1 | Actualizar el README / paper con la metodología corregida, los resultados y las limitaciones | Documento final |
| 6.2 | Resumen ejecutivo de 1–2 páginas sin jerga de ML: qué predice, con cuánta anticipación, cuántas falsas alarmas y en qué pacientes funciona | PDF breve |
| 6.3 | Figuras: espectrogramas preictales e interictales, línea de tiempo de alarmas frente a crisis reales, desempeño por paciente | Set de figuras |
| 6.4 | Lista de preguntas para el especialista (ver abajo) | Agenda de la reunión |
| 6.5 | Nota ética y regulatoria: los datos son públicos y anonimizados, el trabajo no es un dispositivo médico, y un uso real requeriría un estudio clínico y aprobación regulatoria (p. ej., COFEPRIS o FDA) | Sección en el README |

### Preguntas para el neurólogo / epileptólogo

1. ¿Qué anticipación es clínicamente útil (minutos u horas)? ¿Qué tasa de falsas alarmas sería tolerable para un paciente?
2. ¿La definición de preictal (30 min, SPH de 5 min) es razonable para los tipos de crisis de CHB-MIT?
3. ¿Qué canales o regiones esperaría que fueran más informativos según el tipo de epilepsia?
4. ¿Los patrones que resalta el modelo (bandas y canales) tienen sentido fisiológico?
5. ¿Qué población sería un primer caso de uso realista (pediátrica, refractaria, monitoreo hospitalario)?
6. ¿Hay acceso a datos propios (con consentimiento y aprobación de un comité de ética) para validar después?

---

## Orden sugerido y esfuerzo estimado

| Fase | Depende de | Esfuerzo aproximado |
|---|---|---|
| 0 | — | 0.5 día |
| 1 | 0 | 1–2 días |
| 2 | 1 | 1–2 días |
| 3 | 2 | 1–2 días |
| 4 | 3 | 2–4 días (más rápido con GPU) |
| 5 | 4 | 2–3 días |
| 6 | 3–5 | 1 día |

**Recursos:** CHB-MIT ocupa unos 43 GB. Es preferible generar los espectrogramas al vuelo o guardarlos en float16, en lugar de ocupar 256 GB en disco. Las fases 1–3 y los modelos base se pueden hacer en CPU; para la Fase 4 conviene una GPU (Colab, Kaggle o en la nube).
