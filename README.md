# BoostLLM_Anemia_Peru

Predicción de anemia infantil (y, después, HTA no diagnosticada y depresión PHQ-9) con microdatos ENDES, comparando
XGBoost, TabLLM y un híbrido árbol-LLM (BoostLLM) sobre MedGemma.

## Hipótesis
MedGemma aislado no superará a XGBoost en datos tabulares completos. El aporte está en (1) el híbrido que enruta o
enriquece con rutas de decisión en la zona de incertidumbre de XGBoost, (2) regímenes few-shot por distrito y
(3) la auditoría de equidad (altitud, rural/urbano, región, lengua materna).

## Reglas metodológicas
- **Sin fuga de datos**: HC53/HW53/HA53 y HC56/HW56/HA56 (hemoglobina) nunca son predictores; ver `FORBIDDEN_FEATURES` en `data.py`.
- **Etiqueta doble**: histórica (HC57) y recalculada desde HC53 + HV040 según la NTS 213-MINSA/DGIESP-2024 (`labels.py`).
- **Probabilidades de tokens**, nunca confianza verbalizada (`medgemma.py`).
- Particiones idénticas para todos los modelos (`data.split`), diseño muestral complejo (pesos `V005`) en la evaluación.

## Estructura
```
configs/default.yaml     rondas, umbrales, semillas
src/boostllm_anemia/
  data.py                carga, cohorte, split, anti-fuga
  labels.py              anemia histórica vs. recalculada (NTS 213-2024)
  baselines.py           XGBoost / regresión logística + SHAP
  serialize.py           TabLLM: fila -> viñeta clínica en español
  boost_paths.py         BoostLLM: rutas raíz-hoja comprimidas
  medgemma.py            inferencia con probabilidades de tokens (+ LoRA stub)
  evaluate.py            AUROC, calibración, equidad por subgrupo, ADS (tau de Kendall)
tests/
data/raw/                microdatos INEI (no versionados)
```

## Pendientes que requieren verificación humana
- Cortes y ajuste de altitud ya verificados contra la Guía OMS 2024 (`labels.py`); falta confirmar que la NTS 213-2024 no se aparte de la OMS.
- Normativa citada en el informe fuente (RM 903-2026/MINSA, D.S. 115-2025-PCM, fechas de vigencia) no verificada.
- Nombres exactos de variables por ronda de ENDES (cambian entre años).
- Licencia HAI-DEF de MedGemma y acceso a los pesos en Hugging Face.

## Instalación
```
pip install -e .[dev]
pytest
```
