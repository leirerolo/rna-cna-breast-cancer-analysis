# rna-cna-breast-cancer-analysis

Proyecto de Machine Learning sobre heterogeneidad molecular en cáncer de mama (TCGA-BRCA).

**Pregunta general:** ¿Qué aportan el análisis individual y la integración de perfiles moleculares (RNA y CNA) a la caracterización de la heterogeneidad del cáncer de mama en TCGA-BRCA?

## Preguntas por tarea
- **Clasificación:** ¿con qué rendimiento pueden predecirse los subtipos moleculares a partir de RNA, CNA y RNA + CNA, y qué aporta la integración?
- **Regresión:** ¿en qué medida RNA, CNA y su integración permiten estimar la carga mutacional tumoral (TMB) en muestras no vistas en entrenamiento?
- **Clustering:** ¿los tumores con agrupaciones discordantes entre RNA y CNA forman una población molecularmente distinta y con diferente pronóstico?

## Datos
Estudio **Breast Invasive Carcinoma (TCGA, PanCancer Atlas)** de cBioPortal
(`brca_tcga_pan_can_atlas_2018`). Los datos NO se suben al repositorio:
cada persona los descarga desde https://www.cbioportal.org/datasets,
los descomprime y coloca el contenido en `data/raw/`.

## Estructura
```
data/
  raw/         # datos originales descomprimidos (no se sube)
  processed/   # datos ya procesados (no se sube)
scripts/       # un script por bloque de trabajo, en orden numérico
results/       # tablas y figuras para el informe
```

## Puesta en marcha (Windows)
1. Clonar el repositorio y entrar en la carpeta.
2. Crear el entorno: `py -m venv venv`
3. Activarlo: `venv\Scripts\activate`
4. Instalar dependencias: `pip install -r requirements.txt`
5. Descargar los datos a `data/raw/` (ver arriba).

## Flujo de trabajo
- `git pull` antes de empezar a trabajar.
- Un script por bloque, y cada persona en el suyo para evitar conflictos.
- Al terminar: `git add .`, `git commit -m "qué has hecho"`, `git push`.

## Equipo
- Persona 1 —
- Persona 2 —
- Persona 3 —
- Persona 4 —

Entrega: 12 de diciembre de 2026.
