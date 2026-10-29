# Workflow CI dengan MLflow Projects

Proyek ini melatih ulang model klasifikasi Breast Cancer Wisconsin
(Diagnostic) secara otomatis dengan MLflow Projects dan GitHub Actions. Dataset
disertakan di dalam repository sehingga workflow tidak perlu mengunduh dataset
saat CI berjalan.

Dataset berisi 569 sampel, 30 fitur numerik, dan target diagnosis
`malignant`/`benign`. Sumber: UCI Machine Learning Repository,
<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>.

## Struktur

```text
workflow-ci/
├── .github/
│   └── workflows/
│       └── ci.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   ├── dataset_preprocessing.csv
│   └── modelling.py
├── .gitignore
└── README.md
```

## Menjalankan MLflow Project secara lokal

Dari folder `workflow-ci`, pasang dependensi lalu jalankan:

```powershell
python -m pip install mlflow pandas scikit-learn
$trackingDb = (Join-Path (Get-Location) "mlflow.db").Replace("\", "/")
$env:MLFLOW_TRACKING_URI = "sqlite:///$trackingDb"
python -c "import mlflow; from pathlib import Path; name='breast-cancer-workflow-ci'; mlflow.get_experiment_by_name(name) or mlflow.create_experiment(name, artifact_location=(Path.cwd() / 'mlartifacts').as_uri())"
python -m mlflow run .\MLProject --env-manager=local --experiment-name breast-cancer-workflow-ci
```

Untuk menjalankan pada environment Conda terpisah, MLflow Project dapat
membuat environment berdasarkan file `MLProject/conda.yaml`:

```powershell
python -m mlflow run .\MLProject --experiment-name breast-cancer-workflow-ci
```

## Trigger GitHub Actions

Workflow berjalan pada `push` ke branch `main`/`master`, pada pull request ke
branch tersebut, atau secara manual melalui tab **Actions** → **MLflow Project
CI** → **Run workflow**.

Untuk mengaktifkan workflow di GitHub, buat repository dan letakkan **isi folder
ini** sebagai root repository. Workflow GitHub hanya mendeteksi file di
`.github/workflows/` pada root repository; jangan letakkan seluruh folder ini
sebagai subfolder di dalam repository lain.

Skrip membaca `MLProject/dataset_preprocessing.csv`, membagi data secara
stratified, lalu menjalankan imputasi median dan standardisasi di dalam pipeline
model. Imputer dan scaler hanya di-fit pada data latih untuk mencegah kebocoran
data. Dataset dicatat sebagai artifact run bersama model dan metrik uji.
GitHub Actions terlebih dahulu memverifikasi 569 sampel, 30 fitur numerik, dan
pemetaan target `malignant`/`benign`. Setelah pelatihan, workflow memeriksa status
run MLflow, empat metrik evaluasi, serta keberadaan model artifact sebelum
mengunggah database tracking, artefak, log, dan dataset sebagai workflow
artifact.
