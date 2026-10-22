# Workflow CI dengan MLflow Projects

Proyek ini melatih ulang model klasifikasi Iris secara otomatis dengan MLflow
Projects dan GitHub Actions. Dataset Iris dimuat dari Scikit-Learn, jadi tidak
memerlukan file dataset eksternal.

## Struktur

```text
workflow-ci/
├── .github/
│   └── workflows/
│       └── ci.yml
├── MLProject/
│   ├── MLproject
│   ├── conda.yaml
│   └── modelling.py
├── .gitignore
└── README.md
```

## Menjalankan MLflow Project secara lokal

Dari folder `workflow-ci`, pasang dependensi lalu jalankan:

```powershell
python -m pip install mlflow scikit-learn
$trackingDb = (Join-Path (Get-Location) "mlflow.db").Replace("\", "/")
$env:MLFLOW_TRACKING_URI = "sqlite:///$trackingDb"
python -m mlflow run .\MLProject --env-manager=local --experiment-name iris-workflow-ci
```

Untuk menjalankan pada environment Conda terpisah, MLflow Project juga dapat
membuat environment berdasarkan file `MLProject/conda.yaml`:

```powershell
python -m mlflow run .\MLProject --experiment-name iris-workflow-ci
```

## Trigger GitHub Actions

Workflow berjalan pada `push` ke branch `main`/`master`, pada pull request ke
branch tersebut, atau secara manual melalui tab **Actions** → **MLflow Project
CI** → **Run workflow**.

Untuk mengaktifkan workflow di GitHub, buat repository dan letakkan **isi folder
ini** sebagai root repository. Workflow GitHub hanya mendeteksi file di
`.github/workflows/` pada root repository; jangan letakkan seluruh folder ini
sebagai subfolder di dalam repository lain.
