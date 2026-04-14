import json
import os
import warnings

warnings.filterwarnings("ignore")

from uci_tabular_pipeline import pipeline

results_file = "out/results.json"

datasets = [
    # 3 class datasets
    {
        "name": "student-dropout",
        "path": "data/student-dropout/student-dropout.csv",
        "label_col": -1,
        "sep": ";",
        "header": 0,
    },
    {
        "name": "wine",
        "path": "data/wine/wine.data",
        "label_col": 0,
        "sep": ",",
        "header": None,
    },
    {
        "name": "seeds",
        "path": "data/seeds/seeds.data",
        "label_col": -1,
        "sep": r"\s+",
        "header": None,
    },
    {
        "name": "cmc",
        "path": "data/cmc/cmc.csv",
        "label_col": -1,
        "sep": ",",
        "header": None,
    },
    # 4 class datasets
    {
        "name": "wall-robot",
        "path": "data/wall-robot/wall-robot.csv",
        "label_col": -1,
        "sep": ",",
        "header": None,
    },
    # 6 class datasets
    {
        "name": "satellite",
        "path": "data/satellite/satellite.data",
        "label_col": -1,
        "sep": r"\s+",
        "header": None,
    },
    {
        "name": "glass",
        "path": "data/glass/glass.data",
        "label_col": -1,
        "sep": ",",
        "header": None,
        "id_column": 0,
    },
    # 7 class datasets
    {
        "name": "segmentation",
        "path": "data/segmentation/segmentation.data",
        "label_col": -1,
        "sep": r"\s+",
        "header": None,
    },
    # 10 class datasets
    {
        "name": "mfeat-factors",
        "path": "data/mfeat-factors/mfeat-factors.csv",
        "label_col": -1,
        "sep": ",",
        "header": None,
    },
    {
        "name": "mfeat-zernike",
        "path": "data/mfeat-zernike/mfeat-zernike.csv",
        "label_col": -1,
        "sep": ",",
        "header": None,
    },
    {
        "name": "optdigits",
        "path": "data/optdigits/optdigits.data",
        "label_col": -1,
        "sep": ",",
        "header": None,
    },
    {
        "name": "pendigits",
        "path": "data/pendigits/pendigits.data",
        "label_col": -1,
        "sep": ",",
        "header": None,
    },
]

base_args = {
    "test_size": 0.3,
    "calibration_size": 0.0,
    "ensemble_size": 25,
    "random_state": 42,
    "sampling_mode": "shared",
}

os.makedirs("out", exist_ok=True)

if os.path.exists(results_file):
    with open(results_file, "r") as f:
        results = json.load(f)
else:
    results = {"logistic": {}, "tabpfn": {}}

for model_type in ["logistic", "tabpfn"]:
    for ds in datasets:
        key = ds["name"]
        if (
            key in results[model_type]
            and results[model_type][key].get("accuracy") is not None
        ):
            print(f"Skipping {model_type} on {key} (already done)")
            continue

        print(f"\nRunning {model_type} on {key}")
        try:
            ds_args = {
                k: v for k, v in ds.items() if k not in ["name", "path", "label_col"]
            }
            acc, cov = pipeline(
                data_path=ds["path"],
                label_column=ds["label_col"],
                base_model=model_type,
                **base_args,
                **ds_args,
            )
            results[model_type][key] = {"accuracy": acc, "coverage": cov, "error": None}
        except Exception as e:
            print(f"Error: {e}")
            results[model_type][key] = {
                "accuracy": None,
                "coverage": None,
                "error": str(e),
            }

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for model_type in ["logistic", "tabpfn"]:
    print(f"\n{model_type.upper()}:")
    for ds_name, res in results[model_type].items():
        acc = f"{res['accuracy']:.3f}" if res["accuracy"] is not None else "N/A"
        cov = f"{res['coverage']:.3f}" if res["coverage"] is not None else "N/A"
        print(f"  {ds_name}: Acc={acc}, Cov={cov}")
