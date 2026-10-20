# Training and calibration scripts

These scripts were used to **build** the models in `Code/models/`. The app does **not** need them.
They need the full image datasets, which are not in this repository (see the main README).

Run them from the repository root, e.g. `python Code/training/train_stage13.py`.

| Script | What it does | Writes |
|---|---|---|
| `train_stage10.py` | 8-condition model, EfficientNet-B0 at 448 px (GPU) | `Code/models/stage10_*.pt`, `Data/stage10_*` |
| `train_stage12.py` | Second network at 512 px with contrast enhancement + ensemble | `Code/models/stage12_*.pt`, `Data/stage12_*` |
| `train_stage13.py` | Retrains on ODIR-5K + RFMiD, picks the final model on validation | `Code/models/stage13_*.pt`, `Data/stage13_*`, `Data/multiclass_final_choice.json` |
| `test_external_rfmid.py` | Tests the ODIR-only models on the RFMiD dataset (other clinic) | `Data/external_rfmid_*.csv` |
| `calibrate_quality.py` | Sets the photo-quality limits from usable photos | `Data/quality_*` |
| `benchmark_models.py` | Trains 7 standard networks on the same split + the learning curve | `Code/models/benchmark_*.pt`, `Data/benchmark_results.csv`, `Data/learning_curve_results.csv` |
| `ensemble_search.py` | Tries combinations of networks, picks the final model on validation | `Data/stage14_*`, `Data/ensemble_search.csv`, `Data/multiclass_final_choice.json` |
| `calibrate_triage.py` | Sets the batch-screening limits on validation photos and tests them once | `Data/triage_*` |

Expected data folders (in the repository root): `preprocessed_images/` (ODIR-5K, 512 px),
`ODIR-5K/ODIR-5K/Training Images/`, and `external/RFMiD/{train,val,test}/` with `labels.csv`.

The earlier steps (data exploration, cleaning, tabular baseline, first CNN, Myopia fusion model) are in
`Code/notebooks/01_...` to `10_...`.
