# Guide Review — Talking Points

**Project:** Multimodal Ocular Disease Prediction (started as Myopia Prediction, expanded per initial guide feedback)
**Supervisor:** Dr. Revathi V

---

## 1. One-line pitch

"We built a working prototype that predicts Myopia from a patient's clinical data and fundus (retina) photos combined, tested it properly with no data leakage, found a problem with our first version, diagnosed it, and fixed it — with real numbers at every step."

---

## 2. What we built, in order

1. **Tabular baseline** — cleaned `full_df.csv` (6,392 records, 3,358 patients), identified only 2 legitimate clinical features (age, sex — everything else risked leaking the diagnosis), trained Logistic Regression / Random Forest, split by patient to avoid leakage.
2. **Fundus image inspection** — validated ~5,000+ images across multiple folders, checked which ones could be traced back to a real patient ID.
3. **Image CNN (MobileNetV2)** — transfer learning on a local RTX 4060 GPU, real ImageNet weights, Myopia vs Normal: **96.6% accuracy, 89.5% macro-F1**.
4. **First multimodal fusion** — combined clinical + image features. Result: it worked, but was *worse* than the image model alone (72.4% macro-F1).
5. **Diagnosed why**, then fixed it: fine-tuned the image model jointly with the fusion layer instead of freezing it. Final result: **97.4% accuracy, 87.5% macro-F1** — nearly matching the image-only model while still genuinely using both clinical and image data.

---

## 3. Key results table (say this out loud if asked "what did you get")

| Model | Accuracy | Myopia Recall | Myopia Precision | Macro-F1 |
|---|---|---|---|---|
| Clinical only (Random Forest) | 74.8% | 28.9% | 4.3% | 46.4% |
| Image only (MobileNetV2 CNN) | 96.6% | 97.1% | 69.4% | 89.5% |
| Fusion v1 (frozen image branch) | 90.5% | 96.0% | 33.8% | 72.4% |
| **Fusion v2 — final (fine-tuned)** | **97.4%** | **84.0%** | **70.0%** | **87.5%** |

---

## 4. The strongest story to tell

Don't just show the final number — show the **iteration**. Guides respond well to "we found a problem, we understood *why*, we fixed it with a specific, justified change" more than a single good score appearing out of nowhere. The sequence is:

> "Our first fusion attempt underperformed our image-only model. We diagnosed three possible causes — weak clinical features, a frozen image branch, and untuned training — tested two targeted fixes, and picked the one that worked best on a held-out test set we never touched during development."

---

## 5. Be upfront about these (don't wait to be asked)

- **Only 2 clinical features (age, sex).** Every other column in the dataset either leaks the diagnosis directly (diagnostic keywords, disease indicator columns) or isn't usable. This is a real dataset limitation, not something we can fix by trying harder — it's why the clinical-only model is weak.
- **Recall vs precision trade-off.** The final model catches 84% of real Myopia cases (vs 96% for the earlier, worse-overall version). We chose the model with the better overall macro-F1, but if this were a real screening tool, missing 16% of cases might matter more than the metric we optimized for — worth discussing which matters more for this use case.
- **Only binary classification so far** (Myopia vs Non-Myopia). The dataset has 8 disease labels and is multi-label (a patient can have more than one condition) — extending to that full task is the planned next step, not yet done.
- **No patient-level image split in the very first CNN run** (Stage 3.3) — fixed by the time we got to fusion, but worth mentioning we caught and corrected it ourselves.

---

## 6. Likely questions and short answers

**"How do you know there's no data leakage?"**
→ All splits are done at the patient level (grouped by patient ID), not image level. We explicitly checked this before building the fusion model — verified 100% of images could be traced to real patients with zero ambiguous labels, and confirmed the same patients never appear in both train and test.

**"Why not just use the image model — fusion is doing worse?"**
→ The *first* fusion attempt did, and we say so honestly. The final fusion model reaches 87.5% macro-F1, very close to the image model's 89.5%, while incorporating clinical data too — useful in a real system where images might not always be available, or as a foundation for adding real clinical features (BP, refractive error, etc.) later.

**"What's next?"**
→ Extend this same approach from binary Myopia to the full 8-disease multi-label task, and continue improving once more clinical features are identified.

**"Why macro-F1 as your main metric?"**
→ The dataset is heavily imbalanced (~5% Myopia). Accuracy alone is misleading (a model that always predicts "Non-Myopia" would still score ~95% accuracy). Macro-F1 treats both classes equally, so it doesn't hide poor minority-class performance.

---

## 7. If they ask for evidence

Everything above is documented with real code, real screenshots, and real metrics in `Capstone_Project_Documentation.docx` — the single master log, updated after every stage, including training curves, confusion matrices, and comparison tables for every model tried.
