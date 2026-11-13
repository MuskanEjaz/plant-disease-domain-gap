# Plant Disease Classification Across Lab and Field Images

This project measures how well a plant disease classifier trained on controlled **PlantVillage** images (lab) transfers to **PlantDoc** images (field), on the 12 crop–disease categories that both datasets share. It then tests ways to close the gap.

**Team:** Muskan Ejaz, Juwairiya Haroon, Zainab Hashmi.

## Headline results

All numbers are on the fixed, held-out test sets: PlantVillage test (2,757 images) and PlantDoc test (99 images). Values are percentages.

| Model | PV acc | PV macro-F1 | PD acc | PD macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| B0 baseline: ResNet-18, frozen, 3 epochs (reproduced) | 92.46 | 92.08 | 33.33 | 34.89 |
| E1 longer training (best epoch 26 of 30) | 95.14 | 95.08 | 33.33 | 32.00 |
| E2 augmentation, frozen ResNet-18 | 89.92 | 89.15 | 35.35 | 35.09 |
| E3-head: field fine-tuning of the head, 856 images (3 seeds) | 74.73 | 72.99 | 49.49 | 48.60 |
| E3-layer4: field fine-tuning of layer4 + head, 856 images (3 seeds) | 48.42 | 43.92 | 59.26 | 59.63 |
| P1 frozen ConvNeXt-Tiny probe, PlantVillage + field | 95.57 | 95.60 | 58.59 | 62.10 |
| **P2 Swin-T joint fine-tuning + flip TTA (3 seeds)** | **98.28** | **98.45** | **68.35** | **69.51** |

* The source-only baseline loses **57.19 percentage points** of macro-F1 from lab to field.
* Longer training (E1) and augmentation alone (E2) do not close the gap.
* Fine-tuning on 856 labelled field images (E3) raises PlantDoc macro-F1 to 59.63%, but PlantVillage falls to 43.92% because the model forgets the lab domain.
* Training on both domains with a stronger backbone keeps both high. The final **Swin-T joint fine-tuning** reaches **98.45 ± 0.21%** PlantVillage and **69.51 ± 1.40%** PlantDoc macro-F1 over three seeds. Without test-time augmentation it scores 98.37% and 71.05%.
* With only 99 field test images, the 95% bootstrap interval of PlantDoc macro-F1 is about ±8 points (see `outputs/experiment_summary.csv` and `outputs/part2/p2_joint_finetuning_runs.csv`).

The baseline was reproduced locally to all four reported decimals. The per-epoch losses 0.7818 / 0.3228 / 0.2458, the validation macro-F1 scores 86.59 / 90.37 / 92.41%, and the test scores 92.46 / 92.08 / 33.33 / 34.89% all match the original Colab run.

## Data and class matching

Training uses the colour PlantVillage images and the official split lists from the [PlantVillage Hugging Face repository](https://huggingface.co/datasets/mohanty/PlantVillage). Field images come from the classification folders of the [PlantDoc repository](https://github.com/pratikkayal/PlantDoc-Dataset). Categories were matched by both crop and disease:

| Crop | Shared diseases |
| --- | --- |
| Apple | Apple scab |
| Corn | Gray leaf spot |
| Grape | Black rot |
| Potato | Early blight, late blight |
| Squash | Powdery mildew |
| Tomato | Bacterial spot, early blight, late blight, leaf mold, mosaic virus, Septoria leaf spot |

Generic crop-only labels were left out. Tomato two-spotted spider mite has no PlantDoc test images and was excluded.

## Data audit

1. Every pilot image was decoded, and none was unreadable.
2. SHA-256 hashes found three PlantVillage training files identical to PlantVillage test images, plus five more copies inside the training pool. All eight were removed. The seed-42 stratified 15% split then gives **9,796 training / 1,729 validation** images.
3. Before any PlantDoc *training* image is used, each one is compared with all 99 PlantDoc test images by SHA-256 and by a 64-bit perceptual difference hash (dHash, Hamming distance ≤ 6). Of the 1,083 PlantDoc training files in the 12 classes:
   * **6** are byte-identical to a test image;
   * **1** duplicates another training file;
   * **4** are re-encoded visual copies of test images that SHA-256 missed. Two of these carry a *different crop label* (tomato vs. potato) in train and test, so PlantDoc contains label noise.

   All flagged images were removed. The remaining 1,071 images were split (stratified, seed 42) into **856 field-adaptation** and **215 field-validation** images. PlantDoc test was never changed.

## Method

* **B0 (baseline):** ImageNet ResNet-18 with the backbone and batch-norm statistics frozen, and a new 12-way linear head. It uses the weights' own preprocessing (224 × 224) and trains for 3 epochs with cross-entropy, Adam (learning rate 0.001) and batch size 32. The checkpoint is chosen by PlantVillage validation macro-F1.
* **E1 (longer training):** the same model trained for 30 epochs on cached frozen features, with the epoch chosen by PlantVillage validation macro-F1.
* **E2 (augmentation):** the same frozen backbone, with the head trained for 15 epochs on augmented PlantVillage images. Augmentations: random resized crop (0.35–1), flips, rotation up to 25°, colour jitter and blur.
* **E3 (limited-field fine-tuning):** starts from B0 and uses 10/25/50/100% of the field adaptation images with 3 seeds. Two variants: head only, or layer4 + head (BN frozen, AdamW). The epoch is chosen by field-validation macro-F1.
* **P1 (backbone comparison):** ResNet-18, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny and Swin-T as frozen feature extractors. Each gets a linear head trained on PlantVillage only, or on PlantVillage + field adaptation images with equal domain weight.
* **P2 (joint fine-tuning):** the backbone with the best mean validation macro-F1 (Swin-T). Its last two stages and the head are fine-tuned on all field adaptation images plus twice as many random PlantVillage images per epoch. Training uses augmentation, label smoothing 0.1, AdamW (1e-4 for the backbone, 1e-3 for the head) and a cosine schedule for 12 epochs. The epoch is chosen by the mean of PlantVillage-val and field-val macro-F1. Test-time augmentation averages each image with its horizontal flip. Three seeds.

Every choice uses validation data only. PlantDoc test is evaluated once per selected checkpoint.

### E3: effect of the amount of field data (mean ± std over 3 seeds)

| Variant | Field images | PlantDoc test macro-F1 | PlantVillage test macro-F1 |
| --- | ---: | ---: | ---: |
| E3-head | 86 | 43.66 ± 1.41 | 71.15 ± 3.05 |
| E3-head | 215 | 45.68 ± 2.25 | 74.46 ± 3.65 |
| E3-head | 430 | 46.26 ± 3.01 | 76.54 ± 2.55 |
| E3-head | 856 | 48.60 ± 0.76 | 72.99 ± 1.14 |
| E3-layer4 | 86 | 42.31 ± 4.22 | 56.10 ± 3.40 |
| E3-layer4 | 215 | 52.08 ± 3.06 | 55.14 ± 9.35 |
| E3-layer4 | 430 | 53.18 ± 3.31 | 57.27 ± 4.17 |
| E3-layer4 | 856 | 59.63 ± 0.71 | 43.92 ± 0.78 |

### P1: frozen backbones (macro-F1, %)

| Backbone | Training data | PV val | Field val | PV test | PD test |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet-18 | PV only | 94.61 | 27.25 | 95.52 | 31.18 |
| ResNet-18 | PV + field | 88.10 | 60.11 | 89.02 | 53.69 |
| ResNet-50 | PV only | 98.03 | 35.29 | 97.59 | 28.90 |
| ResNet-50 | PV + field | 94.97 | 57.44 | 95.39 | 44.87 |
| EfficientNet-B0 | PV only | 98.61 | 32.28 | 97.30 | 39.61 |
| EfficientNet-B0 | PV + field | 95.29 | 62.02 | 95.36 | 57.45 |
| ConvNeXt-Tiny | PV only | 98.49 | 37.28 | 98.24 | 32.16 |
| ConvNeXt-Tiny | PV + field | 94.61 | 66.14 | 95.60 | 62.10 |
| Swin-T | PV only | 98.61 | 40.10 | 98.23 | 34.86 |
| Swin-T | PV + field | 95.35 | 65.43 | 95.76 | 51.68 |

## Field errors and limitations

* The baseline predicted tomato late blight for **58 of 99** PlantDoc images, and only 10 of those were correct. Tomato bacterial spot, leaf mold and mosaic virus had zero recall.
* The baseline is overconfident in the field. Its wrong PlantDoc predictions have a mean softmax confidence of 0.72, against 0.55 for wrong PlantVillage predictions.
* The 99 field test images give wide confidence intervals. Corn gray leaf spot has only four test images, so read per-class scores together with their support.
* Hash checks catch exact and re-encoded copies only. Cropped or edited near-duplicates can still escape, and two test images have conflicting labels.

## How to run

The notebooks download both datasets themselves. The two new notebooks were executed locally (results below). They also contain Colab paths (`/content`), but have not yet been re-run in Colab.

```bash
pip install -r requirements.txt
jupyter notebook
```

1. `Plant_Disease_Domain_Gap.ipynb`: the original baseline notebook (Colab).
2. `Plant_Disease_Domain_Gap_Experiments.ipynb`: data audit, B0 reproduction, E1, E2, E3 and figures. Outputs go to `outputs/`.
3. `Plant_Disease_Domain_Gap_Part2_Field_Improvement.ipynb`: P1 and P2. Outputs go to `outputs/part2/`.

Locally, data is stored in `../data` (set `PDG_ROOT` to change this). `PDG_SMOKE=1` runs a quick check of the code on a small subset. On macOS, six PlantDoc file names collide because the disk is case-insensitive. Restore the second copy with `git show` (one affected image is in the pilot).

Measured run time on an Apple M4 (B0 on CPU, the rest on the M4 GPU via PyTorch MPS): Part 1 ≈ 45 min, Part 2 ≈ 72 min. Environment: Python 3.12, PyTorch 2.14, torchvision 0.29, scikit-learn 1.9, pandas 3.0.

## Files

| Path | Contents |
| --- | --- |
| `results/baseline_colab/` | Original Colab exports (`baseline_test_scores.csv`, `plantdoc_per_class.csv`, `plantdoc_confusion_matrix.png`) |
| `outputs/experiment_summary.csv` | B0–E3 test scores with bootstrap intervals |
| `outputs/e3_*.csv`, `outputs/*_epoch_log.csv` | Field fine-tuning runs and per-epoch logs |
| `outputs/split_counts.csv`, `outputs/near_duplicates.csv` | Split sizes per class, and flagged PlantDoc copies |
| `outputs/plantdoc_*` | Per-class scores and per-image predictions |
| `outputs/part2/` | Backbone comparison, joint fine-tuning runs, per-class scores, predictions |
| `outputs/fig_*.png` | Notebook figures |

Model checkpoints (`*.pt`) are produced by the notebooks and are not committed, because of their size.

## Remaining work (final term)

* A larger field evaluation: cross-validation over the PlantDoc pool, and more field images per class.
* Unsupervised domain adaptation (e.g. DANN) using unlabelled field images.
* Leaf localisation or cropping before classification, and calibration of confidence scores.
* Extending beyond the 12-class pilot to healthy classes.
