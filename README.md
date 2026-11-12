# Plant Disease Classification Across Lab and Field Images

This project measures how well a plant disease classifier trained on controlled PlantVillage images transfers to PlantDoc images. The midterm experiment evaluates one pretrained ResNet-18 checkpoint on the same 12 crop–disease categories in both datasets.

## Midterm results

| Held-out test set | Images | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| PlantVillage | 2,757 | 92.46% | 92.08% |
| PlantDoc | 99 | 33.33% | 34.89% |

The observed lab-to-field gap was **57.19 percentage points in macro-F1**. The best checkpoint was selected at epoch 3 with **92.41% PlantVillage validation macro-F1**. Macro-F1 averages per-class F1 scores equally, while accuracy measures the fraction of all images classified correctly.

## Data and class matching

Training uses color images from PlantVillage. PlantDoc provides the held-out field-domain evaluation images for the baseline. Categories were matched by both crop and disease. The pilot includes disease categories with test support in both datasets.

| Crop | Shared diseases |
| --- | --- |
| Apple | Apple scab |
| Corn | Gray leaf spot |
| Grape | Black rot |
| Potato | Early blight, late blight |
| Squash | Powdery mildew |
| Tomato | Bacterial spot, early blight, late blight, leaf mold, mosaic virus, Septoria leaf spot |

Generic crop-only labels were left outside the pilot. Tomato two-spotted spider mite had no PlantDoc test examples and was excluded from the paired evaluation.

Dataset sources [PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) and [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset).

## Method

1. **Audit the images.** Check mapped file paths and decode the pilot images. Use SHA-256 hashes to detect byte-identical copies.
2. **Clean the source pool.** Exclude three PlantVillage training files that matched held-out test images and five additional duplicate copies within the source pool. Keep both test sets fixed.
3. **Build validation data.** Take a stratified 15% validation split from the cleaned PlantVillage training pool using seed 42. The resulting split has **9,796 training** and **1,729 validation** images. Both splits cover all 12 classes.
4. **Train the classifier.** Use ImageNet-pretrained ResNet-18 with its backbone and batch-normalization statistics frozen. Replace the final layer with a 12-output linear classifier. Apply the pretrained weights' preprocessing to RGB images to produce 224 × 224 inputs. Train the head for three epochs with cross-entropy loss, Adam at a learning rate of 0.001, and batches of 32.
5. **Select and evaluate the checkpoint.** Select the highest PlantVillage validation macro-F1. Evaluate that same checkpoint on the two held-out sets with the same class IDs and preprocessing.

| Epoch | Mean training loss | PlantVillage validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.7818 | 86.59% |
| 2 | 0.3228 | 90.37% |
| 3 | 0.2458 | 92.41% |

## Field errors and limitations

- The model predicted tomato late blight for **58 of 99** PlantDoc images. **10 of those 58** predictions were correct.
- Tomato bacterial spot, leaf mold, and mosaic virus had zero recall across **25 field test images** combined.
- Corn gray leaf spot scored perfectly on **four** test images. Per-class scores should be read alongside their support counts.
- This is a preliminary result for a 12-class disease-only pilot. The two datasets can differ in several ways, and this experiment does not isolate the effect of any single image characteristic.
- SHA-256 detects exact file duplicates. Resized copies, re-encoded images, and other near-duplicates can escape the check.
- Validation macro-F1 was still improving at epoch 3. The baseline does not establish convergence or quantify how much field fine-tuning would recover.

## Run in Google Colab

1. Download `Plant_Disease_Domain_Gap.ipynb` and open it in Google Colab. Select a GPU runtime before running the notebook if one is available. The baseline also runs on CPU.
2. Run the cells in order, starting with dataset download and label mapping. The notebook uses PyTorch, torchvision, Pillow, pandas, scikit-learn, matplotlib, and huggingface_hub.
3. Check the class coverage and duplicate audit, then run training. Confirm that the best checkpoint is saved before running held-out inference.
4. Run the export cells and download the results before ending the runtime.

| File | Contents |
| --- | --- |
| `Plant_Disease_Domain_Gap.ipynb` | Dataset preparation, training, evaluation, and export |
| `baseline_test_scores.csv` | Overall test accuracy and macro-F1 for both datasets |
| `plantdoc_per_class.csv` | PlantDoc precision, recall, F1, and support for each class |
| `plantdoc_confusion_matrix.png` | PlantDoc counts by true and predicted class |
| `resnet18_baseline.pt` | Checkpoint generated in Colab by the training cell |

## Planned experiments

Compare an augmentation baseline and limited-field fine-tuning against the frozen source-only baseline. Use PlantDoc training data for field adaptation and validation-based model selection. Reserve the field test set for evaluation after experiment settings have been chosen.

## Team

Muskan Ejaz, Juwairiya Haroon, and Zainab Hashmi.
