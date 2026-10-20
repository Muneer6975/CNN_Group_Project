# Person Image Binary Classification — CNN Project

## Goal
Classify an image into exactly one of two categories:
- `single_person`: exactly one visible person
- `multiple_people`: at least two visible people

This is image-level classification, not object detection and not exact people counting.

## 1. Install
Use Python 3.10–3.12 in a virtual environment (TensorFlow support varies by OS/Python version).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare the dataset
Place image files in the matching folders:
```
dataset/
  train/
    multiple_people/
    single_person/
  validation/
    multiple_people/
    single_person/
  test/
    multiple_people/
    single_person/
```
Use a reasonably balanced dataset. Suggested starting point: at least 200–500 images per class if possible; more varied images are preferable. The train/validation/test sets must contain different images. Do not put near-duplicates or images from the same photo burst across splits. Confirm that you have permission to use the images and follow their licence/terms.

Labels must follow the project definition consistently. If an image has no visible person, is ambiguous, or contains only a partial/unclear crowd, either exclude it or define a consistent rule before labelling.

## 3. Train and evaluate
From the project root:
```bash
python src/train.py
```
The script saves the best model, training log, accuracy/loss curves, confusion matrix, and JSON test metrics under `outputs/`.

## 4. Predict a new image
```bash
python src/predict.py path/to/your_image.jpg
```

## Important reporting rule
Do not invent accuracy, precision, recall, F1, dataset counts, or training time. Run the code, then copy the real values from `outputs/test_results.json` and the plots into the report. The report template explicitly marks results that must be filled in after execution.

## Dataset sources
You may create a small manually labelled dataset from images you have permission to use, or use a public dataset with suitable licence/terms. COCO is a well-known object detection/segmentation dataset, but it may require filtering and label conversion for this binary task. Record the dataset name, URL, access date, licence, number of images, and split counts in your final report.
