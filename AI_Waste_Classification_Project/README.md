# AI Waste Classification & Smart Recycling System
**Class 11–12 | Python | TensorFlow/Keras | CNN Transfer Learning | Streamlit**

An end-to-end image classification project that recognises six waste categories:
`cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.

## What is included
- Dataset downloader for the public TrashNet image dataset (2,527 resized images).
- Dataset preparation with stratified train/validation/test split (70/15/15).
- Transfer-learning model using MobileNetV2 (ImageNet weights) and a custom classification head.
- Training history plots, test metrics, classification report and confusion matrix.
- Streamlit app for image upload and live webcam capture (if browser/device permissions allow).
- Recycling guidance for each predicted category.
- Prediction confidence and top-3 alternatives.
- A detailed student project report template and viva questions.

## 1. Requirements
Recommended: Python 3.10 or 3.11. A GPU is optional; CPU training works but may take longer.
Install Git only if you want to clone the project; it is not required to run this ZIP.

## 2. Setup (Windows PowerShell)
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 3. Download the dataset
```powershell
python download_dataset.py
```
The script downloads the resized TrashNet archive and extracts it into `data/raw/dataset-resized/`.
Dataset source: Gary Thung & Mindy Yang, Stanford CS229 TrashNet.
The archive is about 43 MB. Internet is needed only for this step.

## 4. Prepare data
```powershell
python prepare_data.py
```
This creates:
- `data/split/train/<class>/`
- `data/split/val/<class>/`
- `data/split/test/<class>/`

It uses a reproducible stratified split and does not alter the original dataset.

## 5. Train the model
```powershell
python train.py
```
The best model is saved to `models/waste_classifier.keras`; class labels are saved to
`models/class_names.json`. Training charts and metrics are saved in `outputs/`.

For a quick classroom demo:
```powershell
python train.py --epochs 5 --batch-size 32
```
For a more thorough run:
```powershell
python train.py --epochs 15 --batch-size 32
```
Training time and accuracy depend on the computer, dataset split, and random seed. Do not claim a fixed accuracy before running the evaluation.

## 6. Launch the application
```powershell
streamlit run app.py
```
Open the local URL shown in the terminal (usually http://localhost:8501).
Upload a JPG/PNG image or use the camera widget. The app displays the predicted class,
confidence, top-3 classes, and a practical recycling suggestion.

## Project workflow
1. Collect labelled waste images.
2. Resize/augment images and split into train, validation and test sets.
3. Train MobileNetV2 with transfer learning.
4. Evaluate on a held-out test set.
5. Deploy a simple interactive application.
6. Test with new real-world photos and discuss limitations.

## Dataset note
TrashNet is a useful educational starter dataset, not a complete representation of all waste.
Its images were mostly photographed against a plain background. Real-world lighting,
overlapping objects, dirty packaging, unusual angles and mixed waste can reduce accuracy.
For a stronger school demonstration, add 20–50 original photos per category, label them
carefully, and retrain/evaluate. Avoid putting sharp, hazardous or contaminated waste
in front of students.

## Citation
Thung, G. & Yang, M. (2016). *Classification of Trash for Recyclability Status*.
Stanford CS229 project. Dataset: https://github.com/garythung/trashnet
Dataset archive mirror: https://huggingface.co/datasets/garythung/trashnet

## Troubleshooting
- `No dataset found`: run `python download_dataset.py` first.
- TensorFlow installation error: use Python 3.10/3.11 (64-bit), then reinstall requirements.
- Model missing in app: train first with `python train.py`.
- Slow CPU: reduce image size or epochs; try batch size 16.
- Low confidence: use a clearer single-item photo, improve lighting, or add more labelled examples.

## Responsible use
This is an educational prototype, not a certified waste-management or safety system.
The app's recycling suggestions are general. Follow local municipal recycling rules.
