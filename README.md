# Vehicle Damage Detection App

A Streamlit web application that uses a deep learning model to classify visible vehicle damage from uploaded images. The app is designed as a clean portfolio-ready project for GitHub, LinkedIn, and CV demonstrations.

The model predicts one of six vehicle damage classes:

- Front Breakage
- Front Crushed
- Front Normal
- Rear Breakage
- Rear Crushed
- Rear Normal

## Features

- Upload vehicle images in JPG, JPEG, PNG, or WEBP format
- Preview the uploaded image before prediction
- Reject non-vehicle images before showing any damage class
- Show prediction confidence as a percentage
- Avoid forced predictions when model confidence is low
- Display clear warning messages for unclear or invalid images
- Handle missing model files, corrupted images, and prediction errors gracefully
- Professional Streamlit interface suitable for portfolio presentation

## Preview

![Vehicle validation screen](preview/Screenshot%202026-06-22%20140745.png)

![Prediction result screen](preview/Screenshot%202026-06-22%20140827.png)

## Model Details

- Architecture: ResNet50 transfer learning
- Framework: PyTorch and Torchvision
- Input size: 224 x 224 RGB image
- Normalization: ImageNet mean and standard deviation
- Training classes were loaded with `torchvision.datasets.ImageFolder`, so the prediction label order is:

```text
F_Breakage, F_Crushed, F_Normal, R_Breakage, R_Crushed, R_Normal
```

The app maps those folder labels to user-friendly class names.

## Validation Logic

The app uses two checks before showing a final damage class:

1. Vehicle image validation: a pretrained ImageNet model checks whether the upload appears vehicle-related.
2. Damage confidence validation: the trained damage model must meet the configured confidence threshold.

Default thresholds are defined in `model_helper.py`:

```python
VEHICLE_CONFIDENCE_THRESHOLD = 0.15
DAMAGE_CONFIDENCE_THRESHOLD = 0.70
```

Increase these values to make the app stricter, or lower them if valid vehicle images are rejected too often.

## Project Structure

```text
Vehicle-damage-detection/
|-- app.py
|-- model_helper.py
|-- model/
|   `-- saved_model.pth
|-- preview/
|   |-- Screenshot 2026-06-22 140745.png
|   `-- Screenshot 2026-06-22 140827.png
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Vehicle-damage-detection.git
cd Vehicle-damage-detection
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open the local URL shown in the terminal.

## Important Notes

- Keep `model/saved_model.pth` in place. The app cannot run predictions without it.
- The first run may download Torchvision pretrained weights for vehicle image validation.
- Do not upload private files, API keys, virtual environments, or cache folders to GitHub.
- The trained damage model was not retrained in this update.

## Limitations

- The model is best suited for clear front or rear vehicle images.
- It may perform poorly on side views, heavily cropped images, unusual lighting, or very low-resolution photos.
- The app rejects many unrelated images, but no automated image validation system is perfect.
- Low-confidence predictions are intentionally blocked instead of forcing a possibly wrong class.
- This project is a demonstration tool and should not replace professional vehicle inspection.

## Technologies Used

- Python
- Streamlit
- PyTorch
- Torchvision
- Pillow
- ResNet50 transfer learning
