# AI-Powered Medical Image Analysis System

An AI-powered chest X-ray classification system that uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project combines deep learning with an interactive Streamlit dashboard to demonstrate how medical image classification models can be developed and deployed.

## Features

- **CNN-based classification** — predicts whether an uploaded chest X-ray belongs to the NORMAL or PNEUMONIA class.
- **Interactive web dashboard** — upload an X-ray image and view the model's prediction.
- **Training visualizations** — view training loss and validation accuracy plots.
- **Dataset preview** — explore sample chest X-ray images.
- **Saved model** — uses a trained model so you can run predictions without retraining it.
- **Modular codebase** — separates data preparation, model definition, training, evaluation, and visualization.

## Tech Stack

- Python
- PyTorch and Torchvision
- Convolutional Neural Networks (CNN)
- Streamlit
- Hugging Face Datasets
- NumPy, Matplotlib, Seaborn, and Scikit-learn

## Project Structure

```text
AI-Medical-Image-Analysis/
├── app.py
├── requirements.txt
├── .gitignore
├── models/
│   └── chest_xray_cnn.pth
├── results/
│   ├── dataset_preview.png
│   ├── training_loss.png
│   └── validation_accuracy.png
└── src/
    ├── evaluate.py
    ├── load_data.py
    ├── model.py
    ├── prepare_data.py
    ├── preview_data.py
    └── train.py
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shravaniraut8149-design/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will open in your browser. Upload a chest X-ray image to view the model's predicted class.

## Results and Visualizations

Training and dataset visualizations are available in the `results/` directory:

- `dataset_preview.png` — sample X-ray images from the dataset.
- `training_loss.png` — training loss visualization.
- `validation_accuracy.png` — validation accuracy visualization.

## Model Training

To train the CNN model again, run:

```bash
python src/train.py
```

For model evaluation, run:

```bash
python src/evaluate.py
```

Training and evaluation may take time depending on your hardware and dataset availability.

## Author

**Shravani Raut**

---

If you find this project interesting, consider giving the repository a star.
