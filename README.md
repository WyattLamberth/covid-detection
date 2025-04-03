# 🩺 COVID-19 Chest X-Ray Captioning Project

This project is for our machine learning course final project, due **April 27**. We aim to develop a CNN+RNN model that generates short, interpretable captions for chest X-ray images, identifying whether they are normal or show signs of pneumonia (COVID-related).

---

## 🎯 Project Goals

- Convert chest X-ray classification into a **captioning task**
- Use a **CNN encoder** (e.g., ResNet) to extract image features
- Use an **RNN decoder** (e.g., LSTM) to generate medical-style captions
- (Bonus) Add a **self-refinement loop**: the model revises its own caption based on its first output
- Explore **evaluation metrics**, **visualizations**, and **model comparisons** to demonstrate learning

---

## 📅 Important Dates

- **Project Due**: April 27  
- **Penalty**: 20% per day late (No grade if more than 5 days late)
- **Final Deliverable**: PDF report + source code  
- Each member must **submit individually**

---

## 🧠 Project Structure

```
covid-detection/
├── data/
│   └── raw/                  # Original dataset (downloaded via script)
│       ├── Chest_xray_Corona_Metadata.csv  # Original metadata
│       ├── Chest_xray_Corona_dataset_Summary.csv  # Dataset summary
│       └── Coronahack-Chest-XRay-Dataset/
│           └── Coronahack-Chest-XRay-Dataset/
│               ├── image_captions.csv  # Generated captions for training
│               ├── train/              # Training images
│               └── test/               # Testing images
├── scripts/
│   ├── collate.py            # Collates data for model training
│   ├── dataset.py            # Dataset class for PyTorch
│   ├── download_data.py      # Downloads dataset from Kaggle
│   ├── evaluate.py           # Model evaluation
│   ├── model.py              # CNN+RNN model definitions
│   ├── predict.py            # Generate predictions
│   ├── preprocess_and_caption.py  # Processes images and generates captions
│   ├── test_imports.py       # Sanity check for module imports
│   └── train.py              # Model training
├── utils/
│   ├── paths.py              # Shared paths for the project
│   └── tokenizer.py          # Text tokenization utilities
├── notebooks/
│   ├── dataset_exploration.ipynb  # Initial EDA and dataset analysis
│   └── preprocess_and_caption_updated.ipynb  # Caption generation development
├── run_workflow.sh           # Main workflow script
├── requirements.txt
├── README.md
└── .venv/
```

---

## 🛠️ Environment Setup

We use Python `venv` and [`uv`](https://github.com/astral-sh/uv) for dependency management.

### Setup Steps

1. Clone the repository  
2. Create and activate a virtual environment:
   ```bash
   uv venv -p 3.9
   source .venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

---

## 📥 Download the Dataset

We use the [CoronaHack Chest X-ray Dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset).

To download:

1. Place your `kaggle.json` API key in `~/.kaggle/`  (check https://www.kaggle.com/docs/api#authentication for Windows instructions)
2. Run the script:
   ```bash
   python -m scripts.download_data
   ```

This will download and extract the dataset to `data/raw/`.

---

## 🔄 Running the Workflow

We've created a workflow script to handle the entire pipeline. Run:

```bash
./run_workflow.sh
```

### Available Options

- `--no-download`: Skip dataset download (use if already downloaded)
- `--no-captions`: Skip caption generation step
- `--no-train`: Skip model training
- `--evaluate`: Run model evaluation

Example:
```bash
./run_workflow.sh --no-download --evaluate
```

---

## 📊 Dataset Overview

The dataset contains chest X-ray images categorized as:
- Normal (1,576 images)
- Pneumonia (4,334 images)
  - Bacterial pneumonia (2,777 images)
  - Viral pneumonia (1,493 images)
  - COVID-19 (58 images)
  - Other viral types (11 images)

Our preprocessing generates medical-style captions for each image based on these classifications:

| Classification | Generated Caption |
|----------------|------------------|
| Normal | "No signs of pneumonia." |
| Bacterial Pneumonia | "Pneumonia likely due to bacterial infection." |
| Viral Pneumonia | "Pneumonia likely due to viral infection." |
| COVID-19 | "Lung opacity consistent with COVID-19." |
| Other Viral | "Signs of pneumonia, likely viral origin." |

---

## 📓 Notebooks

### Dataset Exploration
The [dataset_exploration.ipynb](notebooks/dataset_exploration.ipynb) notebook provides:
- Analysis of the dataset structure and metadata
- Handling of missing values and duplicates
- Visualization of label distributions

### Caption Generation
The [preprocess_and_caption_updated.ipynb](notebooks/preprocess_and_caption_updated.ipynb) notebook:
- Loads and cleans the metadata
- Implements rule-based caption generation
- Creates and saves image paths and their associated captions for training

---

## 📚 Report Requirements

Our final report will follow the ACM SIG format and include:

- Group members & contributions
- Introduction and problem statement
- Literature review (3+ papers)
- ML models, methods, training
- Results and evaluation
- Conclusion and contributions

---

## 👥 Team Members

- Wyatt Lamberth  
- Gabriel Zermeno
- Tien Phu Tran

---

## 📎 Notes

- Use `python -m` to run scripts inside the project structure (e.g., `python -m scripts.download_data`)
- All paths are managed in `utils/paths.py` for consistency across notebooks and scripts
- Caption generation is rule-based, creating descriptive text for each chest X-ray based on its metadata
- The preprocessing pipeline handles image loading, validation, and caption generation