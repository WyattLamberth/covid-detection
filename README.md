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
├── scripts/
│   ├── download_data.py      # Downloads dataset from Kaggle
│   └── test_imports.py       # Sanity check for module imports
├── utils/
│   └── paths.py              # Shared paths for the project
├── models/                   # CNN and RNN model definitions
├── notebooks/
│   └── 01-exploration.ipynb  # Initial EDA and prototyping
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

1. Place your `kaggle.json` API key in `~/.kaggle/`
2. Run the script:
   ```bash
   python -m scripts.download_data
   ```

This will download and extract the dataset to `data/raw/`.

---

## 🚧 Work In Progress

Planned next steps:

- [ ] Preprocess images and assign rule-based captions
- [ ] Build CNN encoder + LSTM decoder
- [ ] Train baseline captioning model
- [ ] Implement self-refinement pass
- [ ] Run evaluation and ablation experiments
- [ ] Finalize report and submit

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
- Avoid using external `utils` packages — we use a local `utils/` folder
