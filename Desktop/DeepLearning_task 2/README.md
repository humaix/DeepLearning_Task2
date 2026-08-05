# Deep Learning Framework

A modular Deep Learning Framework built with TensorFlow and Python that supports multiple neural network architectures using a common training pipeline.

## Features

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transfer Learning (MobileNetV2)
- Factory Design Pattern
- Generic Training Pipeline
- Config-based Model Selection
- Automatic Evaluation
- Automatic Training Visualization
- Model Checkpoint Saving
- Early Stopping

---

## Project Structure

```
DeepLearning_Task2
│
├── configs/
├── data/
├── models/
├── outputs/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── factory/
│   ├── models/
│   ├── pipeline/
│   ├── trainer/
│   └── utils/
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Supported Models

| Model | Dataset |
|--------|---------|
| ANN | Breast Cancer |
| CNN | MNIST |
| RNN | IMDB Reviews |
| LSTM | IMDB Reviews |
| Transfer Learning | Cats vs Dogs |

---

## Installation

Clone the repository

```bash
git clone https://github.com/humaix/DeepLearning_Task2.git
```

Move into the project

```bash
cd DeepLearning_Task2
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Choose any configuration file.

Example:

```python
config = ConfigLoader(
    "configs/transfer_config.yaml"
).load()
```

Run

```bash
python main.py
```

---

## Outputs

The project automatically generates:

- Trained model
- Evaluation metrics
- Confusion Matrix
- Accuracy
- Training graphs

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- PyYAML

---

## Design Pattern

This project follows a modular architecture using:

- Factory Pattern
- Object-Oriented Programming
- Generic Training Pipeline

---

## Future Improvements

- Vision Transformer (ViT)
- EfficientNet
- Hyperparameter Tuning
- TensorBoard Support
- ONNX Export
- Multi-GPU Training

---

## Author

**Humaiz Ahmed**