# Skin Cancer Classification Using Deep Learning and Genetic Algorithm Optimisation

A multimodal AI system that classifies skin cancer lesion types from 
dermoscopy images combined with patient metadata, with model 
hyperparameters automatically tuned using a custom-built Genetic Algorithm.

---

## Project Overview

Skin cancer diagnosis from images is a challenging classification problem 
due to visual similarity between lesion types. This project builds a 
hybrid model that combines image features extracted from a pretrained 
CNN with structured patient data (age, sex, lesion localisation) to 
improve classification performance.

A Genetic Algorithm is implemented from scratch to automate the search 
for optimal hyperparameters and determine which metadata features 
contribute most to model accuracy.

---

## Key Features

- Multimodal architecture combining image data and patient metadata
- Transfer learning using a pretrained ResNet-18 backbone
- Custom Genetic Algorithm for hyperparameter optimisation and feature selection
- Automated selection of the best learning rate, batch size, hidden 
  layer sizes, and dropout rate
- Full confusion matrix and per-class prediction visualisation

---

## Technologies Used

- Python
- PyTorch / TorchVision — deep learning framework and ResNet-18
- scikit-learn — label encoding, data splitting, evaluation metrics
- Pandas / NumPy — data loading and preprocessing
- Matplotlib / Seaborn — training curves and visualisation
- Google Colab — GPU-accelerated training environment

---

## Model Architecture

The model consists of three components fused at inference time:

1. CNN Branch — ResNet-18 (pretrained on ImageNet) with the final 
   fully connected layer replaced by an identity layer, used as a 
   512-dimensional feature extractor
2. Metadata Branch — A small MLP that processes selected patient 
   features (age, sex, lesion localisation) into a compact embedding
3. Classifier Head — A fully connected classifier that takes the 
   concatenated CNN and metadata features as input

---

## Genetic Algorithm

A custom Genetic Algorithm is implemented to optimise:

| Parameter | Search Space |
|---|---|
| Learning rate | 1e-5 to 1e-2 |
| Batch size | 16, 32, 64 |
| Hidden layer size | 64, 128, 256, 512 |
| Metadata hidden size | 8, 16, 32 |
| Dropout rate | 0.1 to 0.5 |
| Feature selection | Age, Sex, Localisation (on/off) |

The GA uses tournament selection, single-point crossover, and 
random mutation over multiple generations to find the best 
configuration before full model training begins.

---

## Dataset

The project uses a skin cancer dermoscopy dataset with associated 
patient metadata including age, sex, and lesion localisation.

- Images are resized to 224x224 pixels
- Metadata is label-encoded and standardised using StandardScaler
- An 80/20 stratified train/validation split is applied

> The dataset is not included in this repository.
> A similar public dataset is available at
> [ISIC Archive](https://www.isic-archive.com) or
> [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection).

---

## How to Run

1. Clone the repository
```bash
   git clone https://github.com/yourusername/skin-cancer-classification
```

2. Install dependencies
```bash
   pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn
```

3. Mount your Google Drive in Colab and update the paths
```python
   data_dir = 'your/path/to/Skin Cancer'
   csv_path = 'your/path/to/Filtered_metadata.csv'
```

4. Open and run the notebook
```bash
   jupyter notebook COMP1805_Code.ipynb
```

---

## Results

The final model is trained using the hyperparameters selected by the 
Genetic Algorithm. Training accuracy, validation accuracy, training 
loss, and validation loss are tracked across all epochs and visualised 
alongside the GA fitness evolution curve.

---

## Author

**Muhammad Nasif**  
BSc (Hons) Computer Science with Artificial Intelligence  
University of Greenwich  
[LinkedIn](https://linkedin.com/in/muhammad-nasif) | [GitHub](https://github.com/yourusername)
