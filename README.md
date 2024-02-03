# MultiModelClassifier

The MultiModelClassifier project aims to compare the performance of various machine learning classification models, including Support Vector Machines (SVM), Random Forest, Gradient Boosting, and Decision Tree classifiers, on a given dataset. The project evaluates these models based on their accuracy scores to identify which model performs best for the specific classification task.

## Getting Started

These instructions will guide you on how to set up and run the MultiModelClassifier project on your local machine.

### Prerequisites

Ensure you have the following Python packages installed:
- pandas
- scikit-learn
- joblib

You can install these packages using pip by running:

```bash
pip install pandas scikit-learn joblib

Installation
Clone the repository to your local machine:

git clone https://github.com/merttunayilmaz/MultiModelClassifier.git

Navigate to the cloned repository:
cd MultiModelClassifier

Usage
To run the project, execute the main script:

bash
Copy code
python main.py

This script will perform the following tasks:

Load and preprocess data from Data/classification_dataset.csv.
Scale the features using StandardScaler.
Split the data into training and test sets.
Train and evaluate SVM, Random Forest, Gradient Boosting, and Decision Tree models.
Print out accuracy scores for each model and rank them.
Save the Random Forest model using joblib.
Contributing
If you wish to contribute to the MultiModelClassifier project, please fork the repository and submit a pull request with your proposed changes.

License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
