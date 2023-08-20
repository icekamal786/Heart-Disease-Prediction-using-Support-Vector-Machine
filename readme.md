# Support Vector Machine (SVM) Implementation and Comparison

This project demonstrates the implementation of Support Vector Machines (SVMs) from scratch using Python and compares their performance with SVMs implemented using the Scikit-learn library. The project includes three different kernel functions: Linear, Polynomial, and Radial Basis Function (RBF).

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Dependencies](#dependencies)

## Introduction

Support Vector Machines are powerful machine learning models used for classification and regression tasks. In this project, we implement SVMs from scratch with three different kernel functions and compare their performance with Scikit-learn's SVM implementation. The dataset used for this project is the 'heart' dataset, where the goal is to predict the presence of heart disease.

## Project Structure

The project is structured as follows:

- `svm_scratch.py`: Contains the implementation of SVM class with different kernel functions, cross-validation split function, and hyperparameter tuning.
- `heart.csv`: Dataset containing heart-related features and target labels.
- `README.md`: This README file providing an overview of the project.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have the required dependencies (see [Dependencies](#dependencies)).
3. Run the `svm_scratch.py` script to train and test SVMs using different kernel functions.
4. The script will output classification reports for each SVM, including precision, recall, F1-score, and accuracy metrics.

## Results

After running the `svm_scratch.py` script, you will see the results of training and testing SVMs with different kernel functions. The classification reports will provide insights into the performance of each SVM, including precision, recall, F1-score, and accuracy metrics.

## Dependencies

The project relies on the following libraries:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `sklearn`: For Scikit-learn's SVM implementation and classification report generation.

Ensure you have these libraries installed before running the script.

## License

This project is licensed under the [MIT License](LICENSE).
