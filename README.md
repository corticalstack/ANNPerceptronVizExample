# 🧠 ANN Perceptron Visualization Example

A simple Artificial Neural Network (ANN) Perceptron implementation with comprehensive visualizations for educational purposes.

## 📝 Description

This repository contains a straightforward implementation of a neural network perceptron using Keras, designed to demonstrate fundamental concepts of neural networks with visual aids. The example uses a simple 2D dataset to show how a basic neural network can be trained and evaluated, with detailed visualizations of the data, model architecture, training process, and performance metrics.

## ✨ Features

- 📊 Visualization of 2D input data with target classifications
- 🔄 Implementation of a simple neural network with one hidden layer
- 📈 Training process visualization (accuracy and loss over epochs)
- 📉 Comprehensive model evaluation with confusion matrix
- 🔍 Detailed display of network weights and biases
- 📋 Calculation and display of various performance metrics (precision, recall, etc.)

## 🛠️ Prerequisites

- Python 3.x
- TensorFlow/Keras
- pandas
- scikit-learn
- seaborn
- matplotlib

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ANNPerceptronVizExample.git
cd ANNPerceptronVizExample
```

2. Install the required dependencies:
```bash
pip install tensorflow pandas scikit-learn seaborn matplotlib
```

3. Run the example:
```bash
python ann_perceptron_viz.py
```

4. Observe the visualizations:
   - Initial data scatter plot
   - Model architecture diagram (saved as `model.png`)
   - Training accuracy and loss graphs
   - Confusion matrix heatmap

## 📊 Example Output

When you run the script, you'll see:

1. A scatter plot of the input data showing the distribution of the two classes
2. The model summary in the console
3. A plot of the model accuracy over training epochs
4. A plot of the model loss over training epochs
5. Detailed network weights and biases
6. A confusion matrix and various performance metrics including:
   - True/False Positive/Negative counts
   - Precision, Recall, Accuracy
   - True Positive Rate, True Negative Rate
   - And other classification metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

For more information on the concepts demonstrated in this example:

- [Keras Documentation](https://keras.io/documentation/)
- [Neural Networks Introduction](https://www.tensorflow.org/guide/keras/sequential_model)
- [Confusion Matrix Explained](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
