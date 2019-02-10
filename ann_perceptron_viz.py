# Author    Jon-Paul Boyd
# What      Simple Artificial Neural Networks Perceptron Example with vizualization
# Dataset   Simple 2 vector with target

from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt



data = [[2,1,0],[2,5,0],[7,2,0],[3,6,1],[1,2,1],[5,4,0],[6,5,0],[2,3,1],[3,3,0],[3,4,0],[4,4,1],[5,5,1],[5,6,1],[6,7,1]]
dataset_full = pd.DataFrame(data,columns=['vector1','vector2', 'target'],dtype=float)

X_train = dataset_full.drop(['target'], axis = 1)
y_train = dataset_full[['target']].copy()


# Scatterplot to vizualise vectors with labelled target - not a linearly seperable problem
ax = sns.scatterplot(x="vector1", y="vector2",  hue="target", data=dataset_full)
plt.show()


# Configure simple network
classifier = Sequential()

# Input layer with 2 inputs, hidden layer with 1 neuron
classifier.add(Dense(name="dense_one", output_dim = 1, init = 'uniform', activation = 'relu', input_dim = 2))

# Output layer - sigmoid good for binary classification
classifier.add(Dense(name="dense_two", output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Binary cross entropy good for binary classification
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN
history = classifier.fit(X_train, y_train, batch_size = 14, nb_epoch = 20, verbose=1)

# Output network model image and config
classifier.summary()
classifier.get_config()
from keras.utils.vis_utils import plot_model
plot_model(classifier, to_file='model.png', show_shapes=True, show_layer_names=True)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

tp = 0
tn = 0 
fp = 0
fn = 0

# Calculate label value from network using perceptron weights and bias
weights_hidden, bias_hidden = classifier.layers[0].get_weights()
weights_output, bias_output = classifier.layers[1].get_weights()
print('=' * 80)
print('Network Weights & Bias')
print('1st hidden layer - 2 neurons - weights\tn1 {}, n2 {}\t'.format(weights_hidden[0,0], weights_hidden[1,0]))
print('1st hidden layer - 2 neurons - bias\t{}'.format(bias_hidden[0]))
print('Output layer - weight\t\t\t{}'.format(weights_output[0,0]))
print('Output layer - bias\t\t\t{}'.format(bias_output[0]))

for vector1, vector2, target in data:
    hidden_layer = ((vector1 * weights_hidden[0,0]) + (vector2 * weights_hidden[1,0]) + bias_hidden[0])
    output_target = (hidden_layer * weights_output) + bias_output
    if (output_target < 0 and target == 0):
        tn += 1
    elif (output_target > 0 and target == 1):
        tp += 1
    elif (output_target < 0 and target == 1):
        fn += 1
    elif (output_target > 0 and target == 0):
        fp += 1

print('=' * 80)
print('Confusion Matrix')
print('True positive\t\t\t\t', tp)
print('False positive\t\t\t\t', fp)
print('False negative\t\t\t\t', fn)
print('True negative\t\t\t\t', tn)

print('=' * 80)
print('Stats')
# Sensitivity, hit rate, recall, or true positive rate
try:
    tpr = tp/(tp+fn)
except:
    tpr = 0
print('True positive rate\t\t\t', tpr)

# Specificity or true negative rate
try:
    tnr = tn/(tn+fp) 
except:
    tnr = 0
print('True negative rate\t\t\t', tnr)

# Precision or positive predictive value
try:
    ppv = tp/(tp+fp)
except:
    ppv = 0
print('Positive predicitve value (precision)\t', ppv)

# Negative predictive value
try:
    npv = tn/(tn+fn)
except:
    npv = 0
print('Negative predicitve value\t\t', npv)

# Fall out or false positive rate
try:
    fpr = fp/(fp+tn)
except:
    fpr = 0
print('False positive rate (fallout)\t\t', fpr)

# False negative rate
try:
    fnr = fn/(tp+fn)
except:
    fnr = 0
print('False negative rate\t\t\t', fnr)

# False discovery rate
try:
    fdr = fp/(tp+fp)
except:
    fdr = 0
print('False discovery rate\t\t\t', fdr)

# Accuracy
try:
    acc = (tp + tn) / (tp + fp + fn + tn)
except:
    acc = 0
print('Accuracy\t\t\t\t', acc)

print('=' * 80)
title = 'Confusion Matrix'
cm_array = [[tp, fp], [fn, tn]]
df = pd.DataFrame(cm_array, ['Positive', 'Negative'], ['Positive', 'Negative'])
sns.heatmap(df, annot=True, annot_kws={'size': 16}, fmt='g', cbar=False, center=tn, cmap="Greys")
plt.title(title)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
fig = plt.gcf()
plt.show()


        



