# language-detection
This repository provides a command-line tool for training, evaluating, and using a language detection model. The model utilizes a FastText-based architecture for text classification based on character-level n-grams. It also leverages TensorFlow for model building, training, and evaluation, and scikit-learn's OneHotEncoder for label encoding.

The dataset used in this model is wiki40b, loaded from hugging face. After running, it will create a `/dataset` folder under this repository, splitting data into train, validation and test set. A label set will also be generated correspondingly denoting the language of the text.

## Requirements
Before running the code, ensure that the following Python libraries are installed:

- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`

You can install the necessary dependencies using pip:

```bash
pip install tensorflow pandas numpy scikit-learn
```

## File Overview
`model.py`: Contains the definition of the FastTextModel, which is used to build and train the language detection model.
`data.py`: Includes a function to load and preprocess the dataset used for training, validation, and testing.
`main.py`: Main script that orchestrates loading data, processing, training, evaluating, and predicting with the language detection model.


## Usage

You can use the command-line interface to perform different operations with the language detection model.

### 1. **Training the Model**

To train the model on the dataset, run the following command:

```bash
python main.py --train
```
This will train the model using the training data for 20 epochs and a batch size of 500. You can adjust the model parameters in the code if needed. Model checkpoint will be save at `/checkpoints` directory

### 2. **Evaluating the Model**

To evaluate the trained model on the test dataset and obtain the accuracy, run:

```bash
python main.py --evaluate
```
This will run the model on the test data and display the evaluation results, including the accuracy.

### 3. Predicting the Language of Input Text
To predict the language of a single input text, run:

```bash
python main.py --predict "This is an example sentence."
```

## Additional Information
If you want to customize training settings like the number of epochs or batch size, you can modify the script directly before running it.
For each of the commands above, you can pass additional arguments or flags if needed.



