from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class NaiveBayesModel():
    def model(self, X_train, y_train):
        tf = TfidfVectorizer()
        x = tf.fit_transform(X_train)
        print(x.shape)
        le = LabelEncoder()
        y = le.fit_transform(y_train)
        print(y.shape)
        self.model = MultinomialNB().fit(x, y)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        # Calculating metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # Printing results
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")


class FastTextModel():
    def __init__(self, embedding_dim, num_lang, vocab_size, check_point_path):
        """
        Initalize parameters for the model
        
        Args:
        max_sequence_len (str): The max sequence length of each sentence
        n_grams (int): The number of n_grams to be considered
        embedding_dim (int): the dimension of the embedding dimension
        num_lang: number of languages to classify
        vocab_size: the number of vocabulary size in the corpus
        check_point_path: the path to save model checkpoint file
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),                   # Average embeddings (like FastText)
            tf.keras.layers.Dense(64, activation='relu'),               # hidden layer
            tf.keras.layers.Dense(num_lang, activation='softmax')              # Output layer
        ])
        model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
        self.model = model
        self.check_point_path = check_point_path

    def train(self, X_train, y_train, epoch=20, batch_size=500):
        # Train the model, save model checkpoints
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.check_point_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        history = self.model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[cp_callback])
        # Plot the training loss curve
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['accuracy'], label='Training accuracy', linestyle='--')
        plt.title('Training Loss and Accuray Curve')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    def predict(self, X_test):
        if os.path.exists(self.check_point_path):
            self.model.build(X_test.shape)
            self.model.load_weights(self.check_point_path)
        print(self.model.predict(X_test))
    
    def evaluate(self, X_test, y_test, batch_size=128):
        if os.path.exists(self.check_point_path):
            self.model.build(X_test.shape)
            self.model.load_weights(self.check_point_path)
        loss, accuracy = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        print(f'Test Accuracy: {accuracy * 100:.2f}%')