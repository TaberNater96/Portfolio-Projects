import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from typing import List, Tuple
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc
)

class C4NNEvaluator:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate the model on test data and return loss and accuracy.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): True labels for test data
        
        Returns:
            Tuple[float, float]: Test loss and accuracy
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return loss, accuracy
    
    def get_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get model predictions for test data.
        
        Args:
            X_test (np.ndarray): Test features
        
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        return self.model.predict(X_test)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): List of class names
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, class_names: List[str]):
        """
        Plot ROC curve for multi-class classification.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities for each class
            class_names (List[str]): List of class names
        """
        n_classes = len(class_names)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        for i, color in zip(range(n_classes), ['blue', 'red', 'green']):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
        """
        Plot precision, recall, and F1-score for each class.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): List of class names
        """
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        metrics = ['precision', 'recall', 'f1-score']
        values = {metric: [report[class_name][metric] for class_name in class_names] for metric in metrics}
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width, values['precision'], width, label='Precision')
        rects2 = ax.bar(x, values['recall'], width, label='Recall')
        rects3 = ax.bar(x + width, values['f1-score'], width, label='F1-score')
        
        ax.set_ylabel('Scores')
        ax.set_title('Precision, Recall, and F1-score by class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend()
        
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        
        fig.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy and loss.
        
        Args:
            history: Keras history object containing training information
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.show()