import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
from robi_config import RoBiEvaluationConfig, PredictConfig
from hybrid_sentiment_network import predict_sentiment

def evaluate_robi(
    model: RobertaModel, 
    tokenizer: RobertaTokenizer,
    eval_df: pd.DataFrame,
    device: torch.device,
    eval_config: RoBiEvaluationConfig,
    predict_config: PredictConfig
) -> Tuple[np.array, np.array]:
    """
    Evaluates RoBi on an evaluation dataset by predicting the sentiments for each text entry and 
    comparing them with the true labels. The function returns arrays of predicted sentiments and
    true labels for further analysis.
    
    Args:
        model (RoBERTa/BiLSTM): The RoBi model to evaluate.
        tokenizer (RobertaTokenizer): The tokenizer for RoBi.
        eval_df (pd.DataFrame): The evaluation dataset.
        device (torch.device): The device to run the model on, either CPU or GPU if available.
        eval_config (RoBiEvaluationConfig): Configuration parameters for evaluation.
        predict_config (PredictConfig): Configuration parameters for prediction using RoBi.
        
    Returns:
        Tuple[np.array, np.array]: Predicted labels and true labels. 
    """
    # Sets the model to evaluation mode, this disables layers such as dropout
    model.eval() 
    
    # Lists to store prdicted labels and true labels
    all_preds =[]
    all_labels = []
    
    # Map sentiment labels ('Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive') to numerical indices
    sentiment_map = {label: i for i, label in enumerate(eval_config.class_names)}
    
    # Run the prediction and evaluation algoritm with a progress bar
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating RoBi"):
        sentiment = predict_sentiment(model, tokenizer, row['corpus'], device, predict_config)
        all_preds.append(sentiment)
        all_labels.append(sentiment_map[row['sentiment']])
        
    return np.array(all_preds), np.array(all_labels)

def evaluate_roberta(
    model: RobertaModel, 
    tokenizer: RobertaTokenizer, 
    eval_df: pd.DataFrame, 
    device: torch.device, 
    eval_config: RoBiEvaluationConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the standard RoBERTa large model on the evaluation dataset. This function processes
    each row in the evaluation dataframe, tokenizes the input, runs it through the RoBERTa model,
    and collects the predictions and true labels.
    
    Args:
        model (RobertaModel): The RoBERTa model to evaluate.
        tokenizer (RobertaTokenizer): The tokenizer for the model.
        eval_df (pd.DataFrame): The evaluation dataset.
        device (torch.device): The device to run the model on.
        eval_config (RoBiEvaluationConfig): Configuration for evaluation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted labels and true labels.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    sentiment_map = {label: i for i, label in enumerate(eval_config.class_names)}
    
    # This context manager disables gradient calculation, which reduces memory and speeds up computations
    with torch.no_grad():
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating RoBERTa"):
            
            # Process the corpus data by converting text into token IDs and put it on a GPU
            inputs = tokenizer(
                row['corpus'], 
                return_tensors="pt", 
                truncation=True, # truncates to maximum sequence length
                max_length=512, 
                padding="max_length" # pad smaller sequences to consistent length
            ).to(device)
            
            # The model's output is a tensor representing the hidden states of the last layer
            # By taking the mean across the sequence dimension, a single vector is obtained for the entire text
            outputs = model(**inputs).last_hidden_state.mean(dim=1) # the tokenizer is passed through RoBERTa
            
            # A linear layer is applied to the output vector to compute the logits for each class
            logits = torch.nn.Linear( # raw, unnormalized scores of the NN before being transformed into probabilities
                model.config.hidden_size, # maps the hidden size of the model to the number of classes
                eval_config.num_classes
            ).to(device)(outputs) # the linear layer is applied to the outputs tensor (the result of the tokenized input)
            
            # Obtain the predicted labels from the indices of the maximum values along dim=1 (predicted sentiment labels)
            _, preds = torch.max(logits, dim=1)
            
            all_preds.append(preds.item())
            all_labels.append(sentiment_map[row['sentiment']])
            
    # Predicted labels, true labels
    return np.array(all_preds), np.array(all_labels) 
            
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculates the standard evaluation metrics for the predicted labels.
    
    More specifically, this function computes accuracy, precision, recall, and F1 score for 
    the predictions compared to the true labels.
    
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 x (Precision x Recall) / (Precision + Recall) 
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        
    Returns:
        Dict[str, float]: The calculated metrics along with their respective labels.
    """
    # Ratio of correctly predicted instances to the total instances
    accuracy = accuracy_score(y_true, y_pred)
    
    # The metrics are calculated for each class and then averaged, weighted by the number of instances for each class
    precision, recall, f1, = precision_recall_fscore_support(
        y_true, 
        y_pred,
        average='weighted'
    )
    
    return {
        'accuarcy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Plots the confusion matrix for the predicted labels by using a heatmap to show the true
    vs predicted labels for each class.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str]): Names of the classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm, 
        annot=True,
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str]
) -> None:
    """
    Plots the ROC curve for each class. This function will compute and display the Reciever
    Operating Characteristic (ROC) curve for each sentiment label, along with the Area Under
    the Curve (AOC) score.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities for each class.
        class_names (List[str]): Names of the classes.
    """ 
    fpr = dict() # false positive ratio
    tpr = dict() # true positive ratio
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(10,8))
    for i in range(len(class_names)):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
    plt.plot([0, 1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([[0.0, 1.05]])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
def evaluate_and_visualize(
    y_pred: np.ndarray, 
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    model_name: str, 
    class_names: List[str]
) -> None:
    """
    Evaluates the model performance and visualizes the results by pipelining each function
    into a single series of outputs.

    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.
        y_pred_proba (np.ndarray): Predicted probabilities for each class.
        model_name (str): Name of the model being evaluated.
        class_names (List[str]): Names of the classes.
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"Evaluation Metrics for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curve(y_true, y_pred_proba, class_names)