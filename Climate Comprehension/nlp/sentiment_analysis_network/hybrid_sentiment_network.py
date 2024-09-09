###################################################################################################################
#                                                    RoBi                                                         #
#                                Hybrid RoBERTa-BiLSTM Sentiment Analysis Model                                   #
#                                                                                                                 #
#                                            Author: Elijah Taber                                                 #
###################################################################################################################

import torch
import transformers
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from typing import List, Tuple, Dict
import random
import os
import numpy as np
from tqdm import tqdm
from robi_config import (
    SlidingWindowConfig, 
    RoBiConfig, 
    TrainConfig, 
    SaveLoadConfig, 
    PredictConfig
)

# Set random seeds for reproducibility, each package must be individually addressed to lock in randomized settings under the hood
random.seed(10) # standard python
np.random.seed(10) # numpy
transformers.set_seed(10) # transformers
torch.manual_seed(10) # torch
if torch.cuda.is_available(): # GPU
    torch.cuda.manual_seed_all(10)

class SlidingWindow(Dataset):
    """
    A custom Dataset class that implements a sliding window approach for processing long texts.
    
    This class takes a list of texts and their corresponding labels, and splits long texts into
    overlapping chunks using the RoBERTa tokenizer. This allows for processing of texts that are
    longer than the maximum sequence length that RoBERTa can handle (512 tokens). The unique case 
    here of this sliding window approach, is that the chunks reatain their respective sentiment 
    labels, allowing RoBi to process each chunk independently, then allowing the BiLSTM to to 
    capture the sequential dependencies between the chunks of a single corpus.

    Attributes:
        - texts (List[str]): List of input texts.
        - labels (List[int]): List of corresponding labels for each text.
        - tokenizer (RobertaTokenizer): RoBERTa tokenizer for encoding the texts.
        - max_length (int): Maximum length of each chunk.
        - stride (int): Number of overlapping tokens between adjacent chunks.
        - chunks (List[List[int]]): List of tokenized chunks.
        - chunk_labels (List[int]): List of labels corresponding to each chunk.
    """

    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer: RobertaTokenizer, 
        config: SlidingWindowConfig = SlidingWindowConfig()
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.stride = config.stride

        self.chunks = []
        self.chunk_labels = []

        # Process each text and split into overlapping chunks of 512 tokens, overlapping by 256 tokens
        for text, label in zip(texts, labels):
            encodings = self.tokenizer(text, 
                                       return_overflowing_tokens=True,
                                       max_length=self.max_length, 
                                       stride=self.stride)
            
            if len(encodings['input_ids']) == 1:
                # Text is shorter than or equal to max_length, no need to split as it will be padded
                self.chunks.append(encodings['input_ids'][0])
                self.chunk_labels.append(label)
            else:
                # Text is longer than max_length, split into chunks
                for chunk in encodings['input_ids']:
                    self.chunks.append(chunk)
                    self.chunk_labels.append(label)

    def __len__(self) -> int:
        """Returns the total number of chunks in the dataset."""
        return len(self.chunks)

    def __getitem__(
        self, 
        idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieves a specific chunk and its corresponding label. Each passing window through the transformer
        will retain is corresponding sentiment label.

        Args:
            - idx (int): Index of the chunk to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': Tensor of token IDs for the chunk.
                - 'attention_mask': Tensor indicating which tokens should be attended to.
                - 'labels': Tensor containing the label for the chunk.
        """
        chunk = self.chunks[idx]
        label = self.chunk_labels[idx]

        # Prepares a sequence of input ids so that it can be used by the model. This adds 
        # special tokens, truncates sequences if overflowing while taking into account 
        # the special tokens, and controls a sliding window for overflowing tokens.
        encoding = self.tokenizer.prepare_for_model(
            chunk,
            max_length=self.max_length, # RoBERTa's max sequence length
            padding="max_length", # if the sequence is shorter than max_length, pad with 0
            truncation=True # this is a fail-safe that will truncate if the sequence is too long
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RoBi(nn.Module):
    """
    RoBi (RoBERTa-BiLSTM) model for sentiment analysis. A Hybrid Model Synergy.

    This model combines a pre-trained RoBERTa model with a Bidirectional LSTM and an attention
    mechanism for improved sentiment classification. It's designed to capture both the contextual
    information from RoBERTa and the sequential dependencies from the BiLSTM.
    
    The combination of RoBERTa and BiLSTM allows the model to:
    * Leverage pre-trained knowledge from RoBERTa.
    * Capture bidirectional sequential information with BiLSTM.

    Attributes:
        - roberta (RobertaModel): Pre-trained RoBERTa model.
        - bilstm (nn.LSTM): Bidirectional LSTM layer.
        - attention (nn.Sequential): Attention mechanism.
        - fc (nn.Linear): Final fully connected layer for classification.
        - dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self, 
        config: RoBiConfig = RoBiConfig()
    ):
        super(RoBi, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.bilstm = nn.LSTM(
            input_size=config.input_size,  # same size as RoBERTa large's embedding dimensions
            hidden_size=config.hidden_size, # BiLSTM output hidden state
            num_layers=config.num_layers, # stacked BiLSTM layers, where the second layer takes the output of the first
            batch_first=config.batch_first,
            bidirectional=config.bidirectional
        )
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 1),
            nn.Tanh()
        )
        self.fc = nn.Linear(
            config.hidden_size * 2,
            config.num_classes # Very Postiive, Positive, Neutral, Negative, Very Negative
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the RoBi model. Here the input is passed through RoBERTa and the output
        is processed through BiLSTM and attention mechanism.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.

        Returns:
            torch.Tensor: The output logits for each class.

        Process:
            1. Pass input through RoBERTa to get contextual embeddings.
            2. Process RoBERTa output through BiLSTM.
            3. Apply attention mechanism to BiLSTM output.
            4. Pass the weighted sum through a final fully connected layer for classification.
        """
        # RoBERTa embeddings: pass input through RoBERTa and get the last hidden state
        roberta_output = self.roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        sequence_output = roberta_output.last_hidden_state

        # BiLSTM: process the RoBERTa output through the Bidirectional LSTM
        lstm_output, _ = self.bilstm(sequence_output)

        # Attention mechanism: compute attention weights for each time step
        attention_weights = self.attention(lstm_output).squeeze(-1)
        
        # Mask out padding tokens (where attention_mask is 0)
        attention_weights = attention_weights.masked_fill(
            attention_mask.eq(0), 
            float('-inf')
        )
        
        # Apply softmax to get normalized weights
        attention_weights = torch.softmax(
            attention_weights, 
            dim=1
        )
        
        # Compute weighted sum of LSTM outputs using attention weights
        weighted_sum = torch.bmm(
            attention_weights.unsqueeze(1), 
            lstm_output
        ).squeeze(1)

        # Final classification: apply dropout for regularization and pass through the final fully connected layer
        output = self.fc(self.dropout(weighted_sum))
        return output

def train_RoBi(
    model: nn.Module, 
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader, 
    config: TrainConfig = TrainConfig()
) -> nn.Module:
    """
    Trains RoBi on training data and evaluates on validation data, with an option for using a GPU. 
    
    RoBERTa large provides strong contextual understanding through its self-attention mechanism that allows for it 
    to capture the most important parts of the text. The Bidirectional LSTM layer allows for sequential information 
    to flow from both forward and backward directions, which enhances the model's ability to capture the broader 
    and more complex context of scienctific articles.

    Args:
        model (nn.Module): The RoBi model to train.
        train_dataloader (DataLoader): DataLoader for the training data.
        val_dataloader (DataLoader): DataLoader for the validation data.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): Device to train on (CPU or GPU).
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 2e-5.

    Returns:
        nn.Module: The trained RoBi model.

    Process:
        1. Move model to the CPU or GPU if available.
        2. Set up loss function, optimizer, and learning rate scheduler.
        3. For each epoch:
           - Train on the training data.
           - Evaluate on the validation data.
           - Save the best model based on validation loss.
        4. Return the trained model.
    """
    # Move model to the CPU or GPU if available
    model.to(config.device)
    
    # Loss function to use for training (cross-entropy loss) 
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                            lr=config.learning_rate)
    
    # Creates a learning rate scheduler to dynamically decay the larning rate
    # LSTM's benefits from an increased intial learning rate that slowly decays over training time
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=config.learning_rate, 
                                              steps_per_epoch=len(train_dataloader), 
                                              epochs=config.num_epochs)

    # Set the best validation loss to infinity and then save the lowest validation loss model 
    best_val_loss = float('inf')
    best_model = None
    epochs_without_improvement = 0
    
    # Train the model by looping over each epoch
    for epoch in range(config.num_epochs):
        
        # Set the model in training mode
        model.train()
        total_loss = 0
        
        # Loop over each batch within each epoch
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
            
            # Move the input data and labels to the GPU if available
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            # Zero gradients to prevent accumulation
            optimizer.zero_grad()
            
            # Forward pass: compute model outputs
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Update model parameters and learning rate
            optimizer.step()
            scheduler.step()

            # Accumulate the loss for the current batch
            total_loss += loss.item()

        # Compute the average loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        
        # Set the model in evaluation mode
        model.eval()
        val_loss = 0
        
        # Disable gradient calculation during evaluation
        with torch.no_grad():
            
            # Loop over each batch in the validation data
            for batch in val_dataloader:
                
                # Move the input data and labels to the GPU if available
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)

                # Forward pass: compute model outputs and accumulate loss
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Compute the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check if the current validation loss is the best seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss # update
            best_model = model.state_dict() # save model
            epochs_without_improvement = 0 # reset
            torch.save(best_model, 'best_model.pth') # save to path
        else:
            epochs_without_improvement += 1 # for epochs without improvement

        # Check if early stopping is triggered
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Load the true best model, no matter if it was the latest epoch or not
    model.load_state_dict(best_model)
    return model

def save_RoBi(
    model: nn.Module, 
    tokenizer: RobertaTokenizer, 
    config: SaveLoadConfig = SaveLoadConfig()
):
    """
    Saves the RoBi model and its tokenizer to a directory.

    Args:
        model (nn.Module): The RoBi model to save.
        tokenizer (RobertaTokenizer): The tokenizer used with the model.
        path (str): The directory path to save the model and tokenizer to.
    """
    # Create the directory if it doesn't exist
    os.makedirs(config.path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config.path, 'robi_model.pt'))
    tokenizer.save_pretrained(config.path)

def load_RoBi(
    config: SaveLoadConfig = SaveLoadConfig(),
    robi_config: RoBiConfig = RoBiConfig()
) -> Tuple[nn.Module, RobertaTokenizer]:
    """
    Loads a saved RoBi model and its tokenizer from a directory.

    Args:
        path (str): The directory path to load the model and tokenizer from.
        num_classes (int, optional): Number of classes for the model. Defaults to 5.

    Returns:
        Tuple[nn.Module, RobertaTokenizer]: The loaded RoBi model and its tokenizer.
    """
    model = RoBi(config=robi_config)
    model.load_state_dict(torch.load(os.path.join(config.path, 'robi_model.pt')))
    tokenizer = RobertaTokenizer.from_pretrained(config.path)
    
    return model, tokenizer

def predict_sentiment(
    model: nn.Module, 
    tokenizer: RobertaTokenizer, 
    text: str, 
    device: torch.device, 
    config: PredictConfig = PredictConfig()
) -> int:
    """
    Predicts the sentiment of a given text using the RoBi model.

    This function handles long texts by splitting them into overlapping chunks,
    processing each chunk separately, and then averaging the results.

    Args:
        model (nn.Module): The trained RoBi model.
        tokenizer (RobertaTokenizer): The tokenizer used with the model.
        text (str): Unseen input text to analyze.
        device (torch.device): The device to run the model on.
        max_length (int, optional): Maximum length of each chunk. Defaults to 512.
        stride (int, optional): Number of overlapping tokens between chunks. Defaults to 256.

    Returns:
        int: The predicted sentiment class (0-4).

    Process:
        1. Tokenize the input text into overlapping chunks.
        2. Process each chunk through the model.
        3. Average the outputs from all chunks.
        4. Return the class with the highest average score.
    """
    model.eval()
    encodings = tokenizer(
        text, 
        return_overflowing_tokens=config.overflow_tokens, 
        max_length=config.max_length, 
        stride=config.stride
    )
    
    chunk_outputs = []
    for chunk in encodings['input_ids']:
        encoding = tokenizer.prepare_for_model(
            chunk,
            max_length=config.max_length,
            padding="max_length",
            truncation=True
        )
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            chunk_outputs.append(outputs)

    final_output = torch.mean(torch.cat(chunk_outputs, dim=0), dim=0)
    _, preds = torch.max(final_output, dim=0)

    return preds.item()