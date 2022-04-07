"""
File containing all model components and functions. 
It defines the architecture class for RoBERTa model
and creates necessery components like RoBERTa tokenizer 
"""

from numpy import array
from transformers import RobertaModel, RobertaTokenizer
import torch.nn as nn
import torch
from transformers import logging
logging.set_verbosity_error()


class RobertaForFakeNewsDetection(nn.Module):
  """
  RoBERTa Model for classyfing claims to true/false.
  """

  def __init__(self):
    """
    @param    classifier: a torch.nn.Module classifier
    @param    freeze_bert (bool): Set 'False' to fine-tune the BERT model
    """
    super(RobertaForFakeNewsDetection, self).__init__()
    # Instantiate BERT model
    self.roberta = RobertaModel.from_pretrained('roberta-base')
    # Our classifier (feed forward neural network)
    self.classifier = nn.Sequential(
        nn.Dropout(0.2),              # Dropout regulatization method (to avoid overfitting)
        nn.Linear(768, 50),           # Hidden Layer
        nn.ReLU(),                    # Activation Function
        nn.Linear(50, 2),             # Output Layer
        nn.LogSoftmax(dim=1)          # Activation Function in the output 
    )


  def forward(self, input_ids, attention_mask):
    """
    Feed input to BERT and the classifier to compute logits.
    @param    input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
    @param    attention_mask (torch.Tensor): a tensor that hold attention mask information with shape (batch_size, max_length)
    @return   logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
    """

    # Feed input to bert and extract last hidden state of the [CLS] token, that is used for classificaiton task.
    _, last_cls_token = self.roberta(input_ids = input_ids, attention_mask = attention_mask, return_dict = False)

    # Feed input to classifier to compute logits ()
    logits = self.classifier(last_cls_token)

    return logits


def data_tokenization(claims: list, max_claim_length = 80):
    """
    Tokenize data with RoBERTa tokenizer
    """
    data_tokens = tokenizer.__call__(
        claims,                           # Sentence to encode.
        add_special_tokens = True,        # Add '[CLS]' and '[SEP]'.
        padding = True,                   # Pad sentences.
        truncation = True,                # Truncate sentences.
        max_length = max_claim_length,    # Specify max length   
        return_attention_mask = True,     # Construct attn. masks.
    )

    # Convert the lists into tensors.
    input_ids = torch.tensor(data_tokens['input_ids'])
    attention_masks = torch.tensor(data_tokens['attention_mask']) # differentiates padding from non-padding

    return input_ids, attention_masks


def model_predict(claim: str):
    """
    Predict a single claim with probability
    """
    input_id, attention_mask = data_tokenization([claim], 80)

    classifier.eval()
    logits = classifier(input_id, attention_mask)

    # changing log softmax output to probabilitiesv
    prediction_prob = torch.exp(logits).detach().cpu().numpy()

    if prediction_prob[0][0] > prediction_prob[0][1]:
        prediction = False
    else: 
        prediction = True
    
    return prediction, max(prediction_prob[0])


def load_roberta_model():
    """
    Initialise and load RoBERTa model configuration from the file.
    """
    global classifier
    classifier = RobertaForFakeNewsDetection()
    classifier.load_state_dict(torch.load('./model/roberta-base.pt', map_location=torch.device('cpu')))

def get_classifier():
    """
    Get the classifier
    """
    return classifier

def load_tokenizer():
    """
    Cretae RoBERTa tokenizer
    """
    global tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def load_model_components():
    """
    Load RoBERTa model and its tokenizer
    """
    load_roberta_model()
    load_tokenizer()