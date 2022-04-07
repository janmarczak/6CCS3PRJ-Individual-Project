"""
File containing functions for generating and formatting LIME (XAI) 
explanations after the predictions is made
"""

import lime
from lime.lime_text import LimeTextExplainer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import numpy as np
from model import data_tokenization, get_classifier
import torch


def lime_predictor(claims: np.array):
    """
    Model prediction function that takes a numpy array and ouputs prediction probabilities.
    Used in the process of creating explanations for claims with LIME
    """
    input_ids, attention_masks = data_tokenization(claims, 80)
    batch_size = 20
    device = 'cpu'
        
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    classifier = get_classifier()
    classifier.eval()

    probabilities = []
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_attn_mask = batch

        with torch.no_grad():
            logits = classifier(b_input_ids, b_attn_mask)

        pred_probabilities = torch.exp(logits).detach().cpu().data.numpy()
        probabilities.append(pred_probabilities)
        
    return np.vstack(probabilities) # array of probabilities for each sample


def format_lime_html(lime_str: str):
    """
    Format HTML file generated from LIME explanation
    """
    # Edit css files for HTML components
    lime_str = lime_str.replace(r".lime.top_div {\n  display: flex;\n  flex-wrap: wrap;", 
                    r".lime.top_div {\n  display: flex;\n  flex-direction: row;\n  flex-wrap: wrap;\n  justify-content: space-evenly;\n  height:  auto;")
    lime_str = lime_str.replace(r".lime.predict_proba {\n  width: 245px;", 
                    r".lime.predict_proba {\n  width: auto;\n  display: inline-block;\n  color:  rgb(250, 31, 31);")
    lime_str = lime_str.replace(r".lime.explanation {\n  width: 350px;",
                    r".lime.explanation {\n  width: auto;\n  display: inline-block;")
    lime_str = lime_str.replace(r".lime.text_div {\n  max-height:300px;\n  flex: 1 0 300px;\n  overflow:scroll;",
                    r".lime.text_div {\n  width: auto;\n  display: inline-block;")

    # Changing the colors of fake/true predictions to red & green
    lime_str = lime_str.replace("colors = [this.colors_i(0), this.colors_i(1)];", "colors = [this.colors_i(3), this.colors_i(2)];")
    lime_str = lime_str.replace(r", 1, exp_div);", r", 2, exp_div);")   
    lime_str = lime_str.replace("var color = this.colors(names[i]);", "var color = this.colors(names[data.length + 1 - i]);")

    return lime_str


def lime_explanation(claim: str):
    """
    Run process of explaining claim's predictions using LIME
    """
    explainer = LimeTextExplainer(class_names=['fake', 'true'])
    exp = explainer.explain_instance(claim, lime_predictor, num_features=10, num_samples=1000)
    exp_str = exp.as_html()

    return format_lime_html(exp_str)
