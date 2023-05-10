import torch
from sklearn.metrics import hamming_loss, f1_score

def compute_average_f1_score(predicted, truth, num_labels):
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(truth, torch.Tensor)

    if num_labels > 1:
        weighted_avg_f1 = f1_score(truth, predicted, average='weighted')
        unweighted_avg_f1 = f1_score(truth, predicted, average='macro')
        all_f1 = f1_score(truth, predicted, average=None)
        return weighted_avg_f1, unweighted_avg_f1, all_f1
    else:
        avg_f1 = f1_score(truth, predicted, average='binary')
        all_f1 = f1_score(truth, predicted, average=None)
        return avg_f1, all_f1

def label_correctness(predictions, truths, num_labels=1, threshold=0.5):
    #counts up hamming distance and true accuracy
    additional_scores = {}
    if len(predictions.size()) == 1:
        predictions = torch.sigmoid(predictions) > threshold
    else:
        assert len(predictions.size()) == 2
        predictions = torch.max(predictions, dim=-1)[1]

    additional_scores['hamming_accuracy'] = 1 - hamming_loss(truths.squeeze().cpu(), predictions.squeeze().cpu())
    if num_labels > 1:
        w_avg_f1, additional_scores['unweighted_f1'], additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores
    else:
        w_avg_f1, additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores


def lossfxn(out, labels, device):
    eps = 1e-6 # I believe this is the same as the japan paper (see writeup for a reference)
    augmented_out = [out[l][1] if labels[l] == 1 else (torch.ones(1).to(device) - out[l][0] if labels[l] == 2 else torch.ones(1).to(device) - (torch.ones(1).to(device) - out[l][1])*out[l][0]) for l in range(len(out))]
    return torch.tensor(-1).to(device) * torch.mean(torch.log(torch.hstack(augmented_out) + eps)) 

    # If change loss function here, make the corresponding change to lossfxn_t in new_main.py

def value_correctness(predictions, truths, device, num_labels=1, threshold=0.5): #Fixme. Num_labels and threshold not important here.
    with torch.no_grad():
        return lossfxn(predictions, truths, device), None
