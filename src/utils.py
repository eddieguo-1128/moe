import evaluate
import numpy as np

bleu = evaluate.load("bleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics_factory(tokenizer):
    def compute_metrics(eval_preds):

        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds[0], skip_special_tokens=True)

        # replace -100 with pad_token_id in labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["bleu"]}
    
    return compute_metrics
