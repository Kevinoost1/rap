import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from scipy.special import softmax
from collections import Counter


def main():
    analysis = NLP_Analysis('translated_df_copy_heuristic_perplexity_out100.csv')
    analyzed_df = analysis.roberta_tweet()


class NLP_Analysis:
    def __init__(self, csv):
        self.working_df = pd.read_csv(csv)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def roberta_tweet(self):
        sent_model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        sent_tokenizer = AutoTokenizer.from_pretrained(sent_model_name)
        sent_model = AutoModelForSequenceClassification.from_pretrained(sent_model_name).to(self.device)

        classes = []
        confidences = []
        count = 0
        for text in self.working_df['original_text']:
            parts = self.divide_in_parts(text)
            part_predictions = []
            part_confidences = []
            part_word_counts = []

            for part in parts:
                sent_inputs = sent_tokenizer(part, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    sent_output = sent_model(**sent_inputs)
                    sent_scores = sent_output[0][0].detach().cpu().numpy()

                sent_scores = softmax(sent_scores)
                summary = {
                    'negative': sent_scores[0],
                    'neutral': sent_scores[1],
                    'positive': sent_scores[2]
                }
                pred_class = max(summary, key=summary.get)
                confidence = summary[pred_class]

                part_predictions.append(pred_class)
                part_confidences.append((pred_class, confidence))

                part_word_counts.append(len(part.split()))

            # Only takes most common class if the nr. of parts is greater than 2
            # I set it like this, since I did not want a text that was just over the max
            # To have a second part weigh heavily on the prediction (it makes it less accurate i.m.o)
            if len(part_predictions) > 2:
                majority_class = Counter(part_predictions).most_common(1)[0][0]
            # So if it only has two parts, take the first prediction since that is the longest part
            else:
                majority_class = part_predictions[0]

            total_weighted_confidence = sum(
                conf * word_count for (cls, conf), word_count in zip(part_confidences, part_word_counts) if
                cls == majority_class)
            total_words_majority = sum(
                word_count for (cls, w), word_count in zip(part_confidences, part_word_counts) if cls == majority_class)
            weighted_average_confidence = total_weighted_confidence / total_words_majority

            classes.append(majority_class)
            confidences.append(weighted_average_confidence)
            count += 1
            print(f'{count} / {len(self.working_df)}')

        self.working_df['sent_roberta_tweet_class'] = classes
        self.working_df['sent_roberta_tweet_confidence'] = confidences

        top_model_name = "cardiffnlp/twitter-roberta-large-topic-latest"
        top_tokenizer = AutoTokenizer.from_pretrained(top_model_name)
        top_model = AutoModelForSequenceClassification.from_pretrained(top_model_name).to(self.device)

        top_label_1 = []
        top_label_2 = []
        top_confidence_1 = []
        top_confidence_2 = []
        count = 0
        for text in self.working_df['original_text']:
            top_inputs = top_tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                top_output = top_model(**top_inputs)
                top_scores = top_output.logits[0].detach().cpu().numpy()
            top_scores = softmax(top_scores)
            labels = top_model.config.id2label

            sorted_labels = sorted(
                [{"label": labels[idx], "score": float(top_scores[idx])} for idx in range(len(top_scores))],
                key=lambda x: x["score"],
                reverse=True
            )

            sorted_labels = [label for label in sorted_labels if label["score"] > 0.1]

            top_label_1.append(sorted_labels[0]["label"] if len(sorted_labels) > 0 else None)
            top_confidence_1.append(sorted_labels[0]["score"] if len(sorted_labels) > 0 else None)

            top_label_2.append(sorted_labels[1]["label"] if len(sorted_labels) > 1 else None)
            top_confidence_2.append(sorted_labels[1]["score"] if len(sorted_labels) > 1 else None)

            count += 1
            print(f'{count} / {len(self.working_df)}')

        self.working_df['top_roberta_class1'] = top_label_1
        self.working_df['top_roberta_class2'] = top_label_2
        self.working_df.to_csv('analyzed_csv.csv')
        return self.working_df

    def divide_in_parts(self, text, max_length=512):
        words = text.split()
        parts = []
        current_part = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1
            if current_length > max_length:
                parts.append(' '.join(current_part))
                current_part = [word]
                current_length = len(word) + 1
            else:
                current_part.append(word)

        if current_part:
            parts.append(' '.join(current_part))

            return parts



if __name__ == '__main__':
    main()