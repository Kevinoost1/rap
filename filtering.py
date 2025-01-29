import pandas as pd
import torch
from sklearn.decomposition import PCA
from tinycss2 import tokenizer
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

"""
This class: TextFilters, filters messy text (text from which meaning is possibly hard to extract).
It does this for each transcript in the dataframe such that
the end result will be a dataframe of transcripts that contain the least amount of confusing transcripts.
One example of messy text in the current dataframe is:

eng_7283110983239847200.txt: " Run, run, run, run! I’m all set. Oh, oh, oh, oh! One "

Currently, two main methods are used for identifying messy transcripts and filtering them:

- A heuristic approach -> It selects transcripts with a low word count.
                            And transcripts that have little unique words (word diversity)
- A perplexity approach -> It calculates the perplexity of all transcripts (how 'confused' the model is by the word, given all the other words)

In total the heuristic approach takes two manual parameters that can be tuned:
- The word count (maximum words allowed). By default, this is set at 10
- The word diversity (ratio unique words vs. total words). By default, this is set at 0.3

The perplexity approach takes one manual parameter
- The perplexity threshold (maximum perplexity allowed for a transcript). By default, this is set at 100
"""


def main():
    filtered_csv = TextFilter('translated_df_copy.csv',
                              apply_heuristic_filter=False,
                              apply_perplexity_filter=False,
                              filter_out_perplexity=False
                              )


class TextFilter:

    def __init__(self,
                 input_csv,
                 max_words_allowed: int = 10,
                 max_diversity_allowed: float = 0.3,
                 perplexity_threshold: float = 100,

                 model_for_perplexity_calculation: str = 'gpt2-xl',

                 # These are variables that are used if you want to run each function separately
                 apply_heuristic_filter: bool = False,
                 apply_perplexity_filter: bool = False,
                 filter_out_perplexity: bool = False,
                 ):

        self.input_csv_name: str = input_csv
        self.working_df: pd.DataFrame = pd.read_csv(input_csv)
        self.max_words_allowed: int = max_words_allowed
        self.max_diversity_allowed: float = max_diversity_allowed
        self.perplexity_threshold: float = perplexity_threshold

        self.model_for_perplexity_calculation = model_for_perplexity_calculation

        self.apply_heuristic_filter: bool = apply_heuristic_filter
        self.apply_perplexity_filter: bool = apply_perplexity_filter
        self.filter_out_perplexity: bool = filter_out_perplexity

        # All these functions allow for user specification of which function to execute.
        # These are for testing and personalization. These do not have to be looked at.
        if self.apply_heuristic_filter:
            self.working_df = self.apply_heuristics()

            self.working_df.to_csv(f'{self.input_csv_name[:-4]}_heuristic.csv', index=False)
            self.input_csv_name = f'{self.input_csv_name[:-4]}_heuristic.csv'
            print(f'heuristics successfully applied to {self.input_csv_name[:-4]}_heuristic.csv')

        if self.apply_perplexity_filter:
            self.working_df = self.apply_heuristics()

            self.working_df.to_csv(f'{self.input_csv_name[:-4]}_perplexity.csv', index=False)
            print(f'perplexity successfully calculated and added to {self.input_csv_name[:-4]}_perplexity.csv')

        if self.filter_out_perplexity:
            self.working_df = self.delete_perplexity_threshold()

            self.working_df.to_csv(f'{self.input_csv_name[:-4]}_perplexity_out{perplexity_threshold}.csv')
            print(
                f'perplexity over {perplexity_threshold} was correctly filter out in {self.input_csv_name[:-4]}_perplexity_out{perplexity_threshold}.csv')

    # The function for the heuristics approach of filtering
    def apply_heuristics(self):
        word_diversity_dict = {}
        word_count_dict = {}

        for i, text in enumerate(self.working_df['original_text']):
            text = text.lower()
            words = text.split()

            num_words = len(words)
            unique_words = set(words)
            word_diversity = len(unique_words) / num_words

            word_diversity_dict[i] = word_diversity
            word_count_dict[i] = num_words

        index_low_word = []
        for max_words_key, max_words_items in word_count_dict.items():
            if max_words_items < self.max_words_allowed:
                index_low_word.append(max_words_key)

        index_low_div = []
        for min_div_key, min_div_item in word_diversity_dict.items():
            if min_div_item < self.max_diversity_allowed:
                index_low_div.append(min_div_key)

        indices_to_drop_from_df = set(index_low_word).union(index_low_div)
        no_low_df = self.working_df.drop(self.working_df.index[list(indices_to_drop_from_df)])
        no_low_df = no_low_df.reset_index(drop=True)
        deleted = self.working_df.iloc[list(indices_to_drop_from_df)]
        deleted.to_csv('heuristic_applied.csv')

        return no_low_df

    # The function for the perplexity approach to filtering
    def apply_perplexity_filter(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.model_for_perplexity_calculation} was loaded onto {device}')

        tokenizer_gpt = AutoTokenizer.from_pretrained(self.model_for_perplexity_calculation)
        model_gpt = AutoModelForCausalLM.from_pretrained(self.model_for_perplexity_calculation).to(device)

        perplexity_scores = []
        count = 0
        for text in self.working_df['original_text']:
            inputs = tokenizer_gpt(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                outputs = model_gpt(**inputs, labels=inputs['input_ids'])

            loss = outputs.loss.item()

            perplexity_score = torch.exp(torch.tensor(loss)).item()
            perplexity_scores.append(perplexity_score)
            count += 1
            print(f'{count} / {len(self.working_df)}')

            torch.cuda.empty_cache()

        self.working_df['perplexity'] = perplexity_scores
        return self.working_df

    # Function to filter out transcripts lower than perplexity threshold
    def delete_perplexity_threshold(self):
        self.working_df = self.working_df[self.working_df['perplexity'] < self.perplexity_threshold]
        return self.working_df


if __name__ == '__main__':
    main()
