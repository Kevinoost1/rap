import pandas as pd
import torch
from sklearn.decomposition import PCA
from tinycss2 import tokenizer
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples, homogeneity_completeness_v_measure
from sklearn.mixture import GaussianMixture
import shutil
import os

def main():
    new_csv = Filtering('translated_df_copy.csv', do_heuristic_filter=False,
                        do_perplexity_filter=False, filter_out_perplexity=False)

    # perp_name = 'translated_df_copy_heuristic_perplexity'
    # perp_csv = pd.read_csv('translated_df_copy_heuristic_perplexity.csv')
    # filter_perp = perp_csv[perp_csv['perplexity'] < 100]
    # filter_perp.to_csv(f'{perp_name}_out100.csv')

# Again creation of the filtering within a class to allow for a possible automatic pipeline
class Filtering:
    def __init__(self, translated_csv, do_heuristic_filter=False, max_words_heuristic=10,
                 max_diversity_heuristic=0.3, do_perplexity_filter=False, model='gpt2-xl'
                 , filter_out_perplexity=False, perplexity_threshold=100):
        # Initialization of global variables
        self.working_csv = pd.read_csv(translated_csv)
        self.translated_csv = translated_csv
        self.do_heuristic_filter = do_heuristic_filter
        self.max_words_heuristic = max_words_heuristic
        self.max_diversity_heuristic = max_diversity_heuristic
        self.do_perplexity_filter = do_perplexity_filter
        self.model = model
        self.filter_out_perplexity = filter_out_perplexity
        self.perplexity_threshold = perplexity_threshold

        # All these functions allow for user specification of which function to execute
        if self.do_heuristic_filter:
            self.working_csv = self.heuristics(self.working_csv)
            self.working_csv.to_csv(f'{self.translated_csv[:-4]}_heuristic.csv', index=False)
            self.translated_csv = f'{self.translated_csv[:-4]}_heuristic.csv'
            print(f'heuristics successfully applied to {self.translated_csv[:-4]}_heuristic.csv')

        if self.do_perplexity_filter:
            self.working_csv = self.perplexity_filter(self.working_csv)
            self.working_csv.to_csv(f'{self.translated_csv[:-4]}_perplexity.csv')
            print(f'perplexity successfully calculated and added to {self.translated_csv[:-4]}_perplexity.csv')

        if self.filter_out_perplexity:
            self.working_csv = self.filter_away_perplexity(self.working_csv)
            self.working_csv.to_csv(f'{self.translated_csv[:-4]}_perplexity_out{perplexity_threshold}.csv')
            print(f'perplexity over {perplexity_threshold} was correctly filter out in {self.translated_csv[:-4]}_perplexity_out{perplexity_threshold}.csv')

    # The heuristics filter function
    def heuristics(self, translated_df):
        # Makes two empty dicts for the two heuristics
        word_diversity_dict = {}
        word_count_dict = {}
        # Loops through the transcripts and also retrieves the index
        for i, text in enumerate(translated_df['original_text']):
            text = text.lower()
            words = text.split()
            # Calculates the number of words and unique words
            num_words = len(words)
            unique_words = set(words)
            word_diversity = len(unique_words) / num_words
            # Adds the word count and diversity to the dicts along with the index in the df
            word_diversity_dict[i] = word_diversity
            word_count_dict[i] = num_words
        # Loop to check if a line has a low word count
        # If it does, retrieve the index and list it
        index_low_word = []
        for key, items in word_count_dict.items():
            if items < self.max_words_heuristic:
                low_word_item = translated_df['original_text'][key]
                low_word_item_i = key
                index_low_word.append(low_word_item_i)

        # Does the same thing for transcripts with low diversity
        index_low_div = []
        for div_key, div_item in word_diversity_dict.items():
            if div_item < self.max_diversity_heuristic:
                low_div_item = translated_df['original_text'][div_key]
                low_div_item_i = div_key
                index_low_div.append(low_div_item_i)

        # Takes both index lists and drops rows, based on union of the two index lists
        indices_to_drop = set(index_low_word).union(index_low_div)
        no_low_df = translated_df.drop(translated_df.index[list(indices_to_drop)])
        no_low_df = no_low_df.reset_index(drop=True)
        deleted = translated_df.iloc[list(indices_to_drop)]
        deleted.to_csv('heuristic_deleted.csv')

        return no_low_df

    # Function for calculating perplexity
    def perplexity_filter(self, translated_df):
        # Initialize model and tokenizer
        model_name = self.model
        tokenizer_gpt = AutoTokenizer.from_pretrained(model_name)

        # Model is again run on GPU if availlable
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'{self.model} was loaded onto {device}')
        model_gpt = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        perplexity_scores = []
        count = 0
        # Loops through all the transcripts
        for text in translated_df['original_text']:
            # Generates inputs for the model
            inputs = tokenizer_gpt(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

            # With torch no grad to save computation (we don't need to calculate gradients)
            with torch.no_grad():
                # Generates the outputs
                outputs = model_gpt(**inputs, labels=inputs['input_ids'])

            # Retrieves the loss from the model
            loss = outputs.loss.item()
            # Perplexity is the exponent of the loss
            perplexity_score = torch.exp(torch.tensor(loss)).item()
            perplexity_scores.append(perplexity_score)
            count += 1
            print(f'{count} / {len(translated_df)}')
            # Empties GPU memory after each transcript
            torch.cuda.empty_cache()

        # Adds perplexity to the dataframe
        translated_df['perplexity'] = perplexity_scores
        return translated_df

    # Filters out rows with perplexity lower than threshold
    def filter_away_perplexity(self, translated_df):
        translated_df = translated_df[translated_df['perplexity'] < self.perplexity_threshold]
        return translated_df



if __name__ == '__main__':
    main()