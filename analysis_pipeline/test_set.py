import pandas as pd
import numpy as np

"""
This script contains code to generate a test set from random columns in the dataset
"""


def main():
    filtered_dataframe = pd.read_csv('translated_df_copy_heuristic_perplexity_out100.csv')
    test_set = create_test_set(filtered_dataframe)
    print(test_set)
    test_set.to_csv('test_sets/filtered_test_set.csv', index=False)

def create_test_set(dataframe):
    random_indexes = []
    for n in range(1, 301):
        random_row = np.random.randint(1, 5159)
        random_indexes.append(random_row)

    test_set = dataframe.iloc[random_indexes]
    return test_set

# Proposed categories for labelling:

# Politics & Social Issues (politics)
# political campaign
# climate change
# beleid
# polarizing topics (manosphere?)


# Conspiracy theories & Misinformation
# Health & Fitness (health)
# Personal vlogs & Daily life (vlog)
# Relationships (relationships)
# Fashion & Beauty (beauty)
# Food & Dining (food)
# Art & Culture (art)
# Self-Improvement (self)
# Meme
# Misc.



if __name__ == '__main__':
    main()