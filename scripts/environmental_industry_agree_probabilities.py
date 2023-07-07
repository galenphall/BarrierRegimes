import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from figures import plot_agree_probabilities
from utils import get_bipartite_adjacency_matrix_kcore, get_expected_agreements

from main import positions, client_uuid_to_ftm_industry


def main(recompute=False):
    # Check whether we have already calculated the expected probabilities of agreement and disagreement
    if os.path.exists('data/agreement_probabilities/agree.parquet') and not recompute:
        print("Loading agreement probabilities from disk")
        agree_probabilities = pd.read_parquet('data/agreement_probabilities/agree.parquet')
        disagree_probabilities = pd.read_parquet('data/agreement_probabilities/disagree.parquet')

    else:
        print("Calculating agreement probabilities")
        agree_probabilities = {}
        disagree_probabilities = {}

        # We compute the expected probability of agreement and disagreement with pro-environmental
        # policy groups for each state and record type. See the get_expected_agreements function
        # for more details.
        for (state, record_type), _ in tqdm(positions.groupby(['state', 'record_type'])):
            adj_matrix = get_bipartite_adjacency_matrix_kcore(positions[positions.state == state], (1,1))
            disagree_probabilities[state, record_type]: pd.Series = get_expected_agreements(
                adj_matrix,
                client_uuid_to_ftm_industry,
                'oppose')

            agree_probabilities[state, record_type]: pd.Series = get_expected_agreements(
                adj_matrix,
                client_uuid_to_ftm_industry,
                'support')

        agree_probabilities = pd.DataFrame(agree_probabilities).replace(np.nan, 0)
        disagree_probabilities = pd.DataFrame(disagree_probabilities).replace(np.nan, 0)

        agree_probabilities = agree_probabilities.loc[:, agree_probabilities.columns.sortlevel(1)[0].values]
        disagree_probabilities = disagree_probabilities.loc[:, disagree_probabilities.columns.sortlevel(1)[0].values]

        # save the agreement probabilities dataframes to a directory
        # if the directory does not exist, create it
        if not os.path.exists('data/agreement_probabilities'):
            os.makedirs('data/agreement_probabilities')

        agree_probabilities.to_parquet('data/agreement_probabilities/agree.parquet')
        disagree_probabilities.to_parquet('data/agreement_probabilities/disagree.parquet')

    plot_agree_probabilities(agree_probabilities, disagree_probabilities)