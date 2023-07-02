import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from figures import plot_agree_probabilities
from utils import get_bipartite_adjacency_matrix_kcore, get_expected_agreements

from main import positions, client_uuid_to_ftm_industry


###############################
# Figure 4: probabilities of agreement and disagreement
# with pro-environmental policy groups
###############################


# Check whether we have already calculated the expected probabilities of agreement and disagreement
if os.path.exists('data/agreement_probabilities/agree.csv'):
    agree_probabilities = pd.read_csv('data/agreement_probabilities/agree.csv', index_col=[0, 1])
    disagree_probabilities = pd.read_csv('data/agreement_probabilities/disagree.csv', index_col=[0, 1])

else:
    agree_probabilities = {}
    disagree_probabilities = {}

    # We compute the expected probability of agreement and disagreement with pro-environmental
    # policy groups for each state and record type. We do this by computing the expected
    # number of (dis)agreements that a random position from pro-environmental policy groups
    # would have from members of a given industry_id, and then dividing by the total number of
    # members of that industry_id. We then average over all pro-environmental policy groups.
    # We keep track of the record type (lobbying or testimony) because the figure includes these in the axis labels
    for (state, record_type), _ in tqdm(positions.groupby(['state', 'record_type'])):
        adj_matrix = get_bipartite_adjacency_matrix_kcore(positions[positions.state == state], (1,1))
        disagree_probabilities[state, record_type]: pd.Series = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, 'oppose')
        agree_probabilities[state, record_type]: pd.Series = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, 'support')

    agree_probabilities = pd.DataFrame(agree_probabilities).replace(np.nan, 0)
    disagree_probabilities = pd.DataFrame(disagree_probabilities).replace(np.nan, 0)

    agree_probabilities = agree_probabilities.loc[:, agree_probabilities.columns.sortlevel(1)[0].values]
    disagree_probabilities = disagree_probabilities.loc[:, disagree_probabilities.columns.sortlevel(1)[0].values]

    # save the agreement probabilities dataframes to a directory
    # if the directory does not exist, create it
    if not os.path.exists('data/agreement_probabilities'):
        os.makedirs('data/agreement_probabilities')

    agree_probabilities.to_csv('data/agreement_probabilities/agree.csv')
    disagree_probabilities.to_csv('data/agreement_probabilities/disagree.csv')

plot_agree_probabilities(agree_probabilities, disagree_probabilities)