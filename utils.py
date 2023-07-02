import networkx as nx
import numpy as np
import pandas as pd
import sparse
from matplotlib import pyplot as plt


def get_bipartite_adjacency_matrix_kcore(positions: pd.DataFrame, k_core: tuple=(5, 2)):
    """
    Construct the adjacency matrix adj_matrix from the positions dataframe; adj_matrix is a pandas dataframe
    :param positions: [pd.DataFrame] the positions dataframe
    :param k_core: [tuple] the minimum number of clients and bills for the k-core
    :return adj_matrix: [pd.DataFrame] the adjacency matrix
    """

    # Keep only rows with a client_uuid and a bill_identifier and a position_numeric in [-1, 1]
    selection = positions[positions.client_uuid.notnull()].copy()
    selection = selection[selection.bill_identifier.notnull()].copy()
    selection = selection[selection.position_numeric.isin([-1, 1])].copy()

    # Calculate number of positions per client and bill
    n_client_positions = selection.client_uuid.value_counts()
    n_bill_positions = selection.bill_identifier.value_counts()

    # Keep only clients and bills with at least k_core[0] and k_core[1] positions, respectively
    selection = selection[selection.client_uuid.map(n_client_positions) >= k_core[0]]
    selection = selection[selection.bill_identifier.map(n_bill_positions) >= k_core[1]]

    # Calculate the adjacency matrix
    adj_matrix = selection.groupby(['client_uuid', 'bill_identifier']).position_numeric.sum().unstack()

    # Because clients can record multiple positions on the same bill, some values in adj_matrix can be > 1 or < -1
    # We set these values to 1 or -1, respectively
    adj_matrix = np.sign(adj_matrix)

    # Delete the selection of the positions dataframe to save memory
    del selection

    # Create the k-core of the adjacency matrix by iteratively removing nodes with degree < k_core[0]
    # and nodes with degree < k_core[1], for clients and bills, respectively, until no such nodes remain
    while any(abs(adj_matrix).sum(1) < k_core[0]) or any(abs(adj_matrix).sum(0) < k_core[1]):
        adj_matrix = adj_matrix.loc[abs(adj_matrix).sum(1) >= k_core[0], :]
        adj_matrix = adj_matrix.loc[:, abs(adj_matrix).sum(0) >= k_core[1]]

    G = nx.from_edgelist(adj_matrix.stack().dropna().reset_index().values[:, :2].tolist())
    components = sorted(nx.connected_components(G), key=len)

    # If there are no connected components, return None
    if len(components) == 0:
        return None

    # Remove any connected components that are not connected to the largest connected component
    c = components[-1]
    adj_matrix = adj_matrix.reindex(index=sorted(c & {*adj_matrix.index.values}),
                                    columns=sorted(c & {*adj_matrix.columns.values}))

    return adj_matrix


def adjust_label(label):
    """
    Adjust the label for an industry_id so that it is more readable
    :param label: an FTM industry_id name
    :return:
    """

    # Assert that the label is a string in the data/fmt_indstries.txt file
    assert label.upper() in open('data/ftm_industries.txt').read().split('\n')

    # If using latex, replace ampersands with LaTeX-friendly ampersands
    if plt.rcParams['text.usetex']:
        label = label.replace('&', '\&')

    # Split the label by '_' and take the last part of the label, which is the industry_id
    # (the first part is the sector)
    label = label.split('_')[-1].title()

    # Replace incorrect title cased 'S
    label = label.replace("'S", "'s")
    return label


def get_expected_agreements(
        adj_matrix,
        ftm_industries,
        relation='support',
        source_industry='IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY',
):
    """
    Calculate the expected agreements between each industry_id and the source_industry. This is done by calculating
    the average number of supporting or opposing positions that each industry_id has on bills that the source_industry
    has a position on, divided by the total number of bills that the source_industry has a position on, and then
    divided by the total number of interest groups in each industry_id.

    Formally, let B be the bipartite adjacency matrix, where B[i, j] = 1 if interest group i supports bill j,
    B[i, j] = -1 if interest group i opposes bill j, and B[i, j] = 0 otherwise. Let S be the set of interest groups
    in the source industry_id, and let I be the set of interest groups in the industry_id of interest.

    The agreement/disagreement tensor is given by:

        A_{i,j,b} = \delta_{B[i, b], B[j, b]} - \delta_{B[i, b], -B[j, b]}

    where \delta_{i, j} is the Kronecker delta function.

    In the "agree" case, we then calculate the sum total of agreements between the source industry_id and the industry_id
    of interest by summing over the first two indices of A wherever A is equal to 1:

        T_{S, I} = sum_{i in S} sum_{j in I} A_{i, j, b} * \delta_{A_{i, j, b}, 1}

    To calculate the expected agreement between the source industry_id and the industry_id of interest, we divide by
    the total number of bills that the source industry_id has a position on, and then divide by the total number of
    interest groups in the industry_id of interest:

        k   = total number of bills that the source industry_id S has a position on
            = sum_{i in S} (sum_{j} B[i, j]) > 0
        |I| = total number of interest groups in the industry_id of interest

        E[agreement] = (1 / k) * (1 / |I|) * sum_{i in S} sum_{j in I} A_{i, j, b}

    To calculate the expected disagreement between the source industry_id and the industry_id of interest, we follow the
    same procedure, but sum over the first two indices of A wherever A is equal to -1:

        E[disagreement] = (1 / k) * (1 / |I|) * sum_{i in S} sum_{j in I} A_{i, j, b} * \delta_{A_{i, j, b}, -1}

    And finally, to calculate the expected net agreement between the source industry_id and the industry_id of interest,
    we follow the same procedure, but sum over the first two indices of A wherever A is equal to 1 or -1:

        E[net agreement] = (1 / k) * (1 / |I|) * sum_{i in S} sum_{j in I} A_{i, j, b} * \delta_{A_{i, j, b}, 1 or -1}

    :param adj_matrix: the adjacency matrix
    :param ftm_industries: a pandas dataframe with the industry_id names
    :param relation: the type of agreement to calculate. Must be one of 'support', 'oppose', or 'net'
    :param source_industry: the industry_id to compare all other industry_ids to
    :return: a pandas Series with the expected agreements between each industry_id and the source_industry
    """

    assert relation in ['support', 'oppose', 'net'], "relation must be one of 'support', 'oppose', or 'net'"
    assert source_industry in ftm_industries.values(), "source_industry must be in ftm_industries.industry_id"

    if relation == 'support':
        allowed_positions = [1]
    elif relation == 'oppose':
        allowed_positions = [-1]
    else:
        allowed_positions = [-1, 1]

    # Replace NaN values with 0
    adj_matrix = adj_matrix.replace(np.nan, 0)

    # Calculate the pairwise net agreement
    a = sparse.COO.from_numpy(adj_matrix.values)
    b = sparse.COO.from_numpy(adj_matrix.values[:, None])
    c = a * b
    d = c.copy() * (c == 0)
    for k in allowed_positions:
        d += c * (c == k)

    # 1. select an environmental group's position on a bill
    # 2. count how many groups from industry_id X support/oppose that position
    # 3. average this across all environmental groups' positions on bills
    idx_to_client = dict(zip(range(len(adj_matrix.index)), adj_matrix.index.values))
    idx_to_bill = dict(zip(range(len(adj_matrix.columns)), adj_matrix.columns.values))

    data = pd.DataFrame([*d.coords, d.data]).T
    data['source'] = data[0].map(idx_to_client)
    data['target'] = data[1].map(idx_to_client)
    data['bill'] = data[2].map(idx_to_bill)
    data['position_product'] = data[3]
    data['source_ftm'] = data.source.map(ftm_industries)
    data['target_ftm'] = data.target.map(ftm_industries)
    data = data.drop([0, 1, 2, 3], axis=1)

    source_num_positions = abs(adj_matrix[adj_matrix.index.map(ftm_industries) == source_industry]).sum().sum()

    data = data[data.source != data.target]

    summed_relations = data[data.source_ftm == source_industry].groupby(['source', 'bill', 'target_ftm'])[
        'position_product'].sum().unstack().replace(np.nan, 0).sum()

    summed_relations = data[data.source_ftm == source_industry].groupby(['target_ftm'])['position_product'].sum()

    industry_counts = data.drop_duplicates('target').target_ftm.value_counts()

    normalized_summed_positions = (summed_relations / source_num_positions / industry_counts).dropna()

    return normalized_summed_positions.sort_values()


class ConfigurationModel:
    """
    A configuration model for generating random bipartite graphs.
    This is a simple implementation of the configuration model described in
    Newman, M. E. J., Strogatz, S. H., & Watts, D. J. (2001). Random graphs with arbitrary degree distributions and
    their applications. Physical Review E, 64(2), 026118. https://doi.org/10.1103/PhysRevE.64.026118
    """

    @staticmethod
    def from_positions(positions):
        """
        Create a configuration model from a pandas dataframe of positions.
        :param positions: a pandas dataframe with columns 'client_uuid', 'bill_identifier', and 'position_numeric'
        :return: a ConfigurationModel
        """
        edges = positions[positions.position_numeric.isin([1, -1])].drop_duplicates(
            ['client_uuid', 'bill_identifier']
        )[['client_uuid', 'bill_identifier', 'position_numeric']]
        configmodel = ConfigurationModel(edges.values)
        configmodel.positions = positions
        return configmodel

    def __init__(self, edges):
        """
        Create a configuration model from a list of edges. The configuration model will have the same degree sequence
        as the given list of edges, for both the client_uuid and bill_identifier nodes, and for both the positive and
        negative edges, separately. We do this by creating a stub list for each node, and then randomly rewiring the
        stubs.
        :param edges: a list of edges, where each edge is a tuple of the form (client_uuid, bill_identifier, position_numeric)
        """
        self.edges = edges
        #
        self.ig_stubs = pd.DataFrame([[ig, pos] for ig, bill, pos in edges])
        self.bill_stubs = pd.DataFrame([[bill, pos] for ig, bill, pos in edges])
        self.blocks = {}

    def sample(self, position=None):
        """
        Sample a random bipartite graph from the configuration model.
        :param position: if None, sample a random bipartite graph. If 1 or -1, sample a random bipartite graph with the
            given position.
        :return:
        """

        # Randomly shuffle the stubs
        if position is None:
            i = self.ig_stubs.sample(len(self.ig_stubs), replace=False).reset_index(drop=True)
            b = self.bill_stubs.sample(len(self.bill_stubs), replace=False).reset_index(drop=True)
        elif position in [-1, 1]:
            # If a position is given, sample only from the stubs with that position
            i = self.ig_stubs[self.ig_stubs[1] == position].sample(len(self.ig_stubs[self.ig_stubs[1] == position]),
                                                                   replace=False).reset_index(drop=True)
            b = self.bill_stubs[self.bill_stubs[1] == position].sample(
                len(self.bill_stubs[self.bill_stubs[1] == position]), replace=False).reset_index(drop=True)
        else:
            raise ValueError(f"position is {position}, but must be either None, 1, or -1.")

        # Separate the stubs corresponding to positive and negative edges, because a negative stub cannot
        # be connected to a positive stub
        i_pos = i[i[1] == 1]
        b_pos = b[b[1] == 1]
        i_neg = i[i[1] == -1]
        b_neg = b[b[1] == -1]

        # Align the shuffled bill and interest group stubs of each position type
        positive_edges = pd.concat([i_pos, b_pos], axis=1).iloc[:, :3]
        negative_edges = pd.concat([i_neg, b_neg], axis=1).iloc[:, :3]

        # Recombine the positive and negative edges into a single dataframe
        edges = pd.concat([positive_edges, negative_edges])
        edges.columns = ['client_uuid', 'position_numeric', 'bill_identifier']
        edges['client_block'] = edges.client_uuid.map(self.blocks)
        edges['bill_block'] = edges.bill_identifier.map(self.blocks)

        return edges


def process_ncsl(topic_name):
    topics_split = topic_name.split(';')
    return [
        t.split('__')[-1].replace('_', ' ').title()
        for t in topics_split if 'energy' in t]


def extract_and_normalize_topics(ncsl, ael):
    """
    Extracts the normalized topics from the ncsl and ael columns.
    :param ncsl: the ncsl_topics column value
    :param ael: the ael_category column value
    :return: a comma-separated string of normalized topics
    """
    topics_to_add = []
    if isinstance(ael, str):
        topics_to_add += [ael]
    if isinstance(ncsl, str):
        ncsl_topics = process_ncsl(ncsl)
        for topic in ncsl_topics:
            topics_to_add += [topic]

    topics_to_add = set(topics_to_add)

    return ','.join(topics_to_add)