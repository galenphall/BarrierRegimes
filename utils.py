import networkx as nx
import numpy as np
import pandas as pd
import sparse
from matplotlib import pyplot as plt


def oneway_propensity(A_b, allowed=[1]):
    """
  Calculate the oneway propensity of a given bill matrix A_b.

  Parameters:
  A_b (pd.DataFrame): A matrix representing the support/opposition of legislators for a given bill.

  Returns:
  oneway_propensity (pd.DataFrame): A matrix representing the oneway propensity between the legislators.
  """

    A_b_in = A_b.copy()

    # Replace NaN values with 0
    A_b = A_b.replace(np.nan, 0)

    # Calculate the pairwise net agreement
    a = sparse.COO(A_b.values)
    b = sparse.COO(A_b.values[:, None])
    c = a * b
    d = c.copy() * 0
    for k in allowed:
        d += c * (c == k)

    pairwise_net_agreement = pd.DataFrame(d.sum(axis=2).todense(),
                                          index=A_b.index.copy(),
                                          columns=A_b.index.copy())

    # Helper function to calculate the number of unique bills for a legislator
    def set_length(u):
        return len(u.dropna())

    # Calculate the pairwise unique bills
    oneway_unique_bills = A_b_in.replace(0, np.nan).apply(set_length, axis=1)

    # Calculate pairwise propensity
    propensity = pairwise_net_agreement.div(oneway_unique_bills, axis=0)

    return propensity, oneway_unique_bills


def get_bipartite_adjacency_matrix_kcore(positions, k_core=(5, 2)):
    """
    Construct the adjacency matrix adj_matrix from the positions dataframe; adj_matrix is a pandas dataframe
    :param positions (pandas dataframe): the positions dataframe
    :param k_core (tuple): the minimum number of clients and bills for the k-core
    :return adj_matrix (pandas dataframe): the adjacency matrix
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


def grouped_oneway_alignment_matrix(positions, k_core, allowed_positions, groups):
    adj_matrix = get_bipartite_adjacency_matrix_kcore(positions, k_core)

    if adj_matrix is None:
        return None

    adj_matrix, uniqbills = oneway_propensity(adj_matrix, allowed_positions)
    adj_matrix.index = adj_matrix.index.copy()
    adj_matrix.columns = adj_matrix.columns.copy()
    adj_matrix.index.name = 'u'
    adj_matrix.columns.name = 'v'
    E = adj_matrix.stack().dropna().reset_index().rename(columns={0: 'alignment'})
    E['uniqbills'] = E.u.map(uniqbills)
    E['u_cli'] = E.u.map(lambda x: x.split("_")[0])
    E['v_cli'] = E.v.map(lambda x: x.split("_")[0])

    # group-to-group
    group_to_group_adj_matrix = E.groupby([E.u_cli.map(groups), E.v_cli.map(groups)]).apply(
        lambda data: (data.alignment * data.uniqbills).sum() / data.uniqbills.sum()
    ).unstack()

    return group_to_group_adj_matrix


def adjust_label(label):
    """
    Adjust the label for an industry so that it is more readable
    :param label: an FTM industry name
    :return:
    """

    # Assert that the label is a string in the data/fmt_indstries.txt file
    assert label.upper() in open('data/ftm_industries.txt').read().split('\n')

    # If using latex, replace ampersands with LaTeX-friendly ampersands
    if plt.rcParams['text.usetex']:
        label = label.replace('&', '\&')

    # Split the label by '_' and take the last part of the label, which is the industry
    # (the first part is the sector)
    label = label.split('_')[-1].title()

    # Replace incorrect title cased 'S
    label = label.replace("'S", "'s")
    return label


def get_expected_agreements(
        adj_matrix,
        ftm_industries,
        allowed=None,
        source_industry='IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY',
):
    """

    :param adj_matrix:
    :param ftm_industries:
    :param allowed:
    :param source_industry:
    :return:
    """

    if allowed is None:
        allowed = [1]

    # Replace NaN values with 0
    adj_matrix = adj_matrix.replace(np.nan, 0)

    # Calculate the pairwise net agreement
    a = sparse.COO.from_numpy(adj_matrix.values)
    b = sparse.COO.from_numpy(adj_matrix.values[:, None])
    c = a * b
    d = c.copy() * (c == 0)
    for k in allowed:
        d += c * (c == k)

    # 1. select an environmental group's position on a bill
    # 2. count how many groups from industry X support/oppose that position
    # 3. average this across all environmental groups' positions

    idx_to_client = dict(zip(range(len(adj_matrix.index)), adj_matrix.index.values))
    idx_to_bill = dict(zip(range(len(adj_matrix.columns)), adj_matrix.columns.values))

    data = pd.DataFrame([*d.coords, d.data]).T
    data['source'] = data[0].map(idx_to_client)
    data['target'] = data[1].map(idx_to_client)
    data['bill'] = data[2].map(idx_to_bill)
    data['source_ftm'] = data.source.map(ftm_industries)
    data['target_ftm'] = data.target.map(ftm_industries)

    source_num_positions = abs(adj_matrix[adj_matrix.index.map(ftm_industries) == source_industry]).sum().sum()

    data = data[data.source != data.target]

    expected_agreements = data[data.source_ftm == source_industry].groupby(['source', 'bill', 'target_ftm'])[
        3].sum().unstack().replace(np.nan, 0).sum()

    industry_counts = data.drop_duplicates('target').target_ftm.value_counts()

    expected_agreements_pct = (expected_agreements / source_num_positions / industry_counts).dropna()

    return expected_agreements_pct.sort_values()
