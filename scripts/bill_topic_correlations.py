from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from figures import plot_topic_correlation_coefficients, plot_topic_correlation_scatter
from main import positions, bills, topics_dummies


def create_adjacency_matrices():
    adj_matrix = positions.drop_duplicates(['bill_identifier', 'client_uuid']).pivot_table(
        'position_numeric',
        'bill_identifier',
        'ftm_industry',
        'sum')

    clients_per_state_industry = positions.pivot_table(
        'client_uuid',
        'state',
        'ftm_industry',
        'nunique')

    # Retain the industries in adj_matrix
    industries = [*adj_matrix.columns]

    # Extract state from bill identifiers
    adj_matrix['state'] = adj_matrix.index.map(lambda bill_identifier: bill_identifier[:2])

    # Boolean - whether or not bill passed legislature
    adj_matrix['passed'] = adj_matrix.index.map(bills.set_index('bill_identifier').status.isin([4, 5]).to_dict())

    # Divide industry_id sums by number of interest groups per industry_id in each state
    adj_matrix[industries] = adj_matrix[industries] / adj_matrix.state.apply(lambda state: clients_per_state_industry.loc[state])

    # Replacing zeros with nans ensures that the correlations only account for bills on which *both*
    # industries have a non-zero value.
    adj_matrix_intersection = adj_matrix.replace(0, np.nan)

    # Replacing nans with zeros ensures that the correlations account for all bills on which *either*
    # industry has a non-zero value.
    adj_matrix_union = adj_matrix.fillna(0)  # Union matrix has values in all rows

    return adj_matrix_intersection, adj_matrix_union, adj_matrix


def create_correlations(comparison_topics, comparison_industries, adj_matrix_intersection, adj_matrix_union,
                        target_industry='IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY'):
    """
    For each topic, calculate the correlation between each industry in comparison_industries and the
    'IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY' industry. We calculate two sets of correlations:
    1) The correlation between the two industries on bills where both industries have a non-zero value.
    2) The correlation between the two industries on all bills where either industry has a non-zero value.
    :param target_industry:
    :param comparison_topics:
    :param comparison_industries:
    :param adj_matrix_intersection:
    :param adj_matrix_union:
    :return:
    """

    assert target_industry in adj_matrix_intersection.columns
    assert target_industry in adj_matrix_union.columns
    assert all([industry in adj_matrix_intersection.columns for industry in comparison_industries])
    assert all([industry in adj_matrix_union.columns for industry in comparison_industries])
    assert all([topic in topics_dummies.columns for topic in comparison_topics])

    def get_correlations(adj_matrix):
        """
        Calculate the correlation between each industry in comparison_industries and the target_industry.

        :param adj_matrix: the adjacency matrix to use, which should have cell values between -1 and 1 indicating the
            proportion of interest groups in each industry that supported/opposed the bill.
        :return: a dictionary of dictionaries of correlations, with the structure:
            {topic: {comparison_industry: correlation}}
        """

        adj_matrix = adj_matrix.copy()

        correlations = defaultdict(dict)

        for comparison in comparison_industries:

            # Select the columns for the two industries we want to compare
            selected_industries = [comparison, target_industry]

            for topic in comparison_topics:

                # Select bills on which the topic is present
                bills_to_include = topics_dummies[topics_dummies[topic] == 1].index
                filtered_adj_matrix = adj_matrix.reindex(index=bills_to_include, columns=selected_industries).copy()

                if filtered_adj_matrix.notna().min(1).sum() < 2:
                    # If there are fewer than two bills on which both industries have a non-zero value,
                    # we cannot calculate a correlation.
                    correlations[topic][comparison] = None
                else:
                    # Calculate the correlation between the two industries
                    # Drop rows with any nans
                    filtered_adj_matrix = filtered_adj_matrix.dropna(axis=0)

                    # Drop rows with all zeros
                    filtered_adj_matrix = filtered_adj_matrix[(filtered_adj_matrix != 0).any(axis=1)]

                    # Calculate correlation
                    correlations[topic][comparison] = pearsonr(filtered_adj_matrix[selected_industries].values[:, 0],
                                                               filtered_adj_matrix[selected_industries].values[:, 1])

        return correlations

    union_correlations = get_correlations(adj_matrix_union)
    intersection_correlations = get_correlations(adj_matrix_intersection)

    return union_correlations, intersection_correlations


def main(comparison_industries, comparison_topics):

    adj_matrix, adj_matrix_union, adj_matrix_intersection = create_adjacency_matrices()
    union_correlations, intersection_correlations = create_correlations(
        comparison_topics=comparison_topics,
        comparison_industries=comparison_industries,
        adj_matrix_intersection=adj_matrix_intersection,
        adj_matrix_union=adj_matrix_union)

    plot_topic_correlation_coefficients(comparison_industries, intersection_correlations, union_correlations)
    plot_topic_correlation_scatter(adj_matrix, adj_matrix_union, adj_matrix_intersection, topics_dummies)
