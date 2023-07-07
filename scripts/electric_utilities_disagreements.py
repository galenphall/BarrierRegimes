import os

import pandas as pd
from tqdm import tqdm

from figures import plot_utilities_p_disagree_robustness_check, plot_utilities_p_disagree_main
from utils import ConfigurationModel, get_expected_agreements, get_bipartite_adjacency_matrix_kcore

energy_relevant_topics = {
        'Climate Change',
        'Climate Change Emissions Reduction',
        'Electricity Generation',
        'Emissions',
        'Energy Development',
        'Energy Efficiency',
        'Financing Energy Efficiency And Renewable Energy',
        'Fossil Energy',
        'Fossil Energy Coal',
        'Fossil Energy Natural Gas',
        'Natural Gas Development',
        'Nuclear Energy Facilities',
        'Other Energy',
        'Renewable Energy',
        'Renewable Energy Hydrogren',
        'Renewable Energy Solar',
        'Renewable Energy Wind'
}

utilities = 'ENERGY & NATURAL RESOURCES_ELECTRIC UTILITIES'
enviros = 'IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY'


def process_ncsl(topics):
    """Returns a list of topics from the NCSL database."""
    topics_split = topics.split(';')
    return [t.split('__')[-1].replace('_', ' ').title() for t in topics_split if 'energy' in t]


def filter_topics(row):
    """Returns True if the bill is relevant to energy policy."""
    ncsl = process_ncsl(str(row.ncsl_topics))
    ael = row.ael_category
    return any(e in [*ncsl, ael] for e in energy_relevant_topics)

def main(recompute=False):

    from main import positions, bills, client_uuid_to_ftm_industry, deregulated

    # ensure the working directory is the root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    power_generation_bills = bills[bills.apply(filter_topics, 1)].bill_identifier.unique()

    def calculate_oppose_probability(state, power_generation_bills):
        """
        Calculates the probability that a utility company will oppose a pro-environmental policy position in a given state.
        :param state:
        :return:
        """
        state_positions = positions[
            (positions.state == state) &
            (positions.ftm_industry.isin([enviros, utilities])) &
            (positions.bill_identifier.isin(power_generation_bills))
            ]

        adj_matrix = get_bipartite_adjacency_matrix_kcore(state_positions, (1, 1))

        disagree_probabilities = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, "oppose", enviros)

        return abs(disagree_probabilities.loc[utilities])

    def calculate_oppose_probability_config(state, power_generation_bills):
        state_positions = positions[
            (positions.state == state)
        ]

        configuration_model = ConfigurationModel.from_positions(state_positions)
        simulated_positions = configuration_model.sample()
        simulated_positions['ftm_industry'] = simulated_positions.client_uuid.map(client_uuid_to_ftm_industry)
        simulated_positions = simulated_positions[
            (simulated_positions.ftm_industry.isin([enviros, utilities])) &
            (simulated_positions.bill_identifier.isin(power_generation_bills))
            ]

        adj_matrix = get_bipartite_adjacency_matrix_kcore(simulated_positions, (1, 1))

        disagree_probabilities = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, "oppose", enviros)
        if utilities not in disagree_probabilities.index:
            return 0

        return abs(disagree_probabilities.loc[utilities])

    def generate_null_model_disagreements(power_generation_bills):
        config_disagreements = {
            state: [calculate_oppose_probability_config(state, power_generation_bills) for i in tqdm(range(1000))]
            for state in positions.state.unique()}
        config_disagreements = pd.DataFrame(config_disagreements)
        config_disagreements.to_parquet('data/configuration_model_utility_disagreements.parquet')
        return config_disagreements

    if not os.path.exists('data/configuration_model_utility_disagreements.parquet') or recompute:
        print("Generating null model disagreements")
        config_disagreements = generate_null_model_disagreements(power_generation_bills)
    else:
        print("Loading null model disagreements")
        config_disagreements = pd.read_parquet('data/configuration_model_utility_disagreements.parquet')

    # Calculate the actual observed probability of disagreement for each state.
    data = pd.Series({state: calculate_oppose_probability(state, power_generation_bills)
      for state in positions.state.unique()
      }).reset_index()
    data.columns = ['state', 'alignment']
    data['deregulated'] = data.state.map(deregulated)
    data['expected'] = data.state.map(config_disagreements.mean())
    data['expected_low'] = data.state.map(config_disagreements.quantile(0.025))
    data['expected_high'] = data.state.map(config_disagreements.quantile(0.975))

    data = data.sort_values(['deregulated', 'alignment'])

    data['state_num'] = range(len(data))

    histdata = config_disagreements.T.stack().reset_index().iloc[:,[0,2]].rename(columns={'level_0':'state',0:'alignment'})

    plot_utilities_p_disagree_robustness_check(data, histdata)

    plot_utilities_p_disagree_main(data)


