import os

import pandas as pd
from adjustText import adjust_text
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

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

def main():

    from main import positions, bills, client_uuid_to_ftm_industry, deregulated

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

    if not os.path.exists('data/configuration_model_utility_disagreements.parquet'):
        config_disagreements = generate_null_model_disagreements(power_generation_bills)
    else:
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

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[3, 1], sharey=True)

    sns.boxplot(histdata, y='state', x='alignment', order=data.state.drop_duplicates(), ax=ax)

    ax.plot(data.alignment, data.state, marker='o', lw=0, mfc='w', mec='grey')
    ax.set_xlim(0, 0.23)
    ax.set_xticklabels([round(i*100,1) for i in ax.get_xticks()])
    ax.set_xlabel("P(disagree)")
    ax.hlines(6.5, 0, .23, 'k', ':')
    ax.set_xlim(0, 0.23)

    ax.text(0.15, "CO", "Regulated", verticalalignment='bottom')
    ax.text(0.15, "MA", "Deregulated", verticalalignment='top')

    ax2.barh(width=data.alignment-data.expected, y=data.state, edgecolor='k', color ='grey', height=0.5)
    ax2.vlines(data[data.deregulated].alignment.mean() - data[data.deregulated].expected.mean(), 7,11, 'r', '--')
    ax2.vlines(data[~data.deregulated].alignment.mean() - data[~data.deregulated].expected.mean(), 0,6, 'r', '--')
    ax2.hlines(6.5, 0, .13, 'k', ':')
    ax2.set_xticklabels([round(i*100,1) for i in ax2.get_xticks()])
    ax2.set_xlabel("observed - expected")

    fig.suptitle(
        "Probability of electric utilities opposing environmenal nonprofits:\n"+
        "configuration model versus observed\n", )

    fig.savefig("figures/figure_1_appendix.pdf", bbox_inches='tight')
    fig.savefig("figures/figure_1_appendix.png", dpi=300, bbox_inches='tight')

    #########################################
    # Figure 6b
    #########################################

    fig, ax4 = plt.subplots(1, 1, figsize=(3, 3))

    sns.swarmplot(
        data,
        x='deregulated',
        y='alignment',
        edgecolor='none',
        color='none',
        size=1,
        ax=ax4,
        legend=False)

    annotations = [
        ax4.annotate(
            state, [dereg, alignment],
            verticalalignment='center',
            horizontalalignment='center')
        for state, alignment, dereg in data[['state', 'alignment', 'deregulated']].values
    ]

    ax4.set_ylabel("P(Disagree) (\%)")
    ax4.set_yticklabels([round(i*100,1) for i in ax4.get_yticks()])
    ax4.set_xlabel("")
    ax4.set_xticklabels(['Regulated', 'Deregulated'])
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_title("Electric Utilities and\nPro-Environmental Policy", fontsize=10)

    adjust_text(annotations, autoalign=False, ax=ax4, only_move={'text': 'x', 'points': 'x', 'objects': 'x'})

    xlim = ax4.get_xlim()
    for a in annotations:
        x, y = a.get_position()
        ax4.plot(x, y, marker='o', mfc='grey', mec='none', lw=0, alpha=0.3, zorder=-1, ms=17)
    ax4.set_xlim(*xlim)

    ax4.set_ylim(0, ax4.get_ylim()[1] + 0.02)

    fig.savefig("figures/figure_6.png", dpi=300, bbox_inches='tight')
    fig.savefig("figures/figure_6.pdf", bbox_inches='tight')

