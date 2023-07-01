import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from linearmodels.panel.results import PanelEffectsResults
from matplotlib import pyplot as plt
from tqdm import tqdm
from stargazer.stargazer import Stargazer
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

from figures import plot_top_industries_figure, plot_topic_correlations_figures, plot_agree_probabilities, \
    plot_partisanship_figure
from utils import get_expected_agreements, get_bipartite_adjacency_matrix_kcore, ConfigurationModel

data_loc = 'data/'

if __name__ == '__main__':

    # plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # Load data
    positions = pd.read_parquet('data/climate_energy_positions.parquet', engine='pyarrow')
    clients = pd.read_parquet('data/climate_energy_clients.parquet')
    bills = pd.read_parquet('data/climate_energy_bills.parquet')
    all_bill_blocks = pd.read_excel('data/bill_blocks.xlsx')
    all_client_blocks = pd.read_excel('data/client_blocks.xlsx')

    # Create mapping from client_uuid to FTM industry_id
    client_uuid_to_ftm_industry = clients.set_index('client_uuid').ftm.to_dict()

    # Only retain data for the 12 states included in the paper
    states_included = ['IL', 'CO', 'TX', 'IA', 'MA', 'MD', 'MT', 'NE', 'AZ', 'WI', 'NJ', 'MO']
    positions = positions[
        positions.state.isin(states_included) &
        ((positions.record_type == 'lobbying') |
         ~positions.state.isin(['CO', 'MT']))  # remove CO and MT testimony
        ]
    clients = clients[clients.state.isin(states_included)]
    bills = bills[bills.state.isin(states_included)]
    positions['ftm_industry'] = positions.client_uuid.map(client_uuid_to_ftm_industry)

    # partisanship data from Boris Shor and Nolan McCarty, see https://dataverse.harvard.edu/dataverse/bshor
    partisanship = pd.read_table('data/statemetadata/shormccarty.tab')

    # GDP data from https://apps.bea.gov/regional/downloadzip.cfm
    production = pd.read_csv('data/statemetadata/industrygdp.csv')

    production['state'] = production.GeoName.map({
        'Texas': 'TX',
        'Massachussetts': 'MA',
        'Colorado': 'CO',
        'Missouri': 'MO',
        'Iowa': 'IA',
        'Nebraska': 'NE',
        'Wisconsin': 'WI',
        'New Jersey': 'NJ',
        'Arizona': 'AZ',
        'Montana': 'MT',
        'Illinois': 'IL',
        'Maryland': 'MD'
    })

    # State utility regulatory status
    deregulated = pd.Series({
        "MO": False,
        "IA": False,
        "NE": False,
        "CO": False,
        "MA": True,
        "MD": True,
        "NJ": True,
        "TX": True,
        "AZ": False,
        "WI": False,
        "IL": True,
        "MT": False})

    # Save the unique FTM industries to a file
    with open('data/ftm_industries.txt', 'w') as f:
        f.write('\n'.join(sorted(positions.ftm_industry.unique())))

    ###############################
    # Figure 1: Proportion of positions by each industry_id in each state
    ###############################
    # Identify the 20 most prevalent industries by number of unique bills lobbied on;
    # we take this as a mean-of-means of the proportion of bills lobbied on by each industry_id
    # in each state, excluding civil servants.

    no_civil_servants_no_duplicates_positions = positions[
        positions.ftm_industry.notna() &
        ~positions.ftm_industry.astype(str).str.contains('CIVIL SERVANTS/PUBLIC OFFICIALS')].drop_duplicates([
        'client_uuid', 'bill_identifier',
    ])  # remove duplicate positions on the same bill and remove civil servants

    top_industries = no_civil_servants_no_duplicates_positions.groupby('state').ftm_industry.value_counts(
        normalize=True).unstack().mean().sort_values()[::-1][:20].index.values

    # plot_top_industries_figure(positions)

    ###############################
    # Table 1: Summary statistics
    ###############################

    n_records = positions.groupby(['state', 'record_type']).apply(len)
    n_positions = positions.drop_duplicates(['client_uuid', 'bill_identifier']).groupby(['state', 'record_type']).apply(
        len)

    n_bills = positions.groupby(['state', 'record_type']).bill_identifier.nunique()
    n_clients = positions.groupby(['state', 'record_type']).client_uuid.nunique()

    n_sbm_clients = all_client_blocks.groupby(['state', 'record_type']).client_uuid.nunique()
    n_sbm_bills = all_bill_blocks.groupby(['state', 'record_type']).bill_identifier.nunique()

    years_covered = positions.groupby(['state', 'record_type']).year.apply(lambda y: f"{min(y)}-{max(y)}")

    table_1 = pd.concat([n_records, n_positions, n_bills, n_sbm_bills, n_clients, n_sbm_clients, years_covered], axis=1)
    table_1 = table_1.reindex(years_covered.index)
    table_1.columns = ['Records', 'Unique Positions', 'Bills', 'Bills (SBM)', 'IGs', 'IGs (SBM)', 'Years']
    table_1.index.names = ['State', 'Record Type']
    table_1.to_excel('tables/summary_statistics.xlsx')

    ###############################
    # Figure 2 and 3: bill topic-industry_id correlations
    ###############################

    comparison_industries = [top_industries[0], *top_industries[2:11]]
    # plot_topic_correlations_figures(positions, bills, comparison_industries[::-1])

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

    # plot_agree_probabilities(agree_probabilities, disagree_probabilities)

    ###############################
    # Figure 5: partisanship, GDP, and deregulation
    ###############################

    years = sorted(positions.year.astype(str).unique())

    def cast_float(x):
        try:
            return float(x)
        except ValueError:
            return np.nan

    for year in years:
        production[year] = production[year].apply(cast_float)

    mining_percent_gdp = production.groupby('state').apply(
        lambda region: (
                region[region.IndustryClassification.isin(["21", "324"])][years].sum(0) /
                region[region.LineCode == 1.0][years].iloc[0]))
    mining_percent_gdp.columns = mining_percent_gdp.columns.astype(int)
    mining_percent_gdp.columns.name = 'year'
    mining_percent_gdp = mining_percent_gdp.stack() * 100
    mining_percent_gdp.name = 'MiningPctGdp'

    passed = bills.set_index('bill_identifier').status.isin([4, 5]).to_dict()
    yearly_number_passed = positions[positions.bill_identifier.map(passed)].groupby(
        ['state', 'year']).bill_identifier.nunique()

    state_year_index = positions[positions.bill_identifier.map(passed)].groupby(
        ['state', 'unified_session']).count().index


    def industry_passed_session_stats(industry_id):
        if isinstance(industry_id, str):
            industry_positions = positions[
                positions.ftm_industry == industry_id
                ]
        elif isinstance(industry_id, list):
            industry_positions = positions[
                positions.ftm_industry.isin(industry_id)
            ]
        else:
            raise ValueError("industry_id must be a string or list of strings")

        industry_positions = industry_positions.sort_values('start_date')[::-1].drop_duplicates(
            ['client_uuid', 'bill_identifier'])

        session_to_year = positions.set_index('unified_session').year.to_dict()

        industry_positions_passed_bills = industry_positions[industry_positions.bill_identifier.map(passed)]
        net = industry_positions_passed_bills.groupby(['state', 'unified_session']).position_numeric.sum().reindex(
            state_year_index)
        passed_n_positions = industry_positions_passed_bills.groupby(['state', 'unified_session']).apply(len).reindex(
            state_year_index)
        passed_n_bills = industry_positions_passed_bills.groupby(
            ['state', 'unified_session']).bill_identifier.nunique().reindex(state_year_index)

        clients_in_industry = industry_positions.groupby('state').client_uuid.nunique()

        data = pd.concat([net, passed_n_positions, passed_n_bills], axis=1).reset_index(drop=False)
        data.columns = ['state', 'unified_session', 'net_on_passed', 'n_positions_on_passed', 'n_bills_passed']
        data['clients_in_industry'] = data.state.map(clients_in_industry)
        data['year'] = data.unified_session.map(session_to_year)
        data = data.groupby(['state', 'year']).sum(numeric_only=True)
        data['passed_net_pos_ratio'] = data.net_on_passed / (data.clients_in_industry * data.n_bills_passed)
        data['passed_avg_pos'] = data.net_on_passed / data.n_positions_on_passed

        return data

    enviro_positions = industry_passed_session_stats('IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY')
    oilgas_positions = industry_passed_session_stats(
        ['ENERGY & NATURAL RESOURCES_OIL & GAS', 'ENERGY & NATURAL RESOURCES_MINING'])
    elcutl_positions = industry_passed_session_stats('ENERGY & NATURAL RESOURCES_ELECTRIC UTILITIES')

    avgpartisanship = partisanship.groupby(['st', 'year']).apply(
        lambda x: x[['hou_chamber', 'sen_chamber']].mean().mean())
    avgpartisanship.name = 'AvgPartisanship'

    plotdata = pd.concat([mining_percent_gdp,
                          avgpartisanship,
                          enviro_positions.net_on_passed,
                          enviro_positions.passed_net_pos_ratio,
                          enviro_positions.passed_avg_pos,
                          oilgas_positions.net_on_passed,
                          oilgas_positions.passed_net_pos_ratio,
                          oilgas_positions.passed_avg_pos,
                          elcutl_positions.net_on_passed,
                          elcutl_positions.passed_net_pos_ratio,
                          elcutl_positions.passed_avg_pos], axis=1).reset_index()

    plotdata = plotdata.rename(columns={'level_0': 'state'})
    plotdata.columns = [
        'state', 'year', 'MiningPctGdp', 'AvgPartisanship',
        'EnviroNetPos', 'EnviroNetPosRatio', 'EnviroAvgPos',
        'OilGasNetPos', 'OilGasNetPosRatio', 'OilGasAvgPos',
        'ElcUtlNetPos', 'ElcUtlNetPosRatio', 'ElcUtlAvgPos']

    plotdata['deregulated'] = plotdata.state.map(deregulated)
    plotdata = plotdata[plotdata.state.isin(positions.state.unique())]

    plot_partisanship_figure(plotdata)

    ###############################
    # Regression Models for tables 2 and 3
    ###############################

    models = {}
    panel_models = {}
    for industry in ['Enviro', 'OilGas', 'ElcUtl']:

        # Calculate cross-sectional models
        y = plotdata[industry + 'AvgPos'].copy()
        X = sm.add_constant(plotdata[['MiningPctGdp', 'AvgPartisanship', 'deregulated', 'state', 'year']])
        X['deregulated'] = X.deregulated.astype(int, errors='ignore')

        notna = ~(y.isnull() | X.isnull().max(1))
        X_filtered, y_filtered = X[notna], y[notna]

        reg_data = pd.concat([X_filtered, y_filtered], axis=1)

        models[industry] = smf.ols(
            formula=f"{industry + 'AvgPos'} ~ MiningPctGdp + AvgPartisanship + deregulated", data=reg_data).fit()

        # Calculate panel models
        reg_data = plotdata.set_index(['state', 'year'])
        reg_data['state'] = pd.Categorical(plotdata.state.values)
        reg_data['year'] = pd.Categorical(plotdata.year.values)

        y = reg_data[industry + 'AvgPos']
        X = reg_data[['MiningPctGdp', 'AvgPartisanship']]
        notna = ~(y.isnull() | X.isnull().max(1))
        X_filtered, y_filtered = X[notna], y[notna]

        panel_models[industry] = PanelOLS(y_filtered, X_filtered, entity_effects=True, check_rank=False).fit()

    # Use the models and panel_models to create a LaTeX table formatted for the paper.
    # Six columns, one for each industry/fixed effects combination.
    # One row for each coefficient. Has standard errors in parentheses. Has p-values as asterisks.
    # A row for the R-squared.
    # A row for the number of observations.
    # A row for the state fixed effects.

    # Create a dictionary mapping industry names to the LaTeX names used in the paper.
    industry_names = {
        'Enviro': 'Pro-Environmental Policy',
        'OilGas': 'Oil and Gas',
        'ElcUtl': 'Electric Utilities',
    }

    regression_table = pd.DataFrame(index=['Mining', 'Partisanship', 'Deregulated', 'R-squared', 'N', 'State FE'])

    def get_model_coefficient_for_table(model, coefficient):
        coef = model.params[coefficient]
        if isinstance(model, PanelEffectsResults):
            se = model.std_errors[coefficient]
        else:
            se = model.bse[coefficient]
        p = model.pvalues[coefficient]

        n_asterisks = 0
        while p < 0.1 ** n_asterisks:
            n_asterisks += 1

        formatted_coef = f"{coef:.3f}" + (f"*" * n_asterisks) + f"\n ({se:.3f})"
        return formatted_coef

    for industry in ['Enviro', 'OilGas', 'ElcUtl']:
        # add the column for the cross-sectional model
        model = models[industry]
        regression_table[(industry_names[industry], 'Cross-Sectional')] = [
            get_model_coefficient_for_table(model, 'MiningPctGdp'),
            get_model_coefficient_for_table(model, 'AvgPartisanship'),
            get_model_coefficient_for_table(model, 'deregulated'),
            model.rsquared,
            model.nobs,
            'No']

        # add the column for the panel model
        model = panel_models[industry]
        regression_table[(industry_names[industry], 'Panel')] = [
            get_model_coefficient_for_table(model, 'MiningPctGdp'),
            get_model_coefficient_for_table(model, 'AvgPartisanship'),
            np.nan,
            model.rsquared,
            model.nobs,
            'Yes']

        # Save the regression table to an Excel file.
        regression_table.to_excel('tables/regression_table.xlsx')


    ###############################
    # Utility-Environmental Policy Figure
    ###############################

    def process_ncsl(topics):
        """Returns a list of topics from the NCSL database."""
        topics_split = topics.split(';')
        return [t.split('__')[-1].replace('_', ' ').title() for t in topics_split if 'energy' in t]

    def filter_topics(row):
        """Returns True if the bill is relevant to energy policy."""
        ncsl = process_ncsl(str(row.ncsl_topics))
        ael = row.ael_category
        return any(e in [*ncsl, ael] for e in energy_relevant_topics)

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
        'Renewable Energy Wind'}

    power_generation_bills = bills[bills.apply(filter_topics, 1)].bill_identifier.unique()

    utilities = 'ENERGY & NATURAL RESOURCES_ELECTRIC UTILITIES'
    enviros = 'IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY'

    def calculate_oppose_probability(state):
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

        return abs(disagree_probabilities.loc[enviros, utilities])


    def oneway_utilities_renewables_disagreements_config(state):
        state_positions = positions[
            (positions.state == state)
        ]

        configuration_model = ConfigurationModel.from_positions(state_positions)
        simulated_positions = configuration_model.sample()
        simulated_positions['ftm_industry'] = simulated_positions.client_uuid.map(client_uuid_to_ftm_industry)
        simulated_positions = simulated_positions[
            (simulated_positions.ftm_industry.isin([enviros,utilities])) &
            (simulated_positions.bill_identifier.isin(power_generation_bills))
            ]

        adj_matrix = get_bipartite_adjacency_matrix_kcore(simulated_positions, (1, 1))

        disagree_probabilities = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, "oppose", enviros)

        return abs(disagree_probabilities.loc[enviros, utilities])

    # Check if we have generated configuration model data.
    # If not, generate it.
    if not os.path.exists('data/position_simulations.csv'):
        print("Generating configuration model data...")
        states = positions.state.unique()
        data = {}
        for state in states:
            for trial in range(1000):
                data[(state, trial)] = oneway_utilities_renewables_disagreements_config(state)

        configuration_model_data = pd.Series(data)
        configuration_model_data.index.names = ['state', 'trial']
        configuration_model_data.name = 'disagreement_probability'
        configuration_model_data.to_csv('data/position_simulations.csv')
    else:
        configuration_model_data = pd.read_csv('data/position_simulations.csv', index_col=[0,1])

    # Calculate the actual observed probability of disagreement for each state.
    observed_probabilities = pd.Series({state: calculate_oppose_probability(state) for state in positions.state.unique()})

    # Calculate the p-value for each state.
    p_values = pd.Series({state: (configuration_model_data.loc[state] > observed_probabilities[state]).mean() for state in positions.state.unique()})

    # Plot the results.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(observed_probabilities, p_values, s=10, alpha=0.5)
    ax.set_xlabel('Observed Probability of Disagreement')
    ax.set_ylabel('p-value')
    ax.set_title('Observed vs. Simulated Probability of Disagreement')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')

