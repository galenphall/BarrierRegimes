import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from stargazer.stargazer import Stargazer
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

from figures import plot_top_industries_figure, plot_topic_correlations_figures, plot_agree_probabilities, \
    plot_partisanship_figure
from utils import get_expected_agreements, get_bipartite_adjacency_matrix_kcore

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

    # Create mapping from client_uuid to FTM industry
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
    # Figure 1: Proportion of positions by each industry in each state
    ###############################
    # Identify the 20 most prevalent industries by number of unique bills lobbied on;
    # we take this as a mean-of-means of the proportion of bills lobbied on by each industry
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
    # Figure 2 and 3: bill topic-industry correlations
    ###############################

    comparison_industries = [top_industries[0], *top_industries[2:11]]
    # plot_topic_correlations_figures(positions, bills, comparison_industries[::-1])

    ###############################
    # Figure 4: probabilities of agreement and disagreement
    # with pro-environmental policy groups
    ###############################

    agree_probabilities = {}
    disagree_probabilities = {}

    for (state, record_type), _ in tqdm(positions.groupby(['state', 'record_type'])):
        adj_matrix = get_bipartite_adjacency_matrix_kcore(positions[positions.state == state], (1,1))
        disagree_probabilities[state, record_type] = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, [-1])
        agree_probabilities[state, record_type] = get_expected_agreements(adj_matrix, client_uuid_to_ftm_industry, [1])

    agree_probabilities = pd.DataFrame(agree_probabilities).replace(np.nan, 0)
    disagree_probabilities = pd.DataFrame(disagree_probabilities).replace(np.nan, 0)

    agree_probabilities = agree_probabilities.loc[:, agree_probabilities.columns.sortlevel(1)[0].values]
    disagree_probabilities = disagree_probabilities.loc[:, disagree_probabilities.columns.sortlevel(1)[0].values]

    plot_agree_probabilities(agree_probabilities, disagree_probabilities)

    ###############################
    # Figure 5: partisanship, GDP, and deregulation
    ###############################

    years = sorted(positions.year.astype(str).unique())

    def cast_float(x):
        try:
            return float(x)
        except:
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

    mining_percent_gdp = mining_percent_gdp.unstack().reindex(columns=range(2000, 2022)).interpolate(
        axis=1, limit_direction='backward', method='linear').stack()

    passed = bills.set_index('bill_identifier').status.isin([4, 5]).to_dict()
    yearly_npassed = positions[positions.bill_identifier.map(passed)].groupby(
        ['state', 'year']).bill_identifier.nunique()

    state_year_index = positions[positions.bill_identifier.map(passed)].groupby(
        ['state', 'unified_session']).count().index


    def industry_passed_session_stats(industry):
        """

        :param industry:
        :return:
        """
        if isinstance(industry, str):
            industry_positions = positions[
                positions.ftm_industry == industry
                ]
        elif isinstance(industry, list):
            industry_positions = positions[
                positions.ftm_industry.isin(industry)
            ]
        else:
            raise ValueError("industry must be a string or list of strings")

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

        n_clients = industry_positions.groupby('state').client_uuid.nunique()

        data = pd.concat([net, passed_n_positions, passed_n_bills], axis=1).reset_index(drop=False)
        data.columns = ['state', 'unified_session', 'net_on_passed', 'n_positions_on_passed', 'n_bills_passed']
        data['n_clients'] = data.state.map(n_clients)
        data['year'] = data.unified_session.map(session_to_year)
        data = data.groupby(['state', 'year']).sum(numeric_only=True)
        data['passed_net_pos_ratio'] = data.net_on_passed / (data.n_clients * data.n_bills_passed)
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
                          enviro_positions.passed_avg_pos,
                          oilgas_positions.passed_avg_pos,
                          elcutl_positions.passed_avg_pos], axis=1).reset_index()

    plotdata = plotdata.rename(columns={'level_0': 'state'})
    plotdata.columns = [
        'state', 'year', 'MiningPctGdp', 'AvgPartisanship',
        'EnviroAvgPos',
        'OilGasAvgPos',
        'ElcUtlAvgPos']

    plotdata['deregulated'] = plotdata.state.map(deregulated)
    plotdata = plotdata[plotdata.state.isin(positions.state.unique())]

    plot_partisanship_figure(plotdata)

    ###############################
    # Regression Models for tables 2 and 3
    ###############################

    X = sm.add_constant(plotdata[['MiningPctGdp', 'AvgPartisanship', 'deregulated', 'state', 'year']])
    X['deregulated'] = X.deregulated.astype(int, errors='ignore')
    X['testimony'] = X.state.isin(['TX', 'IL', 'MD', 'AZ']).astype(int)

    models = {}
    panel_models = {}
    for industry in ['Enviro', 'OilGas', 'ElcUtl']:

        # Calculate cross-sectional models
        y = (plotdata[industry + 'AvgPos'] + 1) / 2

        notna = ~(y.isna() | X.isna().max(1))
        y_filtered = y[notna]
        X_filtered = X[notna]

        reg_data = pd.concat([X_filtered, y_filtered], 1)

        models[industry] = smf.ols(
            formula=f"{industry + 'AvgPos'} ~ MiningPctGdp + AvgPartisanship + deregulated", data=reg_data).fit()

        # Calculate panel models
        state = plotdata.state.values
        year = plotdata.year.values
        reg_data = plotdata.set_index(['state', 'year'])
        reg_data['state'] = pd.Categorical(state)
        reg_data['year'] = pd.Categorical(year)

        y = reg_data[industry + 'AvgPos']

        X = reg_data[['MiningPctGdp', 'AvgPartisanship']]

        notna = ~(y.isna() | X.isna().max(1))
        y_filtered = y[notna]
        X_filtered = X[notna]

        panel_models[industry] = PanelOLS(y_filtered, X_filtered, entity_effects=True, check_rank=False).fit()

    stargazer = Stargazer([models['Enviro'], models['OilGas'], models['ElcUtl'], panel_models['Enviro'], panel_models['OilGas'], panel_models['ElcUtl']])
    stargazer.rename_covariates({'const': 'Intercept', 'MiningPctGdp': 'Mining % GDP', 'AvgPartisanship': 'Avg. Partisanship', 'deregulated': 'Deregulated'})
    stargazer.covariate_order(['Intercept', 'Mining % GDP', 'Avg. Partisanship', 'Deregulated'])
    stargazer.custom_columns(['Enviro', 'Oil & Gas', 'Electric Utilities', 'Enviro', 'Oil & Gas', 'Electric Utilities'], [1, 1, 1, 2, 2, 2])
    stargazer.show_model_numbers(False)
    stargazer.show_degrees_of_freedom(False)
    stargazer.show_residual_std_err(False)
    stargazer.show_n(True)
    stargazer.show_r2(True)

    # indicate state fixed effects
    stargazer.add_line('State Fixed Effects', ['No', 'No', 'No', 'Yes', 'Yes', 'Yes'], 4)

    # save table as latex
    stargazer.render_latex('regression_results.tex')



