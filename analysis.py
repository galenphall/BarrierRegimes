import pandas as pd
from matplotlib import pyplot as plt

from figures import plot_top_industries_figure, plot_topic_correlations_figures

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

    # State utility regulatory status from https://www.eia.gov/electricity/data/eia861m/
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
    n_positions = positions.drop_duplicates(['client_uuid','bill_identifier']).groupby(['state', 'record_type']).apply(len)

    n_bills = positions.groupby(['state', 'record_type']).bill_identifier.nunique()
    n_clients = positions.groupby(['state', 'record_type']).client_uuid.nunique()

    n_sbm_clients = all_client_blocks.groupby(['state','record_type']).client_uuid.nunique()
    n_sbm_bills = all_bill_blocks.groupby(['state','record_type']).bill_identifier.nunique()

    years_covered = positions.groupby(['state', 'record_type']).year.apply(lambda y: f"{min(y)}-{max(y)}")

    table_1 = pd.concat([n_records, n_positions, n_bills, n_sbm_bills, n_clients, n_sbm_clients, years_covered], axis = 1)
    table_1 = table_1.reindex(years_covered.index)
    table_1.columns = ['Records', 'Unique Positions', 'Bills', 'Bills (SBM)', 'IGs', 'IGs (SBM)', 'Years']
    table_1.index.names = ['State', 'Record Type']
    table_1.to_excel('tables/summary_statistics.xlsx')

    ###############################
    # Figure 2 and 3: bill topic-industry correlations
    ###############################

    comparison_industries = [top_industries[0], *top_industries[2:11]]
    plot_topic_correlations_figures(positions, bills, comparison_industries[::-1])


