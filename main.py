import pandas as pd
from matplotlib import pyplot as plt
import electric_utilities_disagreements

data_loc = 'data/'

# Load data
positions = pd.read_parquet('data/climate_energy_positions.parquet', engine='pyarrow')
clients = pd.read_parquet('data/climate_energy_clients.parquet')
bills = pd.read_parquet('data/climate_energy_bills.parquet')
all_bill_blocks = pd.read_excel('data/bill_blocks.xlsx')
all_client_blocks = pd.read_excel('data/client_blocks.xlsx')

# Create mapping from client_uuid to FTM industry_id
client_uuid_to_ftm_industry = clients.set_index('client_uuid').ftm.to_dict()

# partisanship data from Boris Shor and Nolan McCarty, see https://dataverse.harvard.edu/dataverse/bshor
partisanship = pd.read_table('data/statemetadata/shormccarty.tab')

# GDP data from https://apps.bea.gov/regional/downloadzip.cfm
production = pd.read_csv('data/statemetadata/industrygdp.csv')

# State utility regulatory status
deregulated = pd.read_csv('data/statemetadata/deregulated.csv').set_index('state').deregulated.to_dict()


# Save the unique FTM industries to a file
with open('data/ftm_industries.txt', 'w') as f:
    f.write('\n'.join(sorted(positions.ftm_industry.unique())))

if __name__ == '__main__':
    # plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

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

    years_covered = positions.groupby(['state', 'record_type']).year.apply(lambda y: f"{min(y)}-{max(y)}")

    table_1 = pd.concat([n_records, n_positions, n_bills, n_clients, years_covered], axis=1)
    table_1 = table_1.reindex(years_covered.index)
    table_1.columns = ['Records', 'Unique Positions', 'Bills', 'IGs', 'Years']
    table_1.index.names = ['State', 'Record Type']
    table_1.to_excel('tables/summary_statistics.xlsx')

    ###############################
    # Figure 2 and 3: bill topic-industry_id correlations
    ###############################

    comparison_industries = [top_industries[0], *top_industries[2:11]]
    # plot_topic_correlations_figures(positions, bills, comparison_industries[::-1])

    # agree_disagree_figures.main()

    # structural_factors.main()

    ###############################
    # Probability that Electric Utilities disagree with Pro-Environmental Policy
    ###############################

    electric_utilities_disagreements.main()


