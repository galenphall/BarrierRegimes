import importlib

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from utils import extract_and_normalize_topics
import krippendorff
import numpy as np

# Initialize data and load it into memory
positions = pd.read_parquet('data/climate_energy_positions.parquet', engine='pyarrow')
clients = pd.read_parquet('data/climate_energy_clients.parquet')
bills = pd.read_parquet('data/climate_energy_bills.parquet')

# Create mapping from client_uuid to FTM industry_id
client_uuid_to_ftm_industry = clients.set_index('client_uuid').ftm.to_dict()

# Ensure that positions are sorted by date
positions = positions.sort_values('start_date')

# Ensure that positions ftm_industry is same as client ftm
positions['ftm_industry'] = positions.client_uuid.map(client_uuid_to_ftm_industry)

# # Get a k-n core of clients and bills
# kcore = positions.copy()
# kcore = positions.groupby(['client_uuid', 'bill_identifier', 'ftm_industry']).position_numeric.sum().apply(np.sign).reset_index()
# l = 0
# while l != len(kcore):
#     kcore = kcore[kcore.client_uuid.isin(kcore.client_uuid.value_counts()[kcore.client_uuid.value_counts() >= 5].index)]
#     kcore = kcore[kcore.bill_identifier.isin(kcore.bill_identifier.value_counts()[kcore.bill_identifier.value_counts() >= 5].index)]
#     l = len(kcore)
#
# # Calculate Krippendorff's alpha for each industry
# for ftm_industry, group in kcore.groupby('ftm_industry'):
#     reliability_data = group.groupby(['client_uuid', 'bill_identifier']).position_numeric.sum().map(lambda x: np.sign(x)+3).unstack()
#
#     # Remove bills with only one position
#     reliability_data = reliability_data[reliability_data.count(axis=1) > 1]
#
#     # If length less than 5, skip
#     if len(reliability_data) < 5:
#         continue
#
#     # Calculate Krippendorff's alpha
#     try:
#         alpha = krippendorff.alpha(reliability_data.values, level_of_measurement='nominal')
#     except:
#         alpha = np.nan
#
#     print(f"{ftm_industry}: {alpha}")

# partisanship data from Boris Shor and Nolan McCarty, see https://dataverse.harvard.edu/dataverse/bshor
partisanship = pd.read_table('data/statemetadata/shormccarty.tab')
partycomp = pd.read_csv('data/statemetadata/partycomposition.csv')
partisanship = partisanship.merge(partycomp, on=['st','year'])
partisanship['hou_rep_pct'] = partisanship['hou_rep_ct'] / (partisanship['hou_rep_ct'] + partisanship['hou_dem_ct'])
partisanship['sen_rep_pct'] = partisanship['sen_rep_ct'] / (partisanship['sen_rep_ct'] + partisanship['sen_dem_ct'])

# GDP data from https://apps.bea.gov/regional/downloadzip.cfm
production = pd.read_csv('data/statemetadata/industrygdp.csv')

# State utility regulatory status
deregulated = pd.read_csv('data/statemetadata/deregulated.csv').set_index('state').deregulated.to_dict()

# Save the unique FTM industries to a file
with open('data/ftm_industries.txt', 'w') as f:
    f.write('\n'.join(sorted(positions.ftm_industry.unique())))

# Create a dataframe of dummy columns for each topic.
topics = bills[['ncsl_topics', 'ael_category']].apply(
    lambda row: extract_and_normalize_topics(*row.values),
    axis=1
)
topics_dummies = pd.DataFrame(topics.str.split(',').apply(
    lambda x: {key: 1 for key in x}).values.tolist()
                              ).fillna(0)

topics_dummies.index = bills.bill_identifier
topics_dummies = topics_dummies.groupby(topics_dummies.index).max()


if __name__ == '__main__':

    # Set some global matplotlib settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Minion Pro'

    # Set the math font to Minion Pro
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Minion Pro'
    plt.rcParams['mathtext.it'] = 'Minion Pro:italic'
    plt.rcParams['mathtext.bf'] = 'Minion Pro:bold'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # We need to import these here to avoid circular imports
    from scripts import bill_topic_correlations, most_active_industries, structural_factors, \
    electric_utilities_disagreements, environmental_industry_agree_probabilities

    top_industries = most_active_industries.main(True)
    
    comparison_industries = [top_industries[0], *top_industries[2:11]][::-1]
    comparison_topics = [
        'Renewable Energy Wind',
        'Renewable Energy Solar',
        'Fossil Energy Coal',
        'Fossil Energy Natural Gas',
        'Nuclear Energy Facilities',
        'Energy Efficiency',
        'Emissions'
    ]

    bill_topic_correlations.main(comparison_industries, comparison_topics, True)
    
    environmental_industry_agree_probabilities.main(True)

    structural_factors.main()

    electric_utilities_disagreements.main()