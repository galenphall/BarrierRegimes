import importlib

import pandas as pd
from matplotlib import pyplot as plt
from utils import extract_and_normalize_topics

# Initialize data and load it into memory
positions = pd.read_parquet('data/climate_energy_positions.parquet', engine='pyarrow')
clients = pd.read_parquet('data/climate_energy_clients.parquet')
bills = pd.read_parquet('data/climate_energy_bills.parquet')

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
    # plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # We need to import these here to avoid circular imports
    from scripts import bill_topic_correlations, most_active_industries, structural_factors, \
    electric_utilities_disagreements, environmental_industry_agree_probabilities

    # Reload the modules to ensure that changes are reflected as we iterate
    importlib.reload(bill_topic_correlations)
    importlib.reload(most_active_industries)
    importlib.reload(structural_factors)
    importlib.reload(electric_utilities_disagreements)
    importlib.reload(environmental_industry_agree_probabilities)

    top_industries = most_active_industries.main(False)

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

    # bill_topic_correlations.main(comparison_industries, comparison_topics)

    environmental_industry_agree_probabilities.main()

    # structural_factors.main()

    # electric_utilities_disagreements.main()
