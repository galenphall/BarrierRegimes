import pandas as pd

from main import positions

def main():
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