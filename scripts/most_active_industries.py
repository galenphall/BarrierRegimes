from main import positions
from figures import plot_top_industries_figure


def main():
    no_civil_servants_no_duplicates_positions = positions[
        positions.ftm_industry.notna() &
        ~positions.ftm_industry.astype(str).str.contains('CIVIL SERVANTS/PUBLIC OFFICIALS')].drop_duplicates([
        'client_uuid', 'bill_identifier',
    ])  # remove duplicate positions on the same bill and remove civil servants

    # Collect the percentage of positions by each industry_id in each state
    industry_percentages = no_civil_servants_no_duplicates_positions.groupby(
        ['record_type', 'state']).ftm_industry.value_counts(normalize=True).unstack()
    industry_percentages.loc[''] = None
    industry_percentages.loc['Total'] = industry_percentages.mean()

    top_industries = industry_percentages.fillna(0).mean().sort_values()[::-1][:20].index.values

    # Create a table of the top industries in each state
    table_data = (industry_percentages[top_industries] * 100).round(2)
    table_data.columns = table_data.columns.str.split("_").str[1].str.title()

    plot_top_industries_figure(table_data)

    return top_industries
