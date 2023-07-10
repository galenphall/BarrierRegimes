from main import positions
from num2words import num2words
import textwrap
import pandas as pd

# generates the numbers used in the Q1 text of the results section


def main():
    data = []
    output_string = ''

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

    top_industries = industry_percentages.fillna(0).mean().sort_values()[
        ::-1][:20].index.values

    data = []
    for state in no_civil_servants_no_duplicates_positions.state.unique():
        pos_state = no_civil_servants_no_duplicates_positions[no_civil_servants_no_duplicates_positions['state'] == state].copy(
        )
        len(pos_state[pos_state['ftm_industry'].isin(top_industries)])
        data.append({
            'state': state,
            'top_indust_count': len(pos_state[pos_state['ftm_industry'].isin(top_industries)]),
            'not_top_indust_count': len(pos_state[~pos_state['ftm_industry'].isin(top_industries)]),
            'top_industries': str(sorted(pos_state[pos_state['ftm_industry'].isin(top_industries)].ftm_industry.unique())),
            'top_industriesl': str(len(pos_state[pos_state['ftm_industry'].isin(top_industries)].ftm_industry.unique())),
            'nottop_industries': str(pos_state[~pos_state['ftm_industry'].isin(top_industries)].ftm_industry.unique()),
            'pct_of_state': len(pos_state[pos_state['ftm_industry'].isin(top_industries)])/len(pos_state)
        })
    df = pd.DataFrame(data)

    output_string += (' '.join(textwrap.wrap(
        f'Out of the {len(no_civil_servants_no_duplicates_positions.ftm_industry.unique())} industries identified in the dataset, the 20 most prevalent by average position count are presented in the figure; these account for between {int(df.pct_of_state.max() *100)}% ({df[df.pct_of_state == df.pct_of_state.max()].iloc[0,0]}) and {int(df.pct_of_state.min() *100)}% ({df[df.pct_of_state == df.pct_of_state.min()].iloc[0,0]}) of the total positions taken in each state (median: {int(df.pct_of_state.median() *100)}%) ')))

    # most active sector:
    n = 0
    states = []
    for i in range(len(industry_percentages)):
        if type(industry_percentages.iloc[i, :].name) is tuple:
            if industry_percentages.iloc[i, industry_percentages.columns.get_loc(top_industries[0])] == industry_percentages.iloc[i, :].max():
                n += 1
                states.append(industry_percentages.iloc[i, :].name)

    # average activity by sector (sessions where a group did not take a position are skipped)
    percentile = 0.9
    top_indust_activity = no_civil_servants_no_duplicates_positions[
        no_civil_servants_no_duplicates_positions['ftm_industry'] == top_industries[0]].copy()
    avg_pt = top_indust_activity.pivot_table(
        index='client_uuid', columns='unified_session', values='bill_identifier', aggfunc='count')
    avg_participation_by_session = avg_pt.mean(axis=1)
    avg_participation_by_session_avg = round(
        avg_participation_by_session.mean(), 2)
    avg_participation_by_session_percentile = round(
        avg_participation_by_session.quantile(percentile), 2)

    # identify most active client uuid
    most_active_client_pt = top_indust_activity.pivot_table(
        index='client_uuid', values='bill_identifier', aggfunc='count')
    most_active_client_pt_df = most_active_client_pt[most_active_client_pt['bill_identifier'] == most_active_client_pt.max()[
        0]]
    most_active_client_uuid = most_active_client_pt_df.index[0]
    most_active_client_bill_count = most_active_client_pt_df.iloc[0, 0]

    # map client name
    most_active_client_df = no_civil_servants_no_duplicates_positions[
        no_civil_servants_no_duplicates_positions['client_uuid'] == most_active_client_uuid].copy()
    most_active_client_names = most_active_client_df.client_name.unique()
    most_active_client_states = sorted(most_active_client_df.state.unique())

    output_string += (' '.join(textwrap.wrap(f'The {top_industries[0].rsplit("_")[-1].title()} sector emerges as the most active, holding the majority of positions in {num2words(n)} states. The average {top_industries[0].rsplit("_")[-1].lower()} company or trade association in our dataset lobbies or testifies on {avg_participation_by_session_avg} bills per session in a given state (of the sessions in which they were active); the most active, {most_active_client_names[0].title()}, lobbied on {most_active_client_bill_count} bills in total across {num2words(len(most_active_client_states))} states ({", ".join(most_active_client_states)}).')))
    output_string += (' '.join(textwrap.wrap(
        f'(Extra Info) The top industry was most active in these states: {states}. All names of top client in top industry: {most_active_client_names} ')))

    # 90th percentile of top industry:
    output_string += f'This average conceals a highly skewed distribution, however, with the {round(percentile*100, 0)}th percentile of utilities lobbying on {avg_participation_by_session_percentile} or more bills per session. '

    # least engaged industries
    indust_bill_counts = no_civil_servants_no_duplicates_positions.ftm_industry.value_counts().sort_values()
    bottom_25_pct_highest_bill_count = indust_bill_counts.quantile(
        q=.25, interpolation='higher')
    bottom_25_pct_number_of_orgs = len(
        indust_bill_counts[indust_bill_counts <= bottom_25_pct_highest_bill_count])

    output_string += (' '.join(textwrap.wrap(
        f'Some categories of organizations rarely took positions on climate and clean energy in these states; {bottom_25_pct_number_of_orgs} industries took positions on {bottom_25_pct_highest_bill_count} or fewer bills. ')))

    # sierra club

    #  get a list of all client uuids includes sierra club
    sierra_ids = no_civil_servants_no_duplicates_positions.loc[no_civil_servants_no_duplicates_positions['client_name'].str.lower(
    ).str.replace(' ', '').str.contains('sierraclub')].client_uuid.unique().tolist()

    s = no_civil_servants_no_duplicates_positions.loc[no_civil_servants_no_duplicates_positions['client_uuid'].isin(
        sierra_ids)].copy()

    output_string += (' '.join(textwrap.wrap(
        f'These include national-level organizations such as the Sierra Club (present in {len(s.state.unique())} states; who took positions on {len(s.bill_identifier.unique())} bills). ')))

    return output_string
