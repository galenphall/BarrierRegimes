import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from linearmodels.panel.results import PanelEffectsResults

from main import positions, production, partisanship, deregulated, bills

import statsmodels.api as sm
import statsmodels.formula.api as smf

from figures import plot_partisanship_figure


def main():

    def cast_float(x):
        try:
            return float(x)
        except ValueError:
            return np.nan

    years = sorted(positions.year.astype(str).unique())
    for year in years:
        production[year] = production[year].apply(cast_float)

    # Calculate the percent of state GDP from mining (which includes fossil fuel
    # extraction) and petrochemical production/refining.
    mining_percent_gdp = production.groupby('state').apply(
        lambda region: (
                region[region.IndustryClassification.isin(["21", "324"])][years].sum(0) /
                region[region.LineCode == 1.0][years].iloc[0]))
    mining_percent_gdp.columns = mining_percent_gdp.columns.astype(int)
    mining_percent_gdp.columns.name = 'year'
    # Backfill missing values
    mining_percent_gdp = mining_percent_gdp.fillna(method='bfill', axis=1)
    mining_percent_gdp = mining_percent_gdp.stack() * 100
    mining_percent_gdp.name = 'MiningPctGdp'

    passed = bills.set_index('bill_identifier').status.isin([4, 5]).to_dict()

    state_year_index = positions[positions.bill_identifier.map(passed)].groupby(
        ['state', 'unified_session']).count().index

    def industry_passed_session_stats(industry_id):
        """
        Calculate the average position on passed bills for a given industry or
        set of industries
        :param industry_id: A string or list of strings of industry ids
        :return: A dataframe with the average position on passed bills for each
        state and year from the given industry
        """
        # Select positions from this industry/ies
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

        # Select the latest available positions from each client on each bill,
        # as these are the positions stated on the version closest to the one which actually passed
        industry_positions = industry_positions.sort_values('start_date', ascending=False).drop_duplicates(
            ['client_uuid', 'bill_identifier'])

        # Map the session of each bill's passage to its year
        session_to_year = positions.groupby(['unified_session', 'state']).year.max().to_dict()

        # Select positions on passed bills
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
        data['year'] = data.apply(lambda row: session_to_year[(row.unified_session, row.state)], axis=1)
        data = data.groupby(['state', 'year']).sum(numeric_only=True)
        data['passed_avg_pos'] = data.net_on_passed / data.n_positions_on_passed

        return data

    # Get the average position on passed bills for each industry
    enviro_positions = industry_passed_session_stats('IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY')
    oilgas_positions = industry_passed_session_stats(
        ['ENERGY & NATURAL RESOURCES_OIL & GAS', 'ENERGY & NATURAL RESOURCES_MINING'])
    elcutl_positions = industry_passed_session_stats('ENERGY & NATURAL RESOURCES_ELECTRIC UTILITIES')

    # Calculate average partisanship between both chambers in each state and year.
    # Replace with "sen_majority" and "hou_majority" to use majority party instead of chamber.
    # Replace with "sen_rep_pct" and "hou_rep_pct" to use pct of Republicans instead of partisanship.
    avgpartisanship = partisanship.groupby(['st', 'year']).apply(
        lambda x: x[['hou_majority', 'sen_majority']].mean().mean())
    avgpartisanship.name = 'AvgPartisanship'

    plotdata = pd.concat([mining_percent_gdp,
                          avgpartisanship,
                          enviro_positions.passed_avg_pos,
                          oilgas_positions.passed_avg_pos,
                          elcutl_positions.passed_avg_pos], axis=1).reset_index()
    plotdata.columns = [
        'state', 'year', 'MiningPctGdp', 'AvgPartisanship','EnviroAvgPos','OilGasAvgPos','ElcUtlAvgPos']
    plotdata['deregulated'] = plotdata.state.map(deregulated)
    plotdata = plotdata[plotdata.state.isin(positions.state.unique())]

    plot_partisanship_figure(plotdata)

    plotdata.to_csv('tables/structural_factors_plotdata.csv', index=False)

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

    regression_table = pd.DataFrame(index=['Intercept', 'Mining', 'Partisanship', 'Deregulated', 'R-squared', 'N', 'State FE'])

    def get_model_coefficient_for_table(model, coefficient):
        coef = model.params[coefficient]
        if isinstance(model, PanelEffectsResults):
            se = model.std_errors[coefficient]
        else:
            se = model.bse[coefficient]
        p = model.pvalues[coefficient]

        if p < 0.01:
            n_asterisks = 3
        elif p < 0.05:
            n_asterisks = 2
        elif p < 0.1:
            n_asterisks = 1
        else:
            n_asterisks = 0

        formatted_coef = f"{coef:.3f}" + (f"*" * n_asterisks) + f"\n ({se:.3f})"
        return formatted_coef

    for industry in ['Enviro', 'OilGas', 'ElcUtl']:
        # add the column for the cross-sectional model
        model = models[industry]
        regression_table[(industry_names[industry], 'Cross-Sectional')] = [
            get_model_coefficient_for_table(model, 'Intercept'),
            get_model_coefficient_for_table(model, 'MiningPctGdp'),
            get_model_coefficient_for_table(model, 'AvgPartisanship'),
            get_model_coefficient_for_table(model, 'deregulated'),
            model.rsquared,
            model.nobs,
            'No']

        # add the column for the panel model
        model = panel_models[industry]
        regression_table[(industry_names[industry], 'Panel')] = [
            np.nan, # no intercept in panel model
            get_model_coefficient_for_table(model, 'MiningPctGdp'),
            get_model_coefficient_for_table(model, 'AvgPartisanship'),
            np.nan, # no deregulated in panel model because it is absorbed by the state fixed effects
            model.rsquared,
            model.nobs,
            'Yes']

        # Save the regression table to an Excel file.
        regression_table.to_excel('tables/regression_table.xlsx')
        regression_table.to_csv('tables/regression_table.csv')
