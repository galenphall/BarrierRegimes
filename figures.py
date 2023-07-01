import re
import textwrap
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, linregress
import matplotlib as mpl

from matplotlib.patches import Rectangle

from utils import adjust_label


def plot_top_industries_figure(positions):
    """
    Create Figure 1: Heatmap of top industries in each state
    :param positions: the positions dataframe
    :return: None
    """
    no_civil_servants_no_duplicates_positions = positions[
        positions.ftm_industry.notna() &
        ~positions.ftm_industry.astype(str).str.contains('CIVIL SERVANTS/PUBLIC OFFICIALS')].drop_duplicates([
        'client_uuid', 'bill_identifier',
    ])  # remove duplicate positions on the same bill and remove civil servants

    top_industries = no_civil_servants_no_duplicates_positions.groupby('state').ftm_industry.value_counts(
        normalize=True).unstack().mean().sort_values()[::-1][:20].index.values

    # Collect the percentage of positions by each industry in each state
    industry_percentages = no_civil_servants_no_duplicates_positions.groupby(
        ['record_type', 'state']).ftm_industry.value_counts(normalize=True).unstack()
    industry_percentages.loc[''] = None
    industry_percentages.loc['Total'] = industry_percentages.mean()

    # Create a table of the top industries in each state
    table_data = (industry_percentages[top_industries] * 100).round(2).replace(0, np.nan)
    table_data.columns = table_data.columns.str.split("_").str[1].str.title()

    # Replace ampersands with LaTeX-friendly ampersands, if we are using LaTeX
    if plt.rcParams['text.usetex']:
        table_data.columns.str.replace("&", r"\&")

    # Create a heatmap of the top industries in each state
    fig, (cbar_ax, ax) = plt.subplots(2, 1, figsize=(6.5, 11), height_ratios=[1, 20])

    ax = sns.heatmap(
        table_data.T,
        cmap='magma_r',
        ax=ax,
        square=True,
        cbar_ax=cbar_ax,
        cbar_kws={'orientation': 'horizontal'},
        vmin=0, vmax=25,
        linewidths=0.5,
        linecolor='white',
        annot=True, fmt=".1f"
    )

    # Highlight top industry in each state
    for state, row in table_data.iterrows():

        if state == "":
            continue

        # Get the top industry for this state
        top_industry = row.idxmax()

        # Get the x and y coordinates of the top industry in the heatmap
        y = table_data.columns.get_loc(top_industry)
        x = table_data.index.get_loc(state)

        # Plot a black box around the top industry
        ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='k', lw=3))

    ax.set_ylabel("Industry")
    ax.set_xlabel("\n\nState and Record Type")

    # Format x and y axis labels; figure out where the testimony states start and end
    new_labels = []
    i = 0
    testimony_start = 0
    testimony_end = 0
    for text in ax.get_xticklabels():
        label = text.get_text()
        if label == '':
            testimony_end = i
        elif label == 'Total':
            label = 'Interstate\nmean'
        else:
            rectype, label = re.search("\('(.*)', '(.*)'\)", label).groups()
            if (rectype == "testimony") and (testimony_start == 0):
                testimony_start = i
        new_labels.append(label)
        i += 1

    # Add arrows and labels to indicate lobbying and testimony states
    kwargs = dict(horizontalalignment='center', xycoords='data',
                  arrowprops={'arrowstyle': '|-|', 'color': 'k', 'mutation_scale': 1.5},
                  annotation_clip=False)

    ax.annotate("", [0, 21], [testimony_start, 21], **kwargs)
    ax.annotate("", [testimony_start, 21], [testimony_end, 21], **kwargs)

    kwargs['arrowprops'] = None
    ax.annotate("Lobbying", [testimony_start / 2, 21.5], **kwargs)
    ax.annotate("Testimony", [testimony_start + (testimony_end - testimony_start) / 2, 21.5], **kwargs)

    # Format x and y axis labels
    ax.set_xticklabels(new_labels, rotation=0)
    ax.set_yticklabels(['\n'.join(textwrap.wrap(label.get_text(), 30)) for label in ax.get_yticklabels()])
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        labelbottom=True)

    # Format colorbar
    cbar_ax.set_xlabel("Percent of positions from industry (incl. neutral)")
    cbar_ax.set_xticks([0, 5, 10, 15, 20, 25])
    cbar_ax.set_xticklabels([f"{i:.0f}%" for i in [0, 5, 10, 15, 20, 25]])
    [s.set_visible(True) for s in cbar_ax.spines.values()]

    fig.savefig('figures/top_20_industries.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_topic_correlations_figures(positions: pd.DataFrame,
                                    bills: pd.DataFrame,
                                    comparison_industries: list,
                                    do_scatterplots=True,
                                    do_correlations=True
                                    ):
    """
    Creates a scatterplot showing the average position of Electric Utilities against the
    average position of Pro-Environmental Policy groups for bills in two topic categories:
    Renewable Energy - Wind and Renewable Energy - Solar.

    :param positions: the positions dataframe
    :param bills: the bills dataframe
    :param comparison_industries: the comparison industries list
    :param do_scatterplots: whether to create scatterplots
    :param do_correlations: whether to calculate correlations
    :return: None
    """

    ###############################
    # Create data for scatterplot #
    ###############################

    # Extract clean ncsl topic_name from the topic_name column
    def process_ncsl(topic_name):
        topics_split = topic_name.split(';')
        return [
            t.split('__')[-1].replace('_', ' ').title()
            for t in topics_split if 'energy' in t]

    # Extract clean ael and ncsl topic_name from the topic_name column
    def extract_and_normalize_topics(ncsl, ael):
        topics_to_add = []
        if isinstance(ael, str):
            topics_to_add += [ael]
        if isinstance(ncsl, str):
            ncsl_topics = process_ncsl(ncsl)
            for topic in ncsl_topics:
                topics_to_add += [topic]

        topics_to_add = set(topics_to_add)

        return ','.join(topics_to_add)

    # Create a dataframe of dummy columns for each topic.
    topics = bills[['ncsl_topics', 'ael_category']].apply(
        lambda row: extract_and_normalize_topics(*row.values),
        axis=1
    )
    topics_dummies = pd.DataFrame(topics.str.split(',').apply(
        lambda x: {key: 1 for key in x}).values.tolist()
                                  ).replace(np.nan, 0)

    topics_dummies.index = bills.bill_identifier
    topics_dummies = topics_dummies.groupby(topics_dummies.index).max()

    adj_matrix = positions[
        positions.client_uuid.isin(positions.client_uuid)  # Keep only active interest groups
    ].drop_duplicates(['bill_identifier', 'client_uuid']).pivot_table(
        'position_numeric',
        'bill_identifier',
        'ftm_industry',
        'sum')

    n_clients = positions[
        positions.client_uuid.isin(positions.client_uuid)  # Keep only active interest groups
    ].pivot_table(
        'client_uuid',
        'state',
        'ftm_industry',
        'nunique')

    # Retain the industries in A
    industries = [*adj_matrix.columns]

    # Extract state from bill identifiers
    adj_matrix['state'] = adj_matrix.index.map(lambda bill_identifier: bill_identifier[:2])

    # Boolean - whether or not bill passed legislature
    adj_matrix['passed'] = adj_matrix.index.map(bills.set_index('bill_identifier').status.isin([4, 5]).to_dict())


    # Divide industry sums by number of interest groups per industry in each state
    adj_matrix[industries] = adj_matrix[industries] / adj_matrix['state'].apply(lambda state: n_clients.loc[state])

    adj_matrix_intersection = adj_matrix.replace(0, np.nan)
    adj_matrix_union = adj_matrix.replace(np.nan, 0)

    ###############################
    # Create scatterplots
    ###############################
    if do_scatterplots:
        fig, axes = plt.subplots(2, 1, figsize=(3, 6), sharex=True, sharey=True)

        axes = axes.flat

        kwargs = dict(
            x='IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY',
            y='ENERGY & NATURAL RESOURCES_ELECTRIC UTILITIES',
            s=7,
            alpha=1,
        )

        passed_colors = {True: 'k', False: 'grey'}
        passed_lws = {True: 1, False: 0}

        vlines_kwargs = [0, -1, 1, 'grey', ':']

        label_adjust = lambda c: c.replace('&', '\&').split('_')[-1].title()

        def annotate_stats(ax, *dfs):

            dy = 0
            labels = ['u', 'i', 's']
            for df, label in zip(dfs, labels):
                df_copy = df[[kwargs['x'], kwargs['y']]]
                df_copy = df_copy[abs(df_copy).sum(1) > 0]

                # Annotate the axis with the pearson correlation
                r = df_copy.corr(numeric_only=True).loc[kwargs['x'], kwargs['y']]
                p = df_copy.corr(numeric_only=True, method=lambda x, y: pearsonr(x, y)[1]).loc[kwargs['x'], kwargs['y']]
                slope = df_copy.corr(numeric_only=True, method=lambda x, y: linregress(x, y)[0]).loc[
                    kwargs['x'], kwargs['y']]
                n = len(df_copy.dropna(axis=0))
                ax.annotate(r"$r_{% s} = %.2f$" % (label, r), [1.2, 1 - dy], transform=ax.transData, annotation_clip=False)
                ax.annotate(r"$\beta_{% s} = %.2f$" % (label, slope), [1.2, 0.8 - dy], transform=ax.transData,
                            annotation_clip=False)
                ax.annotate(r"$p_{% s} = %.3f$" % (label, p), [1.2, 0.6 - dy], transform=ax.transData,
                            annotation_clip=False)
                ax.annotate(r"$N_{% s} = %i$" % (label, n), [1.2, 0.4 - dy], transform=ax.transData, annotation_clip=False)

                dy += 1

        def plot_bill_scatter(condition: pd.Series, title: str, ax: plt.Axes):

            # Make sure condition index aligns with adj_matrix index
            condition = condition.reindex(adj_matrix.index).fillna(False)

            df_union = adj_matrix_union[condition]
            df_intersection = adj_matrix_intersection[condition]

            def _plot_bill_scatter_df(df, ax, c):
                ax.scatter(df[kwargs['x']], df[kwargs['y']],
                           s=kwargs['s'],
                           alpha=kwargs['alpha'],
                           c=df.passed.map(passed_colors),
                           linewidth=df.passed.map(passed_lws)
                           )

            _plot_bill_scatter_df(df_union, ax, 'grey')

            ax.set_ylabel(label_adjust(kwargs['y']))
            ax.set_title(title)
            ax.vlines(*vlines_kwargs, zorder=-100)
            ax.hlines(*vlines_kwargs, zorder=-100)
            annotate_stats(ax, df_union, df_intersection)

        condition = (topics_dummies['Renewable Energy Solar'] == 1)
        title = 'Solar'
        plot_bill_scatter(condition, title, axes[0])

        condition = (topics_dummies['Renewable Energy Wind'] == 1)
        title = 'Wind'
        plot_bill_scatter(condition, title, axes[1])

        axes[0].set_xlim(-1.1, 1.1)
        axes[0].set_ylim(-1.1, 1.1)

        axes[-1].set_xlabel(label_adjust(kwargs['x']))

        markers = [mpl.lines.Line2D([0], [0], ms=kwargs['s'] ** 0.5, mfc=passed_colors[False],
                                    marker='o', linewidth=0, mec='none', mew=passed_lws[False]),
                   mpl.lines.Line2D([0], [0], ms=kwargs['s'] ** 0.5, mfc=passed_colors[True],
                                    marker='o', linewidth=0, mec='none', mew=passed_lws[True])]
        axes[0].legend(markers, ['Failed', 'Passed'])

        fig.savefig('figures/figure_2.pdf', bbox_inches='tight')
        fig.savefig('figures/figure_2.png', bbox_inches='tight', dpi=300)
        plt.show()

    ###############################
    # Create correlation charts
    ###############################
    if do_correlations:
        selected_topics = [
            'Renewable Energy Wind',
            'Renewable Energy Solar',
            'Fossil Energy Coal',
            'Fossil Energy Natural Gas',
            'Nuclear Energy Facilities',
            'Energy Efficiency',
            'Emissions'
        ]

        n_comparisons = len(comparison_industries)

        def get_correlations(adj_matrix):
            correlations = defaultdict(dict)

            for comparison in comparison_industries:

                selected_industries = [comparison,
                                       'IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY']

                for topic in selected_topics:

                    df = adj_matrix[(topics_dummies.reindex(adj_matrix.index)[topic] == 1)][selected_industries]

                    if df.notna().min(1).sum() < 2:
                        correlations[topic][comparison] = None
                    else:
                        df = df.dropna(axis=0)
                        correlations[topic][comparison] = pearsonr(df[selected_industries].values[:, 0],
                                                                   df[selected_industries].values[:, 1])

            return correlations

        union_correlations = get_correlations(adj_matrix_union)
        intersection_correlations = get_correlations(adj_matrix_intersection)

        fig, axes = plt.subplots(1, 2, figsize=(8, 11), sharex=True, sharey=True)

        # Map each comparison to a color
        # use a color palette with at least 10 colors
        palette = sns.color_palette('tab10', n_colors=len(comparison_industries))
        comparison_colors = dict(zip(comparison_industries, palette))

        def plot_correlations(correlations, ax):
            """
            Plot the correlations
            :param correlations: dict of dicts of correlations
            :param ax: axis to plot on
            :return: ax
            """

            topic_y = 0 # y position of the topic
            for topic in correlations:

                topic_comparison_y = 0 # y position of the comparison within the topic

                for comparison in comparison_industries:
                    pearson_r_obj = correlations[topic][comparison]

                    if pearson_r_obj is not None:
                        # Plot the correlation with 95% confidence interval
                        y = topic_y + topic_comparison_y
                        xerr = abs(np.array([*pearson_r_obj.confidence_interval(0.95)]) - pearson_r_obj.statistic).T.reshape(
                            (2, 1))
                        ax.errorbar([pearson_r_obj.statistic], [y], yerr=None, xerr=xerr, marker='o', lw=2,
                                    color=comparison_colors[comparison])

                        # Plot the correlation with 99% confidence interval, thinner
                        xerr = abs(np.array([*pearson_r_obj.confidence_interval(0.99)]) - pearson_r_obj.statistic).T.reshape(
                            (2, 1))
                        ax.errorbar([pearson_r_obj.statistic], [y], yerr=None, xerr=xerr, marker='o', lw=1,
                                    color=comparison_colors[comparison])

                    topic_comparison_y += 0.8 / n_comparisons

                topic_y += 1

            ax.set_yticks(np.arange(topic_y) + 0.4)
            ax.set_yticklabels(correlations)

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()

            for y in np.arange(topic_y):
                for v in (y + np.linspace(0, 0.8 * (1 - 1 / n_comparisons), n_comparisons)):
                    ax.hlines(v, -10, 10, lw=0.5, color='grey', zorder=-1, alpha=0.5, linestyle='--')

            ax.vlines(0, -1, topic_y + 1, 'k', zorder=-1, lw=0.5)
            ax.set_ylim(*ylim)
            ax.set_xlim(*xlim)

            return ax

        ax = plot_correlations(union_correlations, axes[0])
        ax.set_xlabel('Correlation coefficient - union')
        ax.set_ylabel('Topic')

        ax = plot_correlations(intersection_correlations, axes[1])
        ax.set_xlabel('Correlation coefficient - intersection')
        ax.set_ylabel('Topic')

        ax.set_xlim(-1.1, 1.1)

        # Add legend
        markers = [mpl.lines.Line2D([0], [0],
                                    mfc=comparison_colors[comparison],
                                    color = comparison_colors[comparison],
                                    marker='o', linewidth=1, mec='none', mew=0)
                    for comparison in comparison_industries]

        labels = [adjust_label(comparison) for comparison in comparison_industries]
        ax.legend(markers[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        fig.savefig('figures/figure_3.pdf', bbox_inches='tight')
        fig.savefig('figures/figure_3.png', bbox_inches='tight', dpi=300)
        plt.show()


def plot_agree_probabilities(agree_probabilities, disagree_probabilities):
    """

    :param agree_probabilities:
    :param disagree_probabilities:
    :return:
    """

    # Take the geometric mean of the probabilities because they are highly skewed
    def geo_mean(iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    # Sort the industries by their (geometric) average probability of agreeing with
    # the pro-environmental policy groups
    most_friendly = abs(agree_probabilities).apply(geo_mean, 1).sort_values()[::-1][:10]
    most_unfriendly = abs(disagree_probabilities).apply(geo_mean, 1).sort_values()[::-1][:10]

    flip_sign = 0

    fig, axes = plt.subplots(2,1, figsize = (6.5,8), sharex=True)

    for table_data in [
        agree_probabilities.reindex(most_friendly.index.values).round(2),
        disagree_probabilities.reindex(most_unfriendly.index.values).round(2)
        ]:
        ax = axes[flip_sign]
        table_data.index = table_data.index.map(
            lambda x: x.split('_')[1].replace('_', ' ').title().replace('&', '\&').replace("'S", "'s"))

        cm = mpl.colors.LinearSegmentedColormap.from_list(
            "mycmap",
            [['white', 'yellowgreen'], ['white', 'crimson']][flip_sign])

        ax = sns.heatmap(
            abs(table_data) * 100,
            cmap=cm,
            vmin=0,
            vmax=20,
            ax=ax,
            square=True,
            linewidths=1,
            linecolor='white',
            annot=True, fmt=".0f",
            cbar_kws={'format': '%.0f', 'label': 'P(' + ['support', 'oppose'][flip_sign] + ') [\%]'}
        )

        ax.set_ylabel("")

        ax.set_title(f"One-way {['support', 'oppose'][flip_sign]} probability towards Pro-Environmental Policy\n")

        flip_sign = 1

    # axes[0].set_xticklabels(None)
    axes[0].get_xaxis().set_visible(False)
    ax.set_xlabel("\n\n\n\n\nState and Record Type")

    new_labels = []
    i = 0
    testimony_start = 0
    for t in ax.get_xticklabels():
        state, rectype = t.get_text().split('-')
        new_labels.append(state)
        if rectype == "testimony":
            testimony_start = i
        else:
            i += 1

    kwargs = dict(horizontalalignment='center', xycoords='data',
                  arrowprops={'arrowstyle': '|-|', 'color': 'k', 'mutation_scale': 1.5}, annotation_clip=False)

    ax.annotate("", [0, 11], [testimony_start, 11], **kwargs)
    ax.annotate("", [testimony_start, 11], [len(new_labels), 11], **kwargs)

    kwargs.update(arrowprops=None)
    ax.annotate("Lobbying", [testimony_start / 2, 12], **kwargs)
    ax.annotate("Testimony", [testimony_start + (len(new_labels) - testimony_start) / 2, 12], **kwargs)

    _ = ax.set_xticklabels(new_labels, rotation=0)

    fig.savefig('figures/figure_4.pdf', bbox_inches='tight')
    fig.savefig('figures/figure_4.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_partisanship_figure(plotdata):
    """

    :param plotdata:
    :return:
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 2.5), sharey=True)

    norm = plt.Normalize(plotdata['MiningPctGdp'].min(), plotdata['MiningPctGdp'].max())
    sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
    sm.set_array([])

    for ax, yval in zip(
            [ax1, ax2, ax3],
            ['EnviroAvgPos', 'OilGasAvgPos', 'ElcUtlAvgPos']):
        sns.scatterplot(x='AvgPartisanship',
                        y=yval,
                        data=plotdata,
                        hue='MiningPctGdp',
                        edgecolor='k',
                        markers={True: 'o', False: 'P'},
                        style='deregulated',
                        alpha=0.7,
                        ax=ax,
                        legend=False,
                        palette='magma_r'
                        )

    for ax in (ax1, ax2, ax3):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.vlines(0, -2, 2, 'grey', ':', zorder=-100)
        ax.hlines(0, -2, 2, 'grey', ':', zorder=-100)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.4, 0.4)

    ax1.set_ylabel("Avg. position on passed bills")
    ax1.set_xlabel("Leg. ideology")
    ax2.set_xlabel("Leg. ideology")
    ax3.set_xlabel("Leg. ideology")

    ax2.yaxis.set_visible(False)
    ax3.yaxis.set_visible(False)

    ax1.set_title("Pro-Environmental Policy", fontsize=10)
    ax2.set_title("Oil \& Gas/Mining", fontsize=10)
    ax3.set_title("Electric Utilities", fontsize=10)

    cbar = fig.colorbar(sm, ax=(ax1, ax2, ax3))
    cbar.ax.set_ylabel('Fossil Fuel \% of GDP')

    fig.savefig('figures/figure_5.pdf', bbox_inches='tight')
    fig.savefig('figures/figure_5.png', bbox_inches='tight', dpi=300)
    plt.show()


