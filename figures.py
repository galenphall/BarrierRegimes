import re
import textwrap
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, linregress
from matplotlib.patches import Rectangle
from utils import adjust_label

# Convert to LaTeX-friendly font
mpl.rcParams['text.usetex'] = True


def plot_top_industries_figure(table_data):
    """
    Create Figure 1: Heatmap of top industries in each state
    :param positions: the positions dataframe, deduplicated and with civil servants removed
    :return: None
    """
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

    # Highlight top industry_id in each state
    for state, row in table_data.iterrows():

        if state == "":
            continue

        # Get the top industry_id for this state
        top_industry = row.idxmax()

        # Get the x and y coordinates of the top industry_id in the heatmap
        y = table_data.columns.get_loc(top_industry)
        x = table_data.index.get_loc(state)

        # Plot a black box around the top industry_id
        # Ensure the box is not cut off by bounds of the heatmap
        ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='k', lw=3, clip_on=False))

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
    cbar_ax.set_xlabel("Percent of positions from industry_id (incl. neutral)")
    cbar_ax.set_xticks([0, 5, 10, 15, 20, 25])
    cbar_ax.set_xticklabels([f"{i:.0f}%" for i in [0, 5, 10, 15, 20, 25]])
    [s.set_visible(True) for s in cbar_ax.spines.values()]

    fig.savefig('figures/figure_1.png', dpi=300, bbox_inches='tight')
    fig.savefig('figures/figure_1.pdf', bbox_inches='tight')
    plt.show()


def plot_topic_correlation_coefficients(
        comparison_industries,
        intersection_correlations,
        union_correlations,
        bill_counts,
        interest_group_counts,
):
    fig, axes = plt.subplots(1, 2, figsize=(8, 11), sharex=True, sharey=True)

    # Map each comparison to a color
    n_comparisons = len(comparison_industries)
    palette = sns.color_palette('tab10', n_colors=n_comparisons)
    comparison_colors = dict(zip(comparison_industries, palette))

    def plot_correlations(correlations, ax):
        """
        Plot the correlations
        :param correlations: dict of dicts of correlations
        :param ax: axis to plot on
        :return: ax
        """

        topic_y = 0  # y position of the topic
        for topic in correlations:

            topic_comparison_y = 0  # y position of the comparison within the topic

            for comparison in comparison_industries:
                pearson_r_obj = correlations[topic][comparison]

                if pearson_r_obj is not None:
                    # Plot the correlation with 95% confidence interval
                    y = topic_y + topic_comparison_y
                    xerr = abs(
                        np.array([*pearson_r_obj.confidence_interval(0.95)]) - pearson_r_obj.statistic).T.reshape(
                        (2, 1))
                    ax.errorbar([pearson_r_obj.statistic], [y], yerr=None, xerr=xerr, marker='o', lw=2,
                                color=comparison_colors[comparison])

                    # Plot the correlation with 99% confidence interval, thinner
                    xerr = abs(
                        np.array([*pearson_r_obj.confidence_interval(0.99)]) - pearson_r_obj.statistic).T.reshape(
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
    ax.set_xlim(-1.1, 1.1)

    # add bill counts to y labels
    yticklabels = [f"{label.get_text()} (N = {int(bill_counts[label.get_text()])})" for label in axes[0].get_yticklabels()]
    axes[0].set_yticklabels(yticklabels)
    axes[1].set_yticklabels(yticklabels)
    ax.set_yticklabels(yticklabels)

    # Add legend showing colors
    markers = [mpl.lines.Line2D([0], [0],
                                mfc=comparison_colors[comparison],
                                color=comparison_colors[comparison],
                                marker='o', linewidth=1, mec='none', mew=0)
               for comparison in comparison_industries]
    labels = [adjust_label(comparison) + f" (N = {interest_group_counts[comparison]})" for comparison in comparison_industries]

    markers = markers[::-1]
    labels = labels[::-1]
    # Add legend showing confidence intervals with a short line
    conf_line_kwargs = dict(color='k', marker='none', mec='none', mew=0)
    markers += [mpl.lines.Line2D([0, 5], [0, 0], **conf_line_kwargs, linewidth=2),
                mpl.lines.Line2D([0, 5], [0, 0], **conf_line_kwargs, linewidth=1),]
    labels += ['95% confidence interval', '99% confidence interval']

    axes[1].legend(markers, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False,
                   fontsize=12, handlelength=0.5, handletextpad=0.5, labelspacing=0.5, title='Comparison industry')

    fig.savefig('figures/figure_3.pdf', bbox_inches='tight')
    fig.savefig('figures/figure_3.png', bbox_inches='tight', dpi=300)
    plt.show()


def plot_topic_correlation_scatter(
        adj_matrix: pd.DataFrame,
        adj_matrix_union: pd.DataFrame,
        adj_matrix_intersection: pd.DataFrame,
        topics_dummies: pd.DataFrame):
    """
    Plots a scatter plot of the correlation between the pro-environmental policy and electric utilities positions
    on solar and wind energy bills.
    :param adj_matrix:
    :param adj_matrix_union:
    :param adj_matrix_intersection:
    :param topics_dummies:
    :return:
    """
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

            # Annotate the axis with the linear regression and correlation coefficient.
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
        """
        Plots a scatter plot of the correlation between the pro-environmental policy and electric utilities positions
        :param condition:
        :param title:
        :param ax:
        :return:
        """
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


def plot_agree_probabilities(agree_probabilities, disagree_probabilities):
    """
    Plot the average probability of (dis)agreeing with the pro-environmental policy groups for each industry
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
            vmax=15,
            ax=ax,
            square=True,
            linewidths=1,
            linecolor='white',
            annot=True, fmt=".1f",
            annot_kws={'fontsize': 6},
            cbar_kws={'format': '%.0f', 'label': 'P(' + ['support', 'oppose'][flip_sign] + ') [\%]'}
        )

        ax.set_ylabel("")
        # make y ticks invisible
        ax.tick_params(axis='y', which='both', length=0)

        ax.set_title(f"One-way {['support', 'oppose'][flip_sign]} probability towards Pro-Environmental Policy\n")

        flip_sign = 1

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
    Plot the partisanship figure, which shows the average position of each industry on passed bills in a given
    session against the partisanship of the state legislature, its mining share of GDP, and its deregulation
    status.
    :param plotdata: dataframe generated in structural_factors.py
    :return: None
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
        ax.set_ylim(-1.1, 1.1)

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


def plot_utilities_p_disagree_robustness_check(data, histdata):
    """
    Plots the observed vs. expected probability of disagreement (P(disagree)) for the pro-environmental policy groups
    from the utilities groups. The observed probabilities are plotted as points, and the expected probabilities are
    plotted as boxplots. The expected probabilities are calculated from the configuration models.
    :param data: the data frame with the observed probabilities
    :param histdata: the data frame with the simulated probabilities from configuration models
    :return: None
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[3, 1], sharey=True)
    sns.boxplot(histdata, y='state', x='alignment', order=data.state.drop_duplicates(), ax=ax)
    ax.plot(data.alignment, data.state, marker='o', lw=0, mfc='w', mec='grey')
    ax.set_xlim(0, 0.23)
    ax.set_xticklabels([round(i * 100, 1) for i in ax.get_xticks()])
    ax.set_xlabel("P(disagree)")
    ax.hlines(6.5, 0, .23, 'k', ':')
    ax.set_xlim(0, 0.23)
    ax.text(0.15, "MO", "Regulated", verticalalignment='bottom')
    ax.text(0.15, "MA", "Deregulated", verticalalignment='top')
    ax2.barh(width=data.alignment - data.expected, y=data.state, edgecolor='k', color='grey', height=0.5)
    ax2.vlines(data[data.deregulated].alignment.mean() - data[data.deregulated].expected.mean(), 7, 11, 'r', '--')
    ax2.vlines(data[~data.deregulated].alignment.mean() - data[~data.deregulated].expected.mean(), 0, 6, 'r', '--')
    ax2.hlines(6.5, 0, .13, 'k', ':')
    ax2.set_xticklabels([round(i * 100, 1) for i in ax2.get_xticks()])
    ax2.set_xlabel("observed - expected")
    fig.suptitle(
        "Probability of electric utilities opposing environmenal nonprofits:\n" +
        "configuration model versus observed\n", )
    fig.savefig("figures/figure_A1.pdf", bbox_inches='tight')
    fig.savefig("figures/figure_A1.png", dpi=300, bbox_inches='tight')


def plot_utilities_p_disagree_main(data):
    """
    Plot the probability that a randomly chosen utility company will oppose a randomly chosen environmental nonprofit
    position. The utilities are grouped by whether they are in a deregulated state or not.
    :param data: data frame with the probabilities, alignment, state, and deregulated status
    :return: None
    """
    fig, ax4 = plt.subplots(1, 1, figsize=(3, 3))
    sns.swarmplot(
        data,
        x='deregulated',
        y='alignment',
        edgecolor='none',
        color='none',
        size=1,
        ax=ax4,
        legend=False)
    annotations = [
        ax4.annotate(
            state, [dereg, alignment],
            verticalalignment='center',
            horizontalalignment='center')
        for state, alignment, dereg in data[['state', 'alignment', 'deregulated']].values
    ]
    ax4.set_ylabel("P(Disagree) (\%)")
    ax4.set_yticklabels([round(i * 100, 1) for i in ax4.get_yticks()])
    ax4.set_xlabel("")
    ax4.set_xticklabels(['Regulated', 'Deregulated'])
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_title("Electric Utilities and\nPro-Environmental Policy", fontsize=10)
    adjust_text(annotations, autoalign=False, ax=ax4, only_move={'text': 'x', 'points': 'x', 'objects': 'x'})
    xlim = ax4.get_xlim()
    for a in annotations:
        x, y = a.get_position()
        ax4.plot(x, y, marker='o', mfc='grey', mec='none', lw=0, alpha=0.3, zorder=-1, ms=17)
    ax4.set_xlim(*xlim)
    ax4.set_ylim(0, ax4.get_ylim()[1] + 0.02)
    fig.savefig("figures/figure_6.png", dpi=300, bbox_inches='tight')
    fig.savefig("figures/figure_6.pdf", bbox_inches='tight')

def plot_industry_to_industry_topic_scatters(adj_matrix, comparison_industries, comparison_topics, topics_dummies):
    """
    For each industry, and each topic, plot the average position of that industry against the average position of
    the IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY industry on each bill in that topic. The result is a matrix of
    scatter plots, where each row and column is an industry, and each cell is a scatter plot of the average position of
    each industry against the average position of the PRO-ENVIRONMENTAL POLICY groups on each bill in that topic.

    :param adj_matrix: the adjacency matrix of the bipartite network, where rows are bills and columns are industries
    :param comparison_industries: the industries to compare
    :param comparison_topics: the topics to compare
    :param topics_dummies: the topics dummies, used to select bills (rows) in adj_matrix
    :return:
    """
    fig, axes = plt.subplots(len(comparison_industries), len(comparison_topics), figsize=(10, 10),
                             sharex=True, sharey=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    target_industry = 'IDEOLOGY/SINGLE ISSUE_PRO-ENVIRONMENTAL POLICY'

    # font needs to be small to fit all the labels
    fontdict = {'fontsize': 5,
                'verticalalignment': 'center',
                'horizontalalignment': 'center',
                'rotation': 0}

    for i, industry1 in enumerate(comparison_industries):
        for j, topic in enumerate(comparison_topics):

            bills = adj_matrix.loc[topics_dummies[topic] == 1].index
            industry1_positions = adj_matrix.loc[bills, industry1]
            target_industry_positions = adj_matrix.loc[bills, target_industry]
            axes[i, j].scatter(industry1_positions, target_industry_positions, alpha=0.6,
                               edgecolor='none', color='grey', s=3)
            axes[i, j].set_xlim(-1.1, 1.1)
            axes[i, j].set_ylim(-1.1, 1.1)
            if i == 0:
                fontdict['horizontalalignment'] = 'center'
                fontdict['rotation'] = 0
                axes[i, j].set_title(topic, fontdict=fontdict)
            if j == 0:
                fontdict['horizontalalignment'] = 'right'
                # rotate the y label so that it reads left to right
                fontdict['rotation'] = 90
                axes[i, j].set_ylabel(adjust_label(industry1), fontdict=fontdict)


    # add Pro-Environmental Policy label at center bottom of figure
    fontdict.update({'fontsize': 12, 'rotation': 0, 'horizontalalignment': 'center', 'verticalalignment': 'center'})
    fig.text(0.5, 0.05, adjust_label(target_industry), fontdict=fontdict)

    fig.savefig("figures/figure_A2.png", dpi=300, bbox_inches='tight')
    fig.savefig("figures/figure_A2.pdf", bbox_inches='tight')

    plt.show()

