""" Utility functions for plotting"""
import json

import matplotlib.pyplot as plt

with open("/local/storage/Remy/natural_mixtures/reports/configs/stim_grid_all.json", 'r') as f:
    stim_grid_all = json.load(f)


def get_stim_grid_subplots(movie_type):
    fig_mosaic = stim_grid_all[movie_type]

    if movie_type in ['kiwi_ea_eb_only', 'control1_top2_ramps']:
        hspace = 0.5
    else:
        hspace = 0.4
    fig, ax_dict = plt.subplot_mosaic(fig_mosaic,
                                      #sharex='all', sharey='all',
                                      # subplot_kw=dict(sharex='all',
                                      #                 sharey='all',
                                      #                 ),
                                      gridspec_kw=dict(
                                          top=0.9,
                                          # height_ratios=hratio,
                                          wspace=0.2,
                                          hspace=0.2
                                      ),
                                      figsize=(11, 8.5),
                                      )
    return fig, ax_dict


def draw_stim_lines(ax, stim_labels, stim_ict, lut=None,
                    text_kwargs=None,
                    line_kwargs=None,
                    fraction_yrange=1):
    """ Draw vertical lines on axes, with stimulus str labels.

    To draw lines on bottom, set line_kwargs['zorder'] = 0.

    """
    xmin, xmax = ax.get_xlim()

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    if fraction_yrange != 1:
        ymax = ymin + fraction_yrange * yrange

    line_kwargs0 = dict(linestyle='-', linewidth=2, alpha=0.5)
    text_kwargs0 = dict(fontsize=8, rotation='vertical', va='top', ha='right')

    if text_kwargs is not None:
        text_kwargs0.update(text_kwargs)

    if line_kwargs is not None:
        line_kwargs0.update(line_kwargs)

    for ict, stim in zip(stim_ict, stim_labels):
        if (lut is not None) and (stim in lut.keys()):
            ax.axvline(ict, color=lut[stim], **line_kwargs0)
        else:
            ax.axvline(ict, **line_kwargs0)
        ax.text(ict, ymax, stim, **text_kwargs0)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    return ax


if __name__ == '__main__':
    fig1, axd = get_stim_grid_subplots('kiwi_ea_eb_only')
    plt.show()
