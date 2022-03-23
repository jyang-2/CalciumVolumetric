def draw_stim_lines(ax, stim_labels, stim_ict, lut=None,
                    text_kwargs=None,
                    line_kwargs=None):
    """ Draw vertical lines on axes, with stimulus str labels.

    To draw lines on bottom, set line_kwargs['zorder'] = 0.

    """
    xmin, xmax = ax.get_xlim()

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

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
    return ax
