from datetime import datetime

import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import DoubleTap, SelectionGeometry
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import BoxSelectTool, FuncTickFormatter, TapTool, Toggle
from bokeh.models import Button, LogAxis, OpenURL, Range1d, RangeSlider
from bokeh.models import CustomJS, Label, LassoSelectTool, Legend, LegendItem
from bokeh.themes import Theme

from utils import csv_creation, get_update_time, log_axis_labels
from utils import deselect, nowvis, reset, sliderselect, unselect, yearselect
from utils import playpause

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# what order to plot things and what the legend labels will say
discovery = ['Transit', 'Radial Velocity', 'Other']

# markers and colors in the same order as the missions above
markers = ['circle', 'square', 'triangle']

# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
colors = ['#228833', '#ee6677', '#ccbb44', '#aa3377', '#ccbb44']

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

for ifig in np.arange(4):
    if ifig == 0:
        # output files
        embedfile = '_includes/period_radius_all_confirmed_embed.html'
        fullfile = '_includes/period_radius_all_confirmed.html'
    elif ifig == 1:
        embedfile = '_includes/period_radius_all_confirmed_year_embed.html'
        fullfile = '_includes/period_radius_all_confirmed_year.html'
    elif ifig == 2:
        # output files
        embedfile = '_includes/period_mass_all_confirmed_embed.html'
        fullfile = '_includes/period_mass_all_confirmed.html'
    elif ifig == 3:
        embedfile = '_includes/period_mass_all_confirmed_year_embed.html'
        fullfile = '_includes/period_mass_all_confirmed_year.html'

    # what to display when hovering over a data point
    TOOLTIPS = [
        ("Planet", "@planet"),
        # only give the decimal and sig figs if needed
        ("Period", "@period{0,0[.][0000]} days"),
        ("Measured Radius",
         "@radius{0,0[.][00]} Earth; @jupradius{0,0[.][0000]} Jup"),
        ("Estimated Radius",
         "@radius_est{0,0[.][00]} Earth; @jupradius_est{0,0[.][0000]} Jup"),
        ("Measured Mass",
         "@mass{0,0[.][00]} Earth; @jupmass{0,0[.][0000]} Jup"),
        ("Estimated Mass",
         "@mass_est{0,0[.][00]} Earth; @jupmass_est{0,0[.][0000]} Jup"),
        ("Discovered by", "@method"),
        ("Confirmed in", "@year")
    ]

    # set up the full output file
    plotting.output_file(fullfile, title='Period Radius Mass Plot')

    # create the figure
    fig = plotting.figure(x_axis_type='log', y_axis_type='log',
                          plot_height=700, tooltips=TOOLTIPS)

    # need to store min and max radius values to create the second axis
    ymin = 1
    ymax = 1
    xmin = 1
    xmax = 1
    # save the output plots to rearrange them in the legend
    glyphs = []
    counts = []
    sources = []
    alphas = []
    legitems = []
    minyr = 2020

    for ii, idisc in enumerate(discovery):
        if ifig < 2:
            if idisc == 'Other':
                good = ((~np.in1d(dfpl['discoverymethod'], discovery[:-1])) & 
                        (np.isfinite(dfpl['rade']) | np.isfinite(dfpl['rade_est'])) &
                        np.isfinite(dfpl['period']) &
                        (dfpl['disposition'] == "Confirmed"))
            else:
                # select the appropriate set of planets for each method
                good = ((dfpl['discoverymethod'] == idisc) & 
                        (np.isfinite(dfpl['rade']) | np.isfinite(dfpl['rade_est'])) &
                        np.isfinite(dfpl['period']) &
                        (dfpl['disposition'] == "Confirmed"))
        else:
            if idisc == 'Other':
                good = ((~np.in1d(dfpl['discoverymethod'], discovery[:-1])) & 
                        (np.isfinite(dfpl['masse']) | np.isfinite(dfpl['masse_est'])) &
                        np.isfinite(dfpl['period']) &
                        (dfpl['disposition'] == "Confirmed"))
            else:
                # select the appropriate set of planets for each method
                good = ((dfpl['discoverymethod'] == idisc) & 
                        (np.isfinite(dfpl['masse']) | np.isfinite(dfpl['masse_est'])) &
                        np.isfinite(dfpl['period']) &
                        (dfpl['disposition'] == "Confirmed"))
        alpha = 0.7
        size = 6

        counts.append(f'{good.sum():,}')

        # what the hover tooltip draws its values from
        source = plotting.ColumnDataSource(data=dict(
                planet=dfpl['name'][good],
                period=dfpl['period'][good],
                radius_plot=np.nanmin(dfpl.loc[good, ['rade', 'rade_est']], axis=1),
                radius=dfpl['rade'][good],
                radius_est=dfpl['rade_est'][good],
                jupradius=dfpl['radj'][good],
                jupradius_est=dfpl['radj_est'][good],
                mass_plot=np.nanmin(dfpl.loc[good, ['masse', 'masse_est']], axis=1),
                mass=dfpl['masse'][good],
                mass_est=dfpl['masse_est'][good],
                jupmass=dfpl['massj'][good],
                jupmass_est=dfpl['massj_est'][good],
                host=dfpl['hostname'][good],
                method=dfpl['discoverymethod'][good],
                url=dfpl['url'][good],
                year=dfpl['year_confirmed'][good]
                ))

        print(idisc, ': ', good.sum())

        # plot the planets
        # nonselection stuff is needed to prevent planets in that category from
        # disappearing when you click on a data point ("select" it)
        # selection_alpha is just requiring it to make its own selection glyph
        # so we can better control that later
        if ifig == 0:
            glyph = fig.scatter('period', 'radius_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.07,
                                selection_alpha=0.8,
                                nonselection_color=colors[ii])
        elif ifig == 1:
            glyph = fig.scatter('period', 'radius_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.,
                                selection_alpha=alpha,
                                nonselection_color=colors[ii])
        elif ifig == 2:
            glyph = fig.scatter('period', 'mass_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.07,
                                selection_alpha=0.8,
                                nonselection_color=colors[ii])
        elif ifig == 3:
            glyph = fig.scatter('period', 'mass_plot', color=colors[ii],
                                source=source, size=size, alpha=alpha,
                                marker=markers[ii], nonselection_alpha=0.,
                                selection_alpha=alpha,
                                nonselection_color=colors[ii])
        glyphs.append(glyph)
        sources.append(source)
        alphas.append(alpha)
        # save the global min/max
        if ifig < 2:
            ymin = min(ymin, source.data['radius_plot'].min())
            ymax = max(ymax, source.data['radius_plot'].max())
        else:
            ymin = min(ymin, source.data['mass_plot'].min())
            ymax = max(ymax, source.data['mass_plot'].max())
        ab0 = source.data['period'][source.data['period'] > 0]
        xmin = min(xmin, ab0.min())
        xmax = max(xmax, source.data['period'].max())

        leg = LegendItem(label=idisc + f' ({counts[ii]})', renderers=[glyph])
        legitems.append(leg)
        minyr = int(min(minyr, source.data['year'].min()))

    # figure out what the default axis limits are
    ydiff = np.log10(ymax) - np.log10(ymin)
    ystart = 10.**(np.log10(ymin) - 0.05*ydiff)
    yend = 10.**(np.log10(ymax) + 0.05*ydiff)

    xdiff = np.log10(xmax) - np.log10(xmin)
    xstart = 10.**(np.log10(xmin) - 0.05*xdiff)
    xend = 10.**(np.log10(xmax) + 0.05*xdiff)

    fig.x_range = Range1d(start=xstart, end=xend)

    # jupiter/earth radius ratio
    radratio = 11.21
    massratio = 317.83

    # set up the second axis with the proper scaling
    if ifig < 2:
        fig.extra_y_ranges = {"jup": Range1d(start=ystart/radratio,
                                             end=yend/radratio)}
    else:
        fig.extra_y_ranges = {"jup": Range1d(start=ystart/massratio,
                                             end=yend/massratio)}
    fig.add_layout(LogAxis(y_range_name="jup"), 'right')

    # add the first y-axis's label and use our custom log formatting for both
    if ifig < 2:
        fig.yaxis.axis_label = 'Radius (Earth Radii)'
    else:
        fig.yaxis.axis_label = 'Mass (Earth Masses)'
    fig.yaxis.formatter = FuncTickFormatter(code=log_axis_labels())

    # add the x-axis's label and use our custom log formatting
    fig.xaxis.axis_label = 'Period (days)'
    fig.xaxis.formatter = FuncTickFormatter(code=log_axis_labels())

    # add the second y-axis's label
    if ifig < 2:
        fig.right[0].axis_label = 'Radius (Jupiter Radii)'
    else:
        fig.right[0].axis_label = 'Mass (Jupiter Masses)'

    # which order to place the legend labels
    topleg = discovery

    # set up all the legend objects
    items1 = [legitems[discovery.index(ii)] for ii in topleg]

    # create the legend
    legend = Legend(items=items1, location="center")
    legend.title = 'Discovered via'
    legend.spacing = 11

    legend.location = (0, 5)
    legend.label_text_align = 'left'
    legend.margin = 0

    fig.add_layout(legend, 'above')

    # overall figure title
    fig.title.text = 'All Confirmed Planets'

    # create the four lines of credit text in the two bottom corners
    if ifig < 2:
        label_opts1 = dict(x=-55, y=42, x_units='screen', y_units='screen')
        label_opts2 = dict(x=-55, y=47, x_units='screen', y_units='screen')
        label_opts3 = dict(x=642, y=79, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')
        label_opts4 = dict(x=642, y=83, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')
    else:
        label_opts1 = dict(x=-85, y=42, x_units='screen', y_units='screen')
        label_opts2 = dict(x=-85, y=47, x_units='screen', y_units='screen')
        label_opts3 = dict(x=612, y=79, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')
        label_opts4 = dict(x=612, y=83, x_units='screen', y_units='screen',
                           text_align='right', text_font_size='9pt')

    msg1 = 'By Exoplots'
    # when did the data last get updated
    modtimestr = get_update_time().strftime('%Y %b %d')
    msg3 = 'Data: NASA Exoplanet Archive'
    #msg4 = 'and ExoFOP-TESS'

    caption1 = Label(text=msg1, **label_opts1)
    caption2 = Label(text=modtimestr, **label_opts2)
    caption3 = Label(text=msg3, **label_opts3)
    #caption4 = Label(text=msg4, **label_opts4)

    fig.add_layout(caption1, 'below')
    fig.add_layout(caption2, 'below')
    fig.add_layout(caption3, 'below')
    #fig.add_layout(caption4, 'below')

    # add the download button
    button = Button(label="Download CSV of Selected Data",
                    button_type="primary")
    # what is the header and what keys correspond to those columns for
    # output CSV files
    keys = source.column_names
    keys.remove('url')
    keys.remove('host')
    csvhead = '# ' + ', '.join(keys)
    button.js_on_click(CustomJS(args=dict(sources=sources, keys=keys,
                                          header=csvhead), code=csv_creation))

    # select multiple points to download
    box = BoxSelectTool()
    fig.add_tools(box)
    lasso = LassoSelectTool()
    fig.add_tools(lasso)

    # the problem with the Tap Tool is that it clears all selections on every
    # click, whether you click on a data point or not
    # allow for something to happen when you click on data points
    fig.add_tools(TapTool())
    # set up where to send people when they click on a planet
    url = "@url"
    taptool = fig.select(TapTool)
    taptool.callback = OpenURL(url=url)

    if ifig in [0, 2]:
        jargs = dict(glyphs=glyphs, alphas=alphas, legends=legitems)
        fig.js_on_event('reset', CustomJS(args=jargs, code=reset))
        des = CustomJS(args=jargs, code=deselect)
        fig.js_on_event(SelectionGeometry, des)
        uns = CustomJS(args=jargs, code=unselect)
        fig.js_on_event(DoubleTap, uns)
        for iglyph in glyphs:
            iglyph.js_on_change('visible', des)
        layout = column(button, fig)
    else:
        if ifig == 1:
            yrlabelopts = dict(x=545, y=455, x_units='screen', y_units='screen',
                               text_align='right', text_font_size='20pt',
                               text_baseline='top')
        else:
            yrlabelopts = dict(x=505, y=455, x_units='screen', y_units='screen',
                               text_align='right', text_font_size='20pt',
                               text_baseline='top')
        curyr = datetime.now().year
        yrtxt = f'{minyr}\u2013{curyr}'
        yrcap = Label(text=yrtxt, **yrlabelopts)
        fig.add_layout(yrcap)
        range_slider = RangeSlider(start=minyr, end=curyr, value=(minyr, curyr),
                                   step=1, title="Year Discovered", width=600,
                                   width_policy='fit')

        jargs = dict(glyphs=glyphs, alphas=alphas, legends=legitems,
                     slider=range_slider, label=yrcap)
        yrsel = CustomJS(args=jargs, code=yearselect)
        range_slider.js_on_change('value', yrsel)
        # even on a reset, reselect only the correct values rather than letting
        # everything get unselected
        rescode = "slider.value = [slider.start, slider.end];"
        fig.js_on_event('reset', CustomJS(args=jargs, code=rescode))
        sls = CustomJS(args=jargs, code=sliderselect)
        fig.js_on_event(SelectionGeometry, sls)
        fig.js_on_event(DoubleTap, yrsel)
        for iglyph in glyphs:
            nw = CustomJS(args=jargs, code=nowvis)
            iglyph.js_on_change('visible', nw)

        playbutton = Toggle(label="\u25b6 Play", width=200, width_policy='fit',
                            button_type='success')
        playbutton.js_on_change('active', CustomJS(args=jargs, code=playpause))

        anirow = row(range_slider, playbutton)

        layout = column(button, anirow, fig)

    plotting.save(layout)

    plotting.show(layout)

    # save the individual pieces so we can just embed the figure without the
    # whole html page
    script, div = components(layout, theme=theme)
    with open(embedfile, 'w') as ff:
        ff.write(script)
        ff.write(div)
