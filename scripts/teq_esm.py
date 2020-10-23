import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import SelectionGeometry
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, FuncTickFormatter, OpenURL, TapTool
from bokeh.models import CustomJS, Label, LassoSelectTool, Legend, LegendItem
from bokeh.models import LogAxis, Range1d
from bokeh.models.widgets import Button
from bokeh.themes import Theme

from utils import csv_creation, get_update_time, log_axis_labels
from utils import deselect, reset

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# what order to plot things and what the legend labels will say
missions = ['Kepler Candidate', 'Kepler Confirmed', 'K2 Candidate',
            'K2 Confirmed', 'TESS Candidate', 'Other Confirmed',
            'TESS Confirmed']

# markers and colors in the same order as the missions above
markers = ['circle_cross', 'circle', 'square_cross', 'square',
           'inverted_triangle', 'diamond', 'triangle']
# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
colors = ['#228833', '#228833', '#ee6677', '#ee6677', '#ccbb44', '#aa3377',
          '#ccbb44']

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# what to display when hovering over a data point
TOOLTIPS = [
    ("Planet", "@planet"),
    # only give the decimal and sig figs if needed
    ("Insolation", "@insolation{0,0[.][00]} Earths"),
    ("Equilibrium Temp", "@teq{0,0[.][0]} K"),
    ("ESM", "@esm{0,0[.][00]}"),
    ("Radius", "@radius{0,0[.][00]} Earth; @jupradius{0,0[.][0000]} Jup"),
    ("Period", "@period{0,0[.][0000]} days"),
    ("Discovered by", "@discovery"),
    ("Status", "@status")
]

# Stefan-Boltzman constant
#(solar luminosity / AU^2 / K^4)
sigma = 3.29800E-12
# calculate the values we want
albedo = 0.3
# 4 = equal heat everywhere, 2 = only day side reradiates
redist = 2
rediststr = 'No Night-side Redistribution'

teq = ((10.**dfpl['st_log_lum']) * (1. - albedo) / 
       (redist * 4 * np.pi * sigma * (dfpl['semi_au']**2)))**0.25
dfpl['t_eq'] = teq

planck = 6.6261e-34
cc = 2.9979e8
boltz = 1.38065e-23

esmscale = 3.44

refwavs = [7.5, 15]
for refwav in refwavs:
    exp1 = np.exp((planck * cc)/(refwav * 0.000001 * boltz * dfpl['st_teff'])) - 1
    exp2 = np.exp((planck * cc)/(refwav * 0.000001 * boltz * 1.1 * dfpl['t_eq'])) - 1
    dfpl['dep'+str(refwav)] = dfpl['tran_depth_ppm'] * exp1 / exp2
    dfpl['esm'+str(refwav)] = esmscale * dfpl['dep'+str(refwav)] * (10.**(-0.2 * dfpl['Kmag']))

for ifig in np.arange(len(refwavs)):

    iesm = 'esm'+str(refwavs[ifig])
    
    # output files
    embedfile = f'_includes/teq_esm{str(refwavs[ifig])}_candidates_embed.html'
    fullfile = f'_includes/teq_esm{str(refwavs[ifig])}_candidates.html'

    # set up the full output file
    plotting.output_file(fullfile, title='Equilibrium Temp vs ESM Plot')

    # create the figure
    fig = plotting.figure(x_axis_type='linear', y_axis_type='log',
                          tooltips=TOOLTIPS, plot_height=700)

    # allow for something to happen when you click on data points
    fig.add_tools(TapTool())

    # need to store min and max radius values to create the second axis
    ymin = 1
    ymax = 1

    # save the output plots to rearrange them in the legend
    glyphs = []
    counts = []
    sources = []
    alphas = []
    legitems = []

    for ii, imiss in enumerate(missions):
        # select the appropriate set of planets for each mission
        # make the confirmed planets more opaque and bigger
        if imiss == 'Other Confirmed':
            good = ((~np.in1d(dfpl['facility'], ['Kepler', 'K2', 'TESS'])) &
                    np.isfinite(dfpl['t_eq']) & np.isfinite(dfpl[iesm]) &
                    (dfpl['disposition'] == 'Confirmed') & dfpl['flag_tran'] &
                    (dfpl['rade'] <= 2.) & (dfpl[iesm] >= 0.))
            alpha = 0.7
            size = 8
        else:
            fac = imiss.split()[0]
            disp = imiss.split()[1]
            good = ((dfpl['facility'] == fac) & (dfpl['disposition'] == disp) &
                    np.isfinite(dfpl['t_eq']) & np.isfinite(dfpl[iesm]) &
                    dfpl['flag_tran'] & (dfpl['rade'] <= 2.) & 
                    (dfpl[iesm] >= 0.))
            if disp == 'Confirmed':
                alpha = 0.7
                size = 7
            else:
                # candidates get these values
                alpha = 0.4
                size = 5
                # make the yellow darker
                if fac == 'TESS':
                    alpha = 0.6

        counts.append(f'{good.sum():,}')

        # what the hover tooltip draws its values from
        source = plotting.ColumnDataSource(data=dict(
                planet=dfpl['name'][good],
                insolation=dfpl['insol'][good],
                period=dfpl['period'][good],
                radius=dfpl['rade'][good],
                jupradius=dfpl['radj'][good],
                host=dfpl['hostname'][good],
                discovery=dfpl['facility'][good],
                status=dfpl['disposition'][good],
                url=dfpl['url'][good],
                teq=dfpl['t_eq'][good],
                esm=dfpl[iesm][good]
                ))
        print(imiss, ': ', good.sum())

        # plot the planets
        # nonselection stuff is needed to prevent planets in that category from
        # disappearing when you click on a data point ("select" it)
        glyph = fig.scatter('teq', 'esm', color=colors[ii],
                            source=source, size=size, alpha=alpha,
                            marker=markers[ii], nonselection_alpha=0.07,
                            selection_alpha=0.8, nonselection_color=colors[ii])
        glyphs.append(glyph)
        sources.append(source)
        alphas.append(alpha)
        # save the global min/max
        ymin = min(ymin, source.data['esm'].min())
        ymax = max(ymax, source.data['esm'].max())

        leg = LegendItem(label=imiss + f' ({counts[ii]})', renderers=[glyph])
        legitems.append(leg)

    # set up where to send people when they click on a planet
    url = "@url"
    taptool = fig.select(TapTool)
    taptool.callback = OpenURL(url=url)

    # figure out what the default axis limits are
    if ifig == 0:
        ydiff = np.log10(ymax) - np.log10(ymin)
        ystart = 10.**(np.log10(ymin) - 0.05*ydiff)
        yend = 10.**(np.log10(ymax) + 0.05*ydiff)
    else:
        ystart = 0
        yend = 3.6

    # jupiter/earth radius ratio
    radratio = 11.21

    # set up the second axis with the proper scaling
    """
    if ifig == 0:
        fig.extra_y_ranges = {"jup": Range1d(start=ystart/radratio,
                                             end=yend/radratio)}
        fig.add_layout(LogAxis(y_range_name="jup"), 'right')
    else:
        fig.y_range.start = ystart
        fig.y_range.end = yend
        fig.x_range.start = 40000
        fig.x_range.end = 0.1
    """
    if ymax < 100:
        fig.y_range.end = 110
    fig.y_range.start = 0.1

    # add the first y-axis's label and use our custom log formatting
    fig.yaxis.axis_label = 'ESM (' + str(refwavs[ifig]) + ' micron)'
    fig.yaxis.formatter = FuncTickFormatter(code=log_axis_labels())

    # add the x-axis's label and use our custom log formatting
    fig.xaxis.axis_label = 'Equilibrium Temp (K)'
    # fig.xaxis.formatter = FuncTickFormatter(code=log_axis_labels())
    # make high t_eqs on the left
    fig.x_range.flipped = True

    """
    # add the second y-axis's label
    if ifig == 0:
        fig.right[0].axis_label = 'Radius (Jupiter Radii)'
    """
    
    # which order to place the legend labels
    topleg = ['Kepler Confirmed', 'K2 Confirmed', 'TESS Confirmed']
    bottomleg = ['Kepler Candidate', 'K2 Candidate', 'TESS Candidate']
    vbottomleg = ['Other Confirmed']

    # set up all the legend objects
    items1 = [legitems[missions.index(ii)] for ii in topleg]
    items2 = [legitems[missions.index(ii)] for ii in bottomleg]
    items3 = [legitems[missions.index(ii)] for ii in vbottomleg]

    # create the two legends
    for ii in np.arange(3):
        if ii == 0:
            items = items3
        elif ii == 1:
            items = items2
        else:
            items = items1
        legend = Legend(items=items, location="center")

        if ii == 2:
            legend.title = 'Discovered by and Status: R < 2 Earth Radii'
            legend.spacing = 10
        else:
            legend.spacing = 11

        """
        if ifig == 0:
            legend.location = (-70, 5)
        else:
        """
        legend.location = (-50, 5)
        legend.label_text_align = 'left'
        legend.margin = 0

        fig.add_layout(legend, 'above')

    # overall figure title
    fig.title.text = 'Transiting Planets and Planet Candidates'

    # create the four lines of credit text in the two bottom corners
    label_opts1 = dict(
        x=-65, y=42,
        x_units='screen', y_units='screen'
    )

    label_opts2 = dict(
        x=-65, y=47,
        x_units='screen', y_units='screen'
    )

    label_opts3 = dict(
        x=632, y=79,
        x_units='screen', y_units='screen', text_align='right',
        text_font_size='9pt'
    )

    label_opts4 = dict(
        x=632, y=83,
        x_units='screen', y_units='screen', text_align='right',
        text_font_size='9pt'
    )
    
    label_opts5 = dict(
        x=310, y=85,
        x_units='screen', y_units='screen', text_align='center'
    )

    msg1 = 'By Exoplots'
    # when did the data last get updated
    modtimestr = get_update_time().strftime('%Y %b %d')
    msg3 = 'Data: NASA Exoplanet Archive'
    msg4 = 'and ExoFOP-TESS'
    msg5 = f'Albedo = {albedo}; {rediststr}'

    caption1 = Label(text=msg1, **label_opts1)
    caption2 = Label(text=modtimestr, **label_opts2)
    caption3 = Label(text=msg3, **label_opts3)
    caption4 = Label(text=msg4, **label_opts4)
    caption5 = Label(text=msg5, **label_opts5)
    
    fig.add_layout(caption1, 'below')
    fig.add_layout(caption2, 'below')
    fig.add_layout(caption3, 'below')
    fig.add_layout(caption4, 'below')
    fig.add_layout(caption5, 'below')
    
    # add the download button
    button = Button(label="Download CSV of Selected Data",
                    button_type="primary")
    # what is the header and what keys correspond to those columns for
    # output CSV files
    csvhead = '# Planet Name, Status, Period (days), Radius (Earths), ' \
              'Radius (Jupiters), Insolation (Earths), Discovered by, ' \
              'T_eq (K), ESM'
    keys = ['planet', 'status', 'period', 'radius', 'jupradius', 'insolation',
            'discovery', 'teq', 'esm']
    button.js_on_click(CustomJS(args=dict(sources=sources, keys=keys,
                                          header=csvhead), code=csv_creation))

    # select multiple points to download
    box = BoxSelectTool()
    fig.add_tools(box)
    lasso = LassoSelectTool()
    fig.add_tools(lasso)

    des = CustomJS(args=dict(glyphs=glyphs, alphas=alphas, legends=legitems),
                   code=deselect)
    fig.js_on_event(SelectionGeometry, des)
    fig.js_on_event('reset', CustomJS(args=dict(glyphs=glyphs, alphas=alphas,
                                                legends=legitems), code=reset))

    for iglyph in glyphs:
        iglyph.js_on_change('visible', des)

    layout = column(button, fig)

    plotting.save(layout)

    plotting.show(layout)

    # save the individual pieces so we can just embed the figure without the
    # whole html page
    script, div = components(layout, theme=theme)
    with open(embedfile, 'w') as ff:
        ff.write(script)
        ff.write(div)
