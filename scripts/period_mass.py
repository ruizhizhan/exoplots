import numpy as np
import pandas as pd
from bokeh import plotting
from bokeh.embed import components
from bokeh.events import SelectionGeometry, Tap
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import BoxSelectTool, FuncTickFormatter, TapTool
from bokeh.models import CustomJS, Label, LassoSelectTool, Legend, LegendItem
from bokeh.models import LogAxis, Range1d
from bokeh.models.widgets import Button
from bokeh.themes import Theme

from utils import csv_creation, get_update_time, log_axis_labels
from utils import deselect, openurl, palette, reset

# get the exoplot theme
theme = Theme(filename="./exoplots_theme.yaml")
curdoc().theme = theme

# what order to plot things and what the legend labels will say
methods = ['Transit', 'Radial Velocity', 'Timing Variations', 'Other']

# markers and colors in the same order as the missions above
markers = ['circle', 'square', 'triangle', 'diamond', 'inverted_triangle']
# colorblind friendly palette from https://personal.sron.nl/~pault/
# other ideas:
# https://thenode.biologists.com/data-visualization-with-flying-colors/research/
colors = [palette['C0'], palette['C1'], palette['C2'], palette['C3']]

# output files
embedfile = '_includes/period_mass_embed.html'
fullfile = '_includes/period_mass.html'

# set up the full output file
plotting.output_file(fullfile, title='Period Mass Plot')

# load the data
dfpl = pd.read_csv('data/exoplots_data.csv')

# what to display when hovering over a data point
TOOLTIPS = [
    ("Planet", "@planet"),
    # only give the decimal and sig figs if needed
    ("Period", "@period{0,0[.][0000]} days"),
    ("Mass", "@mass{0,0[.][00]} Earth; @jupmass{0,0[.][0000]} Jup"),
    ("Discovered via", "@method")
]

# create the figure
fig = plotting.figure(x_axis_type='log', y_axis_type='log', tooltips=TOOLTIPS,
                      plot_height=700, plot_width=750)

# need to store min and max radius values to create the second axis
ymin = 1
ymax = 1
# save the output plots to rearrange them in the legend
glyphs = []
counts = []
sources = []
alphas = []

for ii, imeth in enumerate(methods):
    # select the appropriate set of planets for each mission
    if imeth == 'Other':
        good = ((~np.in1d(dfpl['discoverymethod'], methods)) &
                (~dfpl['discoverymethod'].str.contains('Timing')) &
                np.isfinite(dfpl['masse']) & np.isfinite(dfpl['period']) &
                (dfpl['disposition'] == 'Confirmed'))
    elif imeth == 'Timing Variations':
        good = (dfpl['discoverymethod'].str.contains('Timing') &
                np.isfinite(dfpl['masse']) & np.isfinite(dfpl['period']) &
                (dfpl['disposition'] == 'Confirmed'))
    else:
        good = ((dfpl['discoverymethod'] == imeth) &
                np.isfinite(dfpl['masse']) & np.isfinite(dfpl['period']) &
                (dfpl['disposition'] == 'Confirmed'))

    # make the alpha of large groups lower so they don't dominate so much
    alpha = 1. - good.sum()/1000.
    alpha = max(0.2, alpha)

    # what the hover tooltip draws its values from
    source = plotting.ColumnDataSource(data=dict(
            planet=dfpl['name'][good],
            period=dfpl['period'][good],
            host=dfpl['hostname'][good],
            mass=dfpl['masse'][good],
            method=dfpl['discoverymethod'][good],
            jupmass=dfpl['massj'][good],
            url=dfpl['url'][good]
            ))
    print(imeth, ': ', good.sum())
    counts.append(f'{good.sum():,}')

    # plot the planets
    # nonselection stuff is needed to prevent planets in that category from
    # disappearing when you click on a data point ("select" it)
    glyph = fig.scatter('period', 'mass', color=colors[ii], source=source,
                        size=8, alpha=alpha, marker=markers[ii],
                        nonselection_alpha=0.07, selection_alpha=0.8,
                        nonselection_color=colors[ii])
    glyphs.append(glyph)
    # save the global min/max
    ymin = min(ymin, source.data['mass'].min())
    ymax = max(ymax, source.data['mass'].max())
    sources.append(source)
    alphas.append(alpha)

# allow for something to happen when you click on data points
fig.add_tools(TapTool())
# set up where to send people when they click on a planet
uclick = CustomJS(args=dict(sources=sources), code=openurl)
fig.js_on_event(Tap, uclick)

# figure out what the default axis limits are
ydiff = np.log10(ymax) - np.log10(ymin)
ystart = 10.**(np.log10(ymin) - 0.05*ydiff)
yend = 10.**(np.log10(ymax) + 0.05*ydiff)

# jupiter/earth mass ratio
massratio = 317.8

# set up the second axis with the proper scaling
fig.extra_y_ranges = {"jup": Range1d(start=ystart/massratio,
                                     end=yend/massratio)}
fig.add_layout(LogAxis(y_range_name="jup"), 'right')

# add the first y-axis's label and use our custom log formatting for both axes
fig.yaxis.axis_label = 'Mass (Earth Masses)'
fig.yaxis.formatter = FuncTickFormatter(code=log_axis_labels())

# add the x-axis's label and use our custom log formatting
fig.xaxis.axis_label = 'Period (days)'
fig.xaxis.formatter = FuncTickFormatter(code=log_axis_labels())

# add the second y-axis's label
fig.right[0].axis_label = 'Mass (Jupiter Masses)'

# set up all the legend objects
items = [LegendItem(label=ii + f' ({counts[methods.index(ii)]})',
                    renderers=[jj])
         for ii, jj in zip(methods, glyphs)]
# create the legend
legend = Legend(items=items, location="center")
legend.title = 'Discovered via'
legend.spacing = 10
legend.margin = 8
fig.add_layout(legend, 'above')

# overall figure title
fig.title.text = 'Confirmed Planets'

# create the three lines of credit text in the two bottom corners
label_opts1 = dict(
    x=-84, y=32,
    x_units='screen', y_units='screen'
)

label_opts2 = dict(
    x=-84, y=37,
    x_units='screen', y_units='screen'
)

label_opts3 = dict(
    x=612, y=64,
    x_units='screen', y_units='screen', text_align='right',
    text_font_size='9pt'
)

msg1 = 'By Exoplots'
# when did the data last get updated
modtimestr = get_update_time().strftime('%Y %b %d')
msg3 = 'Data: NASA Exoplanet Archive'

caption1 = Label(text=msg1, **label_opts1)
caption2 = Label(text=modtimestr, **label_opts2)
caption3 = Label(text=msg3, **label_opts3)

fig.add_layout(caption1, 'below')
fig.add_layout(caption2, 'below')
fig.add_layout(caption3, 'below')

# add the download button
button = Button(label="Download CSV of Selected Data", button_type="primary")
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

des = CustomJS(args=dict(glyphs=glyphs, alphas=alphas, legends=items),
               code=deselect)
fig.js_on_event(SelectionGeometry, des)
fig.js_on_event('reset', CustomJS(args=dict(glyphs=glyphs, alphas=alphas,
                                            legends=items), code=reset))

for iglyph in glyphs:
    iglyph.js_on_change('visible', des)

layout = column(button, fig)

plotting.save(layout)

plotting.show(layout)

# save the individual pieces so we can just embed the figure without the whole
# html page
script, div = components(layout, theme=theme)
with open(embedfile, 'w') as ff:
    ff.write(script)
    ff.write(div)
