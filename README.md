## Have Fun with Interactive Figure: Exoplots (maintained by ethankruse)
[All figures | Exoplots](https://ethankruse.github.io/exoplots/)<br>
The data used herein were obtained from the [NExSci Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/).<br>


## Update Data & Make Your Own Plots
1. Download scripts:<br>
    `git clone --depth 1 git@github.com:ruizhizhan/exoplots.git`<br>
or Download .zip.
2. Create python environment:<br>
    `conda env create -f environment.yml && conda activate exoplot`
3. Update data:<br>
    `python download-planet-data.py`<br>
ps. You might have to look into `scripts/util.py` and update special planet lists (e.g. `excluded`,`k2exclude`, `ignores`, ...).
4. Run scripts with up-to-date data: <br>
    `./update.sh`<br>
or `python scripts/mass_radius.py` to generate a single plot.
5. Make your own plots.<br>
(1) Read the data: <br>
`example.read_exoplot_data.ipynb`<br>
(2) Plot the data: <br>
`example.plot_exoplot_data.ipynb`