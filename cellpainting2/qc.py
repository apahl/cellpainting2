#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##
QC
##

*Created on Mon, Mar 26 by A. Pahl*

Tools for interactive plate quality control in the notebook.
Uses Holoviews and its Bokeh backend for visualization.
"""

import os.path as op

import pandas as pd
import numpy as np

import matplotlib as mpl
import holoviews as hv
hv.extension("bokeh")
from bokeh.models import HoverTool

from cellpainting2 import processing as cpp
from cellpainting2 import reporting as cpr
from cellpainting2 import tools as cpt
cp_config = cpt.load_config("config")

ACT_PROF_PARAMETERS = cp_config["Parameters"]
ACT_CUTOFF_PERC = cp_config["Cutoffs"]["ActCutoffPerc"]
ACT_CUTOFF_PERC_H = cp_config["Cutoffs"]["ActCutoffPercH"]
ACT_CUTOFF_PERC_REF = cp_config["Cutoffs"]["ActCutoffPercRef"]
ACT_CUTOFF_PERC_REF_H = cp_config["Cutoffs"]["ActCutoffPercRefH"]
OVERACT_H = cp_config["Cutoffs"]["OverActH"]
LIMIT_ACTIVITY_H = cp_config["Cutoffs"]["LimitActivityH"]
LIMIT_ACTIVITY_L = cp_config["Cutoffs"]["LimitActivityL"]
LIMIT_CELL_COUNT_H = cp_config["Cutoffs"]["LimitCellCountH"]
LIMIT_CELL_COUNT_L = cp_config["Cutoffs"]["LimitCellCountL"]
LIMIT_SIMILARITY_H = cp_config["Cutoffs"]["LimitSimilarityH"]
LIMIT_SIMILARITY_L = cp_config["Cutoffs"]["LimitSimilarityL"]


def add_images(df):
    """Adds an Image column to the MolFrame, used for structure tooltips in plotting.
    Only works on Pandas DataFrames, does not work for Dask DataFrames
    (call `.compute()` first)."""

    assert cpp.is_pandas(df), "Only works when the data object is a Pandas DataFrame. Consider running `.compute()` first."

    def _img_method(x):
        return "data:image/png;base64,{}".format(cpr.b64_mol(cpt.mol_from_smiles(x)))

    result = df.copy()
    result["Image"] = result["Smiles"].apply(_img_method)
    return result


def process_plate_for_qc(plate_full_name, structures=True):
    plate = cpt.split_plate_name(plate_full_name)
    src_templ = op.join(cp_config["Dirs"]["PlatesDir"], "{}-{}")
    src_dir = src_templ.format(plate.date, plate.name)
    #                                                   process as Pandas
    ds_plate = cpp.read_csv(op.join(src_dir, "Results.tsv")).compute()
    ds_plate = ds_plate.group_on_well()
    ds_plate.position_from_well()  # inplace
    if plate.name == "C2017-04":
        print("* skipping Echo filter step.")
    else:
        ds_plate = ds_plate.remove_skipped_echo_direct_transfer(op.join(src_dir, "*_print.xml"))
    ds_plate.well_type_from_position()

    ds_plate.flag_toxic()
    # print(sorted(set(ds_plate.keys())  - set(cpp.ACT_PROF_PARAMETERS))))
    ds_plate.data["Plate"] = "{}-{}".format(plate.date, plate.name)
    ds_plate.qc_stats()
    ds_plate = ds_plate.activity_profile(act_cutoff=1.585)

    ds_plate = ds_plate.keep_cols(cpp.FINAL_PARAMETERS)  # JUST FOR QC

    ds_plate = ds_plate.join_layout_1536(plate.name, keep_ctrls=True)
    ds_plate.data["Plate"] = "{}-{}".format(plate.date, plate.name)

    ds_plate = ds_plate.join_smiles()
    ds_plate = ds_plate.join_batch_data()
    if structures:
        data = add_images(ds_plate.data)
    else:
        data = ds_plate.data
        data["Image"] = "no struct"
    return data


def struct_hover():
    """Create a structure tooltip that can be used in Holoviews.
    Takes a MolFrame instance as parameter."""
    hover = HoverTool(
        tooltips="""
            <div>
                <div>
                    <img src="@Image" alt="Mol" width="70%"><br>
                <div>
                <div>
                    <span style="font-size: 12px; font-weight: bold;">@{}</span>
                    <span style="font-size: 12px;"> (@Producer)</span>
                </div>
                <div>
                    <span style="font-size: 12px;">Induction: @Activity</span>
                </div>
                <div>
                    <span style="font-size: 12px;">Rel_Cell_Count: @Rel_Cell_Count</span>
                </div>
            </div>
        """.format("Batch_Id")
    )
    return hover


def view_plate(plate, parm="Activity",
               cmap="gist_heat_r", low=0, high=50, show=True,
               title="Plate View"):
    if isinstance(plate, str):
        data = process_plate_for_qc(plate)
    else:
        data = plate.copy()  # already processed, Pandas DF

    id_prop = "Batch_Id"
    hover = struct_hover()
    plot_options = {
        "width": 800, "height": 450, "legend_position": "top_left",
        "tools": [hover], "invert_yaxis": True,
        "colorbar": True,
        "colorbar_opts": {"width": 10},
    }
    plot_styles = {"size": 20, "cmap": cmap}
    vdims = ["plateRow", id_prop, "Image", "Producer", "Activity", "Rel_Cell_Count"]
    if parm == "Activity" or parm == "Induction":
        plot_options["color_index"] = 5
    else:
        plot_options["color_index"] = 6
    opts = {'Scatter': {'plot': plot_options, "style": plot_styles}}
    scatter_plot = hv.Scatter(data, "plateColumn", vdims=vdims, label=title)
    range_args = {"plateRow": (0.5, 16.5), "plateColumn": (0.5, 24.5),
                  parm: (low, high)}
    scatter_plot = scatter_plot.redim.range(**range_args)
    # return data
    return scatter_plot(opts)


def ecdf(data, formal=False, x_range=None):
    """
    taken from the "DataFramed" podcast, episode 14 (https://www.datacamp.com/community/podcast/text-mining-nlproc)
    code: https://github.com/justinbois/dataframed-plot-examples/blob/master/ecdf.ipynb

    Get x, y, values of an ECDF for plotting.

    Parameters
    ----------
    data : ndarray
        One dimensional Numpay array with data.
    formal : bool, default False
        If True, generate x and y values for formal ECDF (staircase). If
        False, generate x and y values for ECDF as dots.
    x_range : 2-tuple, default None
        If not None and `formal` is True, then specifies range of plot
        on x-axis.

    Returns
    -------
    x : ndarray
        x-values for plot
    y : ndarray
        y-values for plot
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)

    if formal:
        # Set up output arrays
        x_formal = np.empty(2 * (len(x) + 1))
        y_formal = np.empty(2 * (len(x) + 1))

        # y-values for steps
        y_formal[:2] = 0
        y_formal[2::2] = y
        y_formal[3::2] = y

        # x- values for steps
        x_formal[0] = x[0]
        x_formal[1] = x[0]
        x_formal[2::2] = x
        x_formal[3:-1:2] = x[1:]
        x_formal[-1] = x[-1]

        if x_range is not None:
            if np.all(x >= x_range[0]) and np.all(x <= x_range[1]):
                x_formal = np.concatenate(((x_range[0],), x_formal, (x_range[1],)))
                y_formal = np.concatenate(((0,), y_formal, (1,)))
            else:
                raise RuntimeError('Some data values outside of `x_range`.')

        return x_formal, y_formal
    else:
        return x, y


def view_control_stats(full_plate_name):
    assert isinstance(full_plate_name, str), "`full_plate_name` has to be the full plate name"
    what_stats = ["Min_rel", "Max_rel", "MAD_rel"]
    qc_stats = cpp.read_resource("QCSTATS")
    plate_stats = qc_stats[qc_stats["Plate"] == full_plate_name]
    melt = plate_stats.copy()
    melt = melt.drop("Plate", axis=1)
    melt = pd.melt(melt, id_vars="Stat",
                   var_name="Parameter", value_name="Value")
    melt = melt.reset_index().drop("index", axis=1)
    title = "{} Controls Stats".format(full_plate_name)
    df_d = {"Stat": [], "x": [], "ECDF": []}
    for stat in what_stats:
        x, y = ecdf(melt.loc[(melt['Stat'] == stat), 'Value'])
        df_d["Stat"].extend([stat] * len(x))
        df_d["x"].extend(x)
        df_d["ECDF"].extend(y)
    data = pd.DataFrame(df_d)

    cmap_ecdf = mpl.colors.ListedColormap(colors=["#e5ae38", "#fc4f30", "#30a2da"])  # , "#55aa00"])
    plot_opts = dict(show_legend=True, width=350, height=350, toolbar='right',
                     color_index=2, legend_position="bottom_right")
    plot_styles = dict(size=5, cmap=cmap_ecdf)
    vdims = ["ECDF", "Stat"]
    ecdf_plot = hv.Scatter(
        data, "x", vdims=vdims, label=title,
    ).redim.range(x=(0, 1), ECDF=(0, 1.02)).redim.label(x='Deviation from Median')
    return ecdf_plot(plot=plot_opts, style=plot_styles)
