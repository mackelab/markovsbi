import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import os
import math
import numpy as np

from markovsbi.bm.data_utils import query

_custom_styles = ["pyloric"]
_tueplot_styles = ["aistats2022", "icml2022", "jmlr2001", "neurips2021", "neurips2022"]
_mpl_styles = list(plt.style.available)

PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COLORS = {
    "nle": "#1e81b0",
    "nre": "#76b5c5",
    "diffusion": "#f2a900",
    "npe": "#1f77b4",
}


def plot_metric_by_eval_steps(
    name,
    method=None,
    task=None,
    num_simulations=None,
    num_steps=None,
    sampler=None,
    seed=None,
    metric="c2st",
    eval_num_steps=None,
    ax=None,
    figsize=(1, 1),
    color_map=None,
    hue=None,
    df=None,
    **kwargs,
):
    if df is None:
        df = query(
            name=name,
            method=method,
            task=task,
            num_simulations=num_simulations,
            num_steps=num_steps,
            seed=seed,
            metric=metric,
            sampler=sampler,
            eval_num_steps=eval_num_steps,
        )

    df["eval_num_steps"] = df["eval_num_steps"] - 1
    ylims = get_ylim_by_metric(metric)
    df = df.sort_values("eval_num_steps")
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    else:
        fig = None
    if hue is not None:
        df = df.sort_values(hue, key=get_sorting_key_fn(hue))

    sns.pointplot(
        x="eval_num_steps",
        y="value",
        data=df,
        ax=ax,
        hue=hue,
        marker=".",
        dodge=False,
        palette=color_map,
        alpha=kwargs.get("alpha", 0.7),
        markersize=kwargs.get("markersize", 2.0),
        lw=kwargs.get("lw", 2.0),
    )
    ax.set_xlabel("Transitions")
    ax.set_ylabel(get_metric_plot_name(metric))
    ax.set_ylim(*ylims)
    return fig, ax


def get_style(style, **kwargs):
    if style in _mpl_styles:
        return [style]
    elif style in _tueplot_styles:
        return [getattr(bundles, style)(**kwargs)]
    elif style in _custom_styles:
        return [PATH + os.sep + style + ".mplstyle"]
    elif style == "science":
        return ["science"]
    elif style == "science_grid":
        return ["science", {"axes.grid": True}]
    elif style is None:
        return None
    elif style == "icml_science_grid":
        return [getattr(bundles, "icml2022")(**kwargs), "science", {"axes.grid": True}]
    else:
        return style


class use_style:
    def __init__(self, style, kwargs={}) -> None:
        super().__init__()
        self.style = get_style(style) + [kwargs]
        self.previous_style = {}

    def __enter__(self):
        self.previous_style = mpl.rcParams.copy()
        if self.style is not None:
            plt.style.use(self.style)

    def __exit__(self, *args, **kwargs):
        mpl.rcParams.update(self.previous_style)


def get_ylim_by_metric(metric):
    """Get ylim by metric"""
    if "c2st" in metric:
        return (0.5, 1.0)
    else:
        return None


def get_metric_plot_name(metric):
    """Get metric plot name"""
    if "c2st" in metric:
        return "C2ST"
    elif "nll" in metric:
        return "NLL"
    elif "swd" in metric:
        return r"s$W_1$"
    else:
        return metric


def get_task_plot_name(task):
    """Get task plot name"""
    if "simple" in task:
        parts = task.split("simple")
        d = int(parts[1].split("d")[0])
        return f"Gaussian RW ({d}d)"
    elif task == "lotka_volterra":
        return "Lotka Volterra"
    elif task == "sir":
        return "SIR"
    elif task == "general_sde":
        return "Linear SDE"
    elif task == "periodic_sde":
        return "Periodic SDE"
    elif task == "mixture_rw_5d":
        return "Mixture RW (5d)"
    elif task == "mixture_rw_2d":
        return "Mixture RW (2d)"
    elif task == "double_well":
        return "Double well"
    else:
        return task


def get_method_plot_name(method):
    if method == "nle":
        return "FNLE"
    elif method == "npe":
        return "NPE"
    elif method == "nre":
        return "FNRE"
    elif method == "diffusion":
        return "FNSE"
    else:
        return method


def get_sampler_plot_name(sampler):
    if "fnpe" in sampler:
        return "FNPE"
    elif "gaus" in sampler:
        return "GAUSS"
    elif "jac" in sampler:
        return "JAC"
    else:
        return sampler


def float_to_power_of_ten(val: float):
    exp = math.log10(val)
    exp = int(exp)
    return rf"$10^{exp}$"


def get_sorting_key_fn(name):
    if name == "method":

        def key_fn(method):
            if method == "npe":
                return 0
            elif method == "nle":
                return 1
            elif method == "nre":
                return 2
            elif "diffusion" in method:
                return 3
            else:
                return 4

        return np.vectorize(key_fn)
    elif name == "task":

        def key_fn(task):
            if "simple1d" in task:
                return 0
            elif "simple2d" in task:
                return 1
            elif "simple10d" in task:
                return 2
            elif task == "general_sde":
                return 4
            elif task == "periodic_sde":
                return 3
            elif task == "mixture_rw_5d":
                return 5
            elif task == "double_well":
                return 6
            elif task == "gaussian_mixture" or "marcov" in task:
                return 1
            elif task == "two_moons" or task == "two_moons_all_cond":
                return 2
            elif task == "slcp" or task == "two_moons_all_cond":
                return 3
            else:
                return 4

        return np.vectorize(key_fn)
    else:
        return lambda x: x


def get_plot_name_fn(name):
    """Get plot name fn"""

    if name == "task":
        return get_task_plot_name
    elif name == "metric":
        return get_metric_plot_name
    else:
        return lambda x: x


def use_all_plot_name_fn(name):
    """Get plot name fn"""

    return get_sampler_plot_name(
        get_method_plot_name(get_task_plot_name(get_metric_plot_name(name)))
    )


def multi_plot(
    name,
    cols,
    rows,
    plot_fn,
    fig_title=None,
    y_label_by_row=True,
    y_labels=None,
    scilimit=3,
    x_labels=None,
    y_lims=None,
    fontsize_title=None,
    figsize_per_row=2,
    figsize_per_col=2.3,
    legend_bbox_to_anchor=[0.5, -0.1],
    legend_title=False,
    legend_ncol=10,
    legend_kwargs={},
    fig_legend=True,
    df=None,
    verbose=False,
    **kwargs,
):
    if df is None:
        df = query(name, **kwargs)
    else:
        df = df.copy()

    df = df.sort_values(cols, na_position="first", key=get_sorting_key_fn(cols))
    cols_vals = df[cols].dropna().unique()

    df = df.sort_values(rows, na_position="first", key=get_sorting_key_fn(rows))
    rows_vals = df[rows].dropna().unique()

    # Creating a color map if hue is specified:
    if "hue" in kwargs and "color_map" not in kwargs:
        hue_col = kwargs["hue"]
        df = df.sort_values(
            hue_col, na_position="first", key=get_sorting_key_fn(hue_col)
        )
        unique_vals = df[hue_col].unique()
        unique_vals.sort()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {}
        for i in range(len(unique_vals)):
            color_map[unique_vals[i]] = colors[min(i, len(colors) - 1)]
    else:
        if "color_map" not in kwargs:
            color_map = None
        else:
            color_map = kwargs.pop("color_map")

    n_cols = len(cols_vals)
    n_rows = len(rows_vals)

    if n_cols == 0:
        raise ValueError(f"No columns found in the dataset with label {cols}")

    if n_rows == 0:
        raise ValueError(f"No rows found in the dataset with label {rows}")

    print(figsize_per_col)

    figsize = (n_cols * figsize_per_col, n_rows * figsize_per_row)

    print(figsize)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    else:
        if n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        if n_rows == 1:
            axes = np.array([axes])

    max_legend_elements = 0

    for i in range(n_rows):
        for j in range(n_cols):
            axes[i, j].ticklabel_format(axis="y", scilimits=[-scilimit, scilimit])
            if y_labels is not None:
                y_label = y_labels[i]
            else:
                if y_label_by_row:
                    name_fn = get_plot_name_fn(rows)
                    y_label = name_fn(rows_vals[i])
                else:
                    y_label = None

            if x_labels is not None:
                x_label = x_labels[i]
            else:
                x_label = None

            if y_lims is not None:
                if isinstance(y_lims, tuple):
                    y_lim = y_lims
                else:
                    if isinstance(y_lims[0], tuple):
                        y_lim = y_lims[i]
                    else:
                        if isinstance(y_lims[0, 0], tuple):
                            y_lim = y_lims[i, j]
                        else:
                            raise ValueError()
            else:
                y_lim = None

            plot_dict = {cols: cols_vals[j], rows: rows_vals[i]}
            plot_kwargs = {**kwargs, **plot_dict}

            if verbose:
                print(plot_kwargs)
            try:
                plot_fn(name, ax=axes[i, j], color_map=color_map, **plot_kwargs)
            except Exception as e:
                if verbose:
                    print(str(e))
                    # Print traceback
                    import traceback

                    traceback.print_exc()

            if y_label is not None:
                axes[i, j].set_ylabel(y_label)
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)
            else:
                fn = get_plot_name_fn(cols)
                y_label = axes[i, j].get_ylabel()
                axes[i, j].set_ylabel(fn(y_label))
                axes[i, j].yaxis.set_label_coords(-0.3, 0.5)

            if x_label is not None:
                axes[i, j].set_xlabel(x_label)
            else:
                fn = get_plot_name_fn(rows)
                x_label = axes[i, j].get_xlabel()
                axes[i, j].set_xlabel(fn(x_label))
            if i == 0:
                name_fn = get_plot_name_fn(cols)
                axes[i, j].set_title(name_fn(cols_vals[j]))

            if i < n_rows - 1:
                axes[i, j].set_xlabel(None)
                axes[i, j].set_xticklabels([])

            if j > 0:
                axes[i, j].set_ylabel(None)

            if y_lim is not None:
                axes[i, j].set_ylim(y_lim)

            if i > 0:
                axes[i, j].set_title(None)

            if axes[i, j].get_legend() is not None:
                legend = axes[i, j].get_legend()
                if len(legend.get_texts()) > max_legend_elements:
                    max_legend_elements = len(legend.get_texts())
                    legend_text = [t._text for t in legend.get_texts()]
                    if legend_title:
                        legend_title = legend.get_title()._text
                    else:
                        legend_title = ""
                    legend_handles = legend.legend_handles
                legend.remove()

    for i in range(n_rows):
        for j in range(n_cols):
            if len(axes[i, j].lines) == 0 and len(axes[i, j].collections) == 0:
                axes[i, j].text(
                    0.5,
                    0.5,
                    "No data",
                    bbox={
                        "facecolor": "white",
                        "alpha": 1,
                        "edgecolor": "none",
                        "pad": 1,
                    },
                    ha="center",
                    va="center",
                )

    if fig_legend and "legend_text" in locals() and len(legend_text) > 0:
        text = [use_all_plot_name_fn(t) for t in list(dict.fromkeys(legend_text))]
        handles = list(dict.fromkeys(legend_handles))
        fig.legend(
            labels=text,
            handles=handles,
            title=use_all_plot_name_fn(str(legend_title)),
            ncol=legend_ncol,
            loc="lower center",
            bbox_to_anchor=legend_bbox_to_anchor,
            **legend_kwargs,
        )

    # fig.tight_layout()
    if fig_title is not None:
        fig.suptitle(fig_title)
    return fig, axes
