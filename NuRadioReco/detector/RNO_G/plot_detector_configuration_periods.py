"""
Visualize the detector-configuration periods of one or more RNO-G stations.

A station's detector description changes whenever the station itself, one of
its channels, or a calibration is (de)commissioned. `Database.
query_modification_timestamps_per_station` (see `db_mongo_read.py`) returns,
per station, the sorted set of all such timestamps ("modification_timestamps")
plus the station-level (de)commission timestamps. Between two consecutive
modification timestamps the full detector description is constant - this
script draws each such interval as one "period" on a timeline.

Usage
-----
python plot_detector_configuration_periods.py --station-id 11 21
python plot_detector_configuration_periods.py --save periods.png
"""
import argparse
import datetime

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


from NuRadioReco.detector.RNO_G.db_mongo_read import Database

# Categorical palette (validated for readability/colorblind-safety), used to
# color the rows for different stations.
STATION_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
COLOR_COMMISSION = "#0ca30c"    # status "good"
COLOR_DECOMMISSION = "#d03b3b"  # status "critical"
COLOR_CHANNEL_CHANGE = "#898781"      # muted grey, dashed
COLOR_CALIBRATION_CHANGE = "#4a3aa7"  # violet, dotted


def _classify_change(t, station_data):
    """ Classify a modification timestamp as a channel or calibration change. """
    if t in station_data["channel_modification_timestamps"]:
        return "channel"
    if t in station_data["calibration_modification_timestamps"]:
        return "calibration"
    return "other"


def _describe_channel_change(t, station_data):
    """ Format which channel(s) changed at timestamp `t`, e.g. "Ch 3 decommissioned, Ch 7 commissioned". """
    changes = station_data["channel_modification_details"].get(t, [])
    return ", ".join(f"Ch {channel_id} {event}" for channel_id, event in changes)


def _station_periods(station_data, now, show_calibration_changes=True):
    """
    Split a station's commissioned time into configuration periods.

    Parameters
    ----------
    station_data: dict
        One entry of the dict returned by
        `Database.query_modification_timestamps_per_station`.
    now: datetime.datetime
        Used to cap the last, still-open commissioned interval for plotting.
    show_calibration_changes: bool (default: True)
        If False, calibration-only modification timestamps are ignored, so
        the timeline does not split into a new period at a calibration
        change (unless it coincides with a channel/station change).

    Returns
    -------
    periods: list of (start, end, ongoing, change_type, detail)
        One entry per configuration period. `change_type` is None for the
        first period of a commissioned interval (bounded by the station
        commissioning itself), otherwise "channel", "calibration", or
        "other", classifying the change at `start`. `detail` names the
        channel(s) that changed, for `change_type == "channel"` (else "").
    """
    mod_ts = sorted(station_data["modification_timestamps"])
    if not show_calibration_changes:
        mod_ts = [t for t in mod_ts if _classify_change(t, station_data) != "calibration"]
    commissions = sorted(station_data["station_commission_timestamps"])
    decommissions = sorted(station_data["station_decommission_timestamps"])

    periods = []
    for comm, decomm in zip(commissions, decommissions):
        ongoing = decomm > now
        end = min(decomm, now)

        # modification timestamps strictly inside this commissioned interval
        # split it into sub-periods (caused by a channel/calibration change)
        boundaries = sorted(set([comm] + [t for t in mod_ts if comm < t < decomm] + [end]))

        for i in range(len(boundaries) - 1):
            is_last = i == len(boundaries) - 2
            change_type = _classify_change(boundaries[i], station_data) if i > 0 else None
            detail = _describe_channel_change(boundaries[i], station_data) if change_type == "channel" else ""
            periods.append((boundaries[i], boundaries[i + 1], ongoing and is_last, change_type, detail))

    return periods


def _format_period_table(data, show_calibration_changes=True):
    """ Format the configuration periods of every station in `data` as a string table. """
    now = datetime.datetime.now()
    lines = []
    for station_id in sorted(data.keys()):
        periods = _station_periods(data[station_id], now, show_calibration_changes=show_calibration_changes)
        lines.append(f"Station {station_id}: {len(periods)} configuration period(s)")
        for i, (start, end, ongoing, change_type, detail) in enumerate(periods):
            end_str = "ongoing" if ongoing else end.date().isoformat()
            duration = (end - start).days
            if change_type == "channel":
                reason = f" [{detail}]"
            elif change_type:
                reason = f" [{change_type} change]"
            else:
                reason = ""
            lines.append(f"  [{i}] {start.date().isoformat()} -> {end_str}  ({duration} days){reason}")
        lines.append("")
    return "\n".join(lines).rstrip()


def print_period_table(data, show_calibration_changes=True):
    """ Print the configuration periods of every station in `data` to stdout. """
    print(_format_period_table(data, show_calibration_changes=show_calibration_changes))


def plot_configuration_periods(data, ax=None, show_calibration_changes=True):
    """
    Draw one timeline row per station, with each detector-configuration
    period shown as a colored bar. Station (de)commissioning is marked with
    triangles; sub-periods within a commissioned interval (caused by a
    channel/calibration change) alternate between two shades of the
    station's color and are marked with a short tick at their boundary.

    Parameters
    ----------
    show_calibration_changes: bool (default: True)
        If False, calibration changes are not shown: the timeline does not
        split into a new period at a calibration-only change, and the
        legend/table omit them.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(3, 1 + 0.6 * len(data))))

    now = datetime.datetime.now()
    station_ids = sorted(data.keys())

    change_style = {
        "channel": (COLOR_CHANNEL_CHANGE, "--"),
        "calibration": (COLOR_CALIBRATION_CHANGE, ":"),
        "other": ("black", "-."),
    }

    for row, station_id in enumerate(station_ids):
        color = STATION_COLORS[row % len(STATION_COLORS)]
        periods = _station_periods(data[station_id], now, show_calibration_changes=show_calibration_changes)

        for i, (start, end, ongoing, change_type, _) in enumerate(periods):
            start_num, end_num = mdates.date2num(start), mdates.date2num(end)
            ax.broken_barh(
                [(start_num, end_num - start_num)],
                (row - 0.4, 0.8),
                facecolors=color,
                alpha=0.9 if i % 2 == 0 else 0.55,
                hatch="///" if ongoing else None,
                edgecolor="white", linewidth=0.5, zorder=2,
            )
            if change_type:
                change_color, change_ls = change_style[change_type]
                ax.vlines(start_num, row - 0.4, row + 0.4, color=change_color, lw=1.2, ls=change_ls, zorder=3)

        for t in data[station_id]["station_commission_timestamps"]:
            ax.plot(mdates.date2num(t), row, marker="^", color=COLOR_COMMISSION, markersize=7, zorder=4)
        for t in data[station_id]["station_decommission_timestamps"]:
            if t <= now:
                ax.plot(mdates.date2num(t), row, marker="v", color=COLOR_DECOMMISSION, markersize=7, zorder=4)

    ax.set_yticks(range(len(station_ids)))
    ax.set_yticklabels([f"St. {station_id}" for station_id in station_ids])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(station_ids) - 0.3)
    ax.invert_yaxis()

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.xaxis.grid(True, color="#e1e0d9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_COMMISSION,
               markeredgecolor=COLOR_COMMISSION, markersize=8, label="station commissioned"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor=COLOR_DECOMMISSION,
               markeredgecolor=COLOR_DECOMMISSION, markersize=8, label="station decommissioned"),
        Line2D([0], [0], color=COLOR_CHANNEL_CHANGE, lw=1.2, ls="--", label="channel change"),
    ]
    if show_calibration_changes:
        legend_elements.append(
            Line2D([0], [0], color=COLOR_CALIBRATION_CHANGE, lw=1.2, ls=":", label="calibration change"))
    legend_elements.append(mpatches.Patch(facecolor="0.6", edgecolor="white", hatch="///", label="ongoing period"))
    legend = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.01, 1),
                        frameon=False, fontsize=9)

    # Place the period table (identical to `print_period_table`'s stdout output)
    # in a text box below the legend, in the figure's right-hand margin.
    fig = ax.get_figure()
    fig.canvas.draw()
    legend_bbox_axes = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(
        1.01, legend_bbox_axes.y0 - 0.04, _format_period_table(data, show_calibration_changes=show_calibration_changes),
        transform=ax.transAxes, va="top", ha="left", fontsize=4, family="monospace",
        bbox=dict(boxstyle="round", facecolor="#fcfcfb", edgecolor="#c3c2b7", linewidth=0.8),
    )

    ax.set_title("RNO-G detector configuration periods")
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--station-id", type=int, nargs="*", default=None,
        help="Station id(s) to plot. Default: all stations found in the database.")
    parser.add_argument(
        "--database", default="RNOG_public",
        help='Database connection, see `Database.__init__` (default: "RNOG_public", read-only).')
    parser.add_argument(
        "--save", default=None,
        help="Additionally save the figure to this path.")
    parser.add_argument(
        "--hide-calibration-changes", action="store_true",
        help="Do not show calibration changes (legend, table, and period splitting).")
    args = parser.parse_args()
    show_calibration_changes = not args.hide_calibration_changes

    db = Database(database_connection=args.database)
    data = db.query_modification_timestamps_per_station(args.station_id)

    print_period_table(data, show_calibration_changes=show_calibration_changes)

    fig, ax = plt.subplots(figsize=(10, max(3, 1 + 0.6 * len(data))))
    plot_configuration_periods(data, ax=ax, show_calibration_changes=show_calibration_changes)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
