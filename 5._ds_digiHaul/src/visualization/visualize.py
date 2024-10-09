import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
from cycler import cycler

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["set"] == 3]

plt.plot(set_df["acc_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# subset
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
colors = cycler(color=plt.get_cmap("tab10").colors)  # ["b", "r", "g"]

mpl.style.use("ggplot")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "lightgray"
mpl.rcParams["axes.prop_cycle"] = colors
mpl.rcParams["axes.linewidth"] = 1
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.titlesize"] = 25
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# category_df = df.query("label=='squat'")

category_df = df.query("label=='squat'").query("participant=='A'").reset_index()


fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participants_df = df.query("label=='bench'").sort_values("participant").reset_index()

fig, ax = plt.subplots()
participants_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)


fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("acc_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} {participant}".title())
            plt.legend()


labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} {participant}".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"

Combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
Combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
Combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=[1])


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
