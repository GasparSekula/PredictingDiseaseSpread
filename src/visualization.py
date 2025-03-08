import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def missing_data_percentage_by_city(
    data_city_sj: pd.DataFrame, data_city_iq: pd.DataFrame
) -> None:
    missing_percentage_sj = pd.isnull(data_city_sj).mean() * 100
    missing_percentage_iq = pd.isnull(data_city_iq).mean() * 100
    plt.figure(figsize=(12, 6))
    missing_percentage_sj.sort_index().plot(
        kind="bar", alpha=0.7, position=0, width=0.4, label="San Juan"
    )
    missing_percentage_iq.sort_index().plot(
        kind="bar", color="orange", alpha=0.7, position=1, width=0.4, label="Iquitos"
    )
    plt.xlabel("Columns")
    plt.ylabel("Missing Data (%)")
    plt.title("Missing Data Percentage by Column")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


def weekly_cases_by_year(labels: pd.DataFrame, city: str) -> None:
    weekofyear = labels.index.get_level_values("weekofyear")
    year = labels.index.get_level_values("year")
    labels_plot = labels[["total_cases"]].reset_index()
    labels_plot["weekofyear"] = weekofyear
    labels_plot["year"] = year
    labels_plot = (
        labels_plot.groupby(["weekofyear", "year"])
        .mean()[["total_cases"]]
        .reset_index()
    )
    years = labels_plot["year"].unique()
    colors = plt.get_cmap("tab20")

    plt.figure(figsize=(16, 12))
    for i, y in enumerate(years):
        if i > 0:
            plt.plot(
                "weekofyear",
                "total_cases",
                data=labels_plot[labels_plot["year"] == y],
                color=colors(i),
                label=y,
            )
            plt.text(
                labels_plot.loc[labels_plot.year == y, :].shape[0] + 0.1,
                labels_plot.loc[labels_plot.year == y, "total_cases"][-1:].values[0],
                y,
                fontsize=12,
                color=colors(i),
            )

    plt.gca().set(ylabel="Total Cases", xlabel="Week of Year")
    plt.yticks(fontsize=12, alpha=0.7)
    plt.title(f"Seasonal Plot - Weekly Cases for {city}", fontsize=20)
    plt.ylabel("Total Cases")
    plt.xlabel("Week of Year")
    plt.show()


def plot_heatmap(data_city_sj: pd.DataFrame, data_city_iq: pd.DataFrame) -> None:
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    correlation_matrix_sj = data_city_sj.drop("week_start_date", axis=1).corr()
    axes[0].set_title("Correlation Heatmap (San Juan)")
    sns.heatmap(correlation_matrix_sj, cmap="YlGnBu", ax=axes[0])
    correlation_matrix_iq = data_city_iq.drop("week_start_date", axis=1).corr()
    sns.heatmap(correlation_matrix_iq, cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Correlation Heatmap (Iquitos)")
    plt.show()


def plot_correlation_with_total_cases(
    data_city_sj: pd.DataFrame, data_city_iq: pd.DataFrame
) -> None:
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharey=True)
    correlation_matrix_sj = data_city_sj.drop("week_start_date", axis=1).corr()
    correlation_matrix_iq = data_city_iq.drop("week_start_date", axis=1).corr()

    sorted_index = (
        correlation_matrix_sj.total_cases.drop("total_cases")
        .sort_values(ascending=False)
        .index
    )

    correlation_matrix_sj.total_cases.drop("total_cases").reindex(
        sorted_index
    ).plot.barh(ax=axes[0])
    axes[0].set_title("Correlation with total cases - San Juan (Sorted)")

    correlation_matrix_iq.total_cases.drop("total_cases").reindex(
        sorted_index
    ).plot.barh(ax=axes[1])
    axes[1].set_title("Correlation with total cases - Iquitos (Same Order)")
    plt.show()


def distribution_plots(data_city_sj: pd.DataFrame, data_city_iq: pd.DataFrame) -> None:
    data_city_sj["City"] = "San Juan"
    data_city_iq["City"] = "Iquitos"

    combined_data = pd.concat([data_city_sj, data_city_iq])

    num_cols = len(data_city_sj.columns) - 1
    num_rows = (num_cols + 2) // 3

    _, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 4 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(data_city_sj.columns[:-1]):
        sns.boxplot(x="City", y=col, data=combined_data, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")

    for j in range(num_cols, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def weekly_cases(data: pd.DataFrame, city: str) -> None:
    sns.lineplot(x="weekofyear", y="total_cases", data=data, label=city)


def pairplot(data: pd.DataFrame) -> None:
    sns.PairGrid(data, corner=True).map(sns.scatterplot)


def train_vs_test_distribution(
    data_city_sj: pd.DataFrame, data_city_iq: pd.DataFrame
) -> None:
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    sns.histplot(
        data=data_city_sj,
        x=data_city_sj["week_start_date"],
        hue="dataset",
        edgecolor="none",
        ax=axes[0],
    )

    sns.histplot(
        data=data_city_iq,
        x=data_city_iq["week_start_date"],
        hue="dataset",
        edgecolor="none",
        ax=axes[1],
    )
    axes[0].set_xticks(data_city_sj["week_start_date"][::50])
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Train vs test dates (San Juan)")

    axes[1].set_xticks(data_city_iq["week_start_date"][::50])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Train vs test dates (Iquitos)")

    plt.tight_layout()
    plt.show()


def train_vs_test_features(
    data_city_sj: pd.DataFrame, data_city_iq: pd.DataFrame
) -> None:

    n_cols = len(data_city_sj.columns) - 1
    _, ax = plt.subplots(nrows=n_cols, ncols=2, figsize=(15, 4 * n_cols))

    for i, col in enumerate([c for c in data_city_sj.columns if c != "dataset"]):
        sns.histplot(
            data=data_city_sj[data_city_sj["dataset"] == "Train"],
            x=col,
            kde=False,
            label="Train",
            ax=ax[i, 0],
            stat="density",
            edgecolor="none",
            alpha=0.7,
        )
        sns.histplot(
            data=data_city_sj[data_city_sj["dataset"] == "Test"],
            x=col,
            kde=False,
            label="Test",
            color="orange",
            ax=ax[i, 0],
            stat="density",
            edgecolor="none",
            alpha=0.7,
        )
        ax[i, 0].legend(loc="upper right")
        ax[i, 0].set_title(f"San Juan - {col}")

        sns.histplot(
            data=data_city_iq[data_city_iq["dataset"] == "Train"],
            x=col,
            kde=False,
            label="Train",
            ax=ax[i, 1],
            stat="density",
            edgecolor="none",
            alpha=0.7,
        )
        sns.histplot(
            data=data_city_iq[data_city_iq["dataset"] == "Test"],
            x=col,
            kde=False,
            label="Test",
            color="orange",
            ax=ax[i, 1],
            stat="density",
            edgecolor="none",
            alpha=0.7,
        )
        ax[i, 1].legend(loc="upper right")
        ax[i, 1].set_title(f"Iquitos - {col}")

    plt.tight_layout()
    plt.show()
