import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def missing_data_percentage(data: pd.DataFrame) -> None:
    missing_percentage = pd.isnull(data).mean() * 100
    plt.figure(figsize=(12, 6))
    missing_percentage.sort_values(ascending=False).plot(kind="bar")
    plt.xlabel("Columns")
    plt.ylabel("Missing Data (%)")
    plt.title("Missing Data Percentage by Column")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

def weekly_cases_by_year(labels: pd.DataFrame) -> None:
    weekofyear = labels.index.get_level_values('weekofyear')
    year = labels.index.get_level_values('year')
    labels_plot = labels[['total_cases']].reset_index()
    labels_plot['weekofyear'] = weekofyear
    labels_plot['year'] = year
    labels_plot = labels_plot.groupby(['weekofyear', 'year']).mean()[['total_cases']].reset_index()
    years = labels_plot['year'].unique()
    colors = plt.get_cmap("tab20")

    plt.figure(figsize=(16,12))
    for i, y in enumerate(years):
        if i > 0:        
            plt.plot('weekofyear', 'total_cases', data=labels_plot[labels_plot['year'] == y], color=colors(i), label=y)
            plt.text(labels_plot.loc[labels_plot.year==y, :].shape[0]+0.1, labels_plot.loc[labels_plot.year==y, 'total_cases'][-1:].values[0], y, fontsize=12, color=colors(i))


    plt.gca().set(ylabel= 'Total Cases', xlabel = 'Week of Year')
    plt.yticks(fontsize=12, alpha=.7)
    plt.title("Seasonal Plot - Weekly Cases", fontsize=20)
    plt.ylabel('Total Cases')
    plt.xlabel('Week of Year')
    plt.show()

def plot_correlations(data: pd.DataFrame) -> None:
    correlation_matrix = data.drop("week_start_date", axis=1).corr()
    plt.figure()
    sns.heatmap(correlation_matrix, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    
    plt.figure()
    correlation_matrix.total_cases.drop('total_cases').sort_values(ascending=False).plot.barh()
    plt.title("Correlation with Total Cases")
    
    plt.show()
