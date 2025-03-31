import seaborn as sns
import matplotlib.pyplot as plt

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def plot_boxplots(data, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(y=data["Sales Volume"], ax=axes[0], color="skyblue")
    axes[0].set_title(f"Boxplot - Sales Volume ({title})")

    sns.boxplot(y=data["Price"], ax=axes[1], color="lightcoral")
    axes[1].set_title(f"Boxplot - Price ({title})")

    plt.tight_layout()
    return fig

def assign_region(country):
    est = ['Romania', 'Poland', 'Hungary', 'Bulgaria', 'Czech Republic']
    vest = ['France', 'Germany', 'Italy', 'Spain', 'United Kingdom']
    asia = ['China', 'Japan', 'India', 'South Korea']
    america = ['United States', 'Canada', 'Brazil']
    other = ['Australia']

    if country in est:
        return "Europa de Est"
    elif country in vest:
        return "Europa de Vest"
    elif country in asia:
        return "Asia"
    elif country in america:
        return "America"
    elif country in other:
        return "Altele"
    return "Necunoscut"