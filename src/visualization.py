import seaborn as sns
import matplotlib.pyplot as plt

def visualize_target_variable(df):
    """Visualize the target variable."""
    sns.countplot(x='diagnosis', data=df)
    plt.show()

# def box_hist_plots(data):
#     """Make box plots and histograms of data."""
#     Name = data.name.upper()
#     mean = data.mean()
#     median = data.median()
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     fig.suptitle("SPREAD OF DATA FOR " + Name)
#     sns.boxplot(x=data, data=data, ax=ax1)
#     sns.histplot(data, ax=ax2)
#     ax2.axvline(mean, color='r', linestyle='--', linewidth=2)
#     ax2.axvline(median, color='g', linestyle='-', linewidth=2)
#     plt.legend({'Mean': mean, 'Median': median})
#     plt.show()
def box_hist_plots(data):
    """Make box plots and histograms of data"""
    Name = data.name.upper()
    mean = data.mean()
    median = data.median()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("SPREAD OF DATA FOR " + Name)
    
    # Create a DataFrame from the Series for the boxplot
    sns.boxplot(data=data.to_frame(), ax=ax1)
    
    sns.histplot(data, ax=ax2, kde=True)
    
    # create mean and median line on histogram plot
    ax2.axvline(mean, color='r', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(median, color='g', linestyle='-', linewidth=2, label='Median')
    ax2.legend()

def visualize_correlation_matrix(df):
    """Visualize the correlation matrix."""
    sns.heatmap(df.corr(), annot=True)
    plt.show()

def pairplot_features(df, features):
    """Create pair plots for specified features."""
    sns.pairplot(df[features])
    plt.show()