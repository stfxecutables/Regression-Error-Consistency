import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.constants import OUT, PLOT_OUTPUT_PATH

CSVS = [
    # OUT / "Wine_error.csv",
    # OUT / "Bike_error.csv",
    OUT / "House_error.csv",
    # OUT / 'Breast-Cancer_error.csv',
    # OUT / 'Cancer_error.csv',
    # OUT / 'Diabetics_error.csv',
]

df = pd.concat([pd.read_csv(csv).assign(Dataset=csv.stem) for csv in CSVS]).reset_index(drop=True)
df = df.drop(df[(df.Regressor == "SVR") | (df.Regressor == "Knn") | (df.Regressor == "RF") |
                (df.Regressor == "Ridge") | (df.Regressor == "Lasso") | 
                (df.Method == "negative_incon") | (df.Method == "positive_incon")].index) # removing non hypertune models data

df['Dataset'].replace(regex=True,inplace=True,to_replace='_error',value=r'')
df['Regressor'].replace(regex=True,inplace=True,to_replace='-H',value=r'')

def ec_vs_accuracy_dataset():
    for dataset in df.Dataset.unique():
        data = df[df["Dataset"] == dataset]
        plt.figure(figsize=(15, 15))
        sns.scatterplot(data=data, x="EC", y="R2", hue="Method", style="Regressor", s=350)
        plt.legend(bbox_to_anchor=(1, 0.5), borderaxespad=0, loc="center left", fontsize=20, markerscale=2.5)
        plt.xlabel("Error Consistency (EC)", fontsize=25)
        plt.ylabel("R-Squared(R2)", fontsize=25)
        plt.title(f"Error Consistency vs R2 for the {dataset} dataset ", fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.savefig(PLOT_OUTPUT_PATH / 'new_medical_plots' / f"R2_{dataset}_only.png", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    ec_vs_accuracy_dataset()