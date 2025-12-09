import os
import pandas as pd

def download_population_data():
    # Example: assume you manually download CSV from Kaggle and place in raw/
    print("Please download population dataset from Kaggle (e.g. 'World Population Dataset') and place as data/raw/world_population.csv")

def download_crop_yield_data():
    # Using FAOSTAT API to pull a subset
    import faostat
    dataset_code = 'QCL'
    # Example: fetch for one item and one country
    pars = {
        'area': 'Canada',
        'item': 'Wheat',
        'year': list(range(2000, 2023))
    }
    df = faostat.get_data_df(dataset_code, pars=pars, strval=False)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/faostat_crop_yield_subset.csv', index=False)
    print("Saved FAOSTAT crop yield subset to data/raw/faostat_crop_yield_subset.csv")

def main():
    os.makedirs('data/processed', exist_ok=True)
    download_population_data()
    download_crop_yield_data()

if __name__ == '__main__':
    main()
