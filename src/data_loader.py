import pandas as pd
import faostat

def fetch_crop_yield_data(country: str, item: str, years: list):
    """
    Fetch data from FAOSTAT for a given crop (item) in a country, across years.
    """
    dataset_code = 'QCL'  # Crops & livestock products domain :contentReference[oaicite:4]{index=4}
    # Get parameter codes
    items = faostat.get_par(dataset_code, 'item')
    # find item code for `item`
    item_code = items.get(item)
    if item_code is None:
        raise ValueError(f"Item {item} not found in FAOSTAT items.")
    pars = {
        'area': country,
        'item': item_code,
        'year': years
    }
    df = faostat.get_data_df(dataset_code, pars=pars, strval=False)
    return df

def load_local_processed(path: str):
    return pd.read_csv(path)
