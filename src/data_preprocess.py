import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import OrdinalEncoder

class DataPreprocessor:
    @staticmethod
    def remove_price_outlier(df) -> pd.DataFrame:
        df_ = df.copy()
        df_.drop(df_[zscore(df_['price']) > 3].index, inplace=True)
        return df_

    """ 1. handling missing data
        2. transform cat data
        3. drop the columns
    """
    @staticmethod
    def preprocess_title(df) -> pd.DataFrame:
        df_ = df.copy()
        """get property_type form title
        """
        df_['title_property_type'] = df_['title'].str.split(' for').str[0].str.split('bed').str[-1].str.strip()
        """get num bed form title
        """
        df_['title_n_beds'] = df_['title'].str.split(' for').str[0].str.split('bed').str[0].str.strip()
        """ check is all sale
        """
        print(df_['title'].str.split('for ').str[-1].str.split(' ').str[0].unique())
        """ get address form title
        """
        df_['title_address'] = df_['title'].str.split('in ').str[-1]
        return df_

    @staticmethod
    def preprocess_property_type(df) -> pd.DataFrame:
        df_ = df.copy()
        """ convert to small letters
        """
        df_['property_type'] = df_['property_type'].str.lower()

        """ drop hdb rooms: this feature will be reflected on num_beds	num_baths
        """
        df_.loc[df_['property_type'].str.contains('hdb'), 'property_type'] = 'hdb'
        
        cat_order = df_.groupby('property_type').mean().sort_values('price').index.to_list()
        enc = OrdinalEncoder(categories=[cat_order])
        df_['property_type_cat'] = enc.fit_transform(df_['property_type'].values.reshape(-1, 1))

        return df_

    @staticmethod
    def preprocess_tenure(df) -> pd.DataFrame:
        df_ = df.copy()
        """ ref https://www.theorigins.com.sg/post/freehold-vs-leasehold-condo-is-99-years-really-enough
        """
        df_.loc[df_['tenure'].isna(), 'tenure'] = ''
        df_.loc[df_['tenure'].str.contains(r'^1[0-9]{2}-year leasehold$'), 'tenure'] = '103/110-year leasehold'
        df_.loc[df_['tenure'].str.contains(r'^9[0-9]{2}-year leasehold$'), 'tenure'] = '999-year leasehold'
        df_.loc[df_['property_type']=='hdb', 'tenure'] = '99-year leasehold'
        return df_