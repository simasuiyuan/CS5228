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

    @staticmethod
    def remove_duplicates(df) -> pd.DataFrame:
        df_ = df.copy()
        """ same attribute records same price
        """
        df_ = df_.drop_duplicates(subset=df_.columns[1:].tolist(), keep='first')
        """ same attribute records different price => take average (+/- 200,000)
        """
        df_ = df_.groupby(df_.columns[1:-1].tolist(), dropna=False).mean().reset_index()
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
        df_['property_type_clean'] = df_['property_type'].str.lower()

        """ drop hdb rooms: this feature will be reflected on num_beds	num_baths
        """
        df_.loc[df_['property_type_clean'].str.contains('hdb') & (df_['property_type_clean'] != 'hdb executive'), 'property_type_clean'] = 'hdb'
        
        cat_order = df_.groupby('property_type_clean').median().sort_values('price').index.to_list()
        enc = OrdinalEncoder(categories=[cat_order])
        df_['property_type_cat'] = enc.fit_transform(df_['property_type_clean'].values.reshape(-1, 1))

        return df_

    @staticmethod
    def preprocess_tenure(df) -> pd.DataFrame:
        df_ = df.copy()
        """ ref https://www.theorigins.com.sg/post/freehold-vs-leasehold-condo-is-99-years-really-enough
        """
        df_.loc[df_['tenure'].isna(), 'tenure'] = 'others'
        df_.loc[df_['tenure'].str.contains(r'^1[0-9]{2}-year leasehold$'), 'tenure'] = '103/110-year leasehold'
        df_.loc[df_['tenure'].str.contains(r'^9[0-9]{2}-year leasehold$'), 'tenure'] = '999-year leasehold'
        df_.loc[df_['property_type']=='hdb', 'tenure'] = '99-year leasehold'

        cat_order = df_.groupby('tenure').median().sort_values('price').index.to_list()
        enc = OrdinalEncoder(categories=[cat_order])
        df_['tenure_cat'] = enc.fit_transform(df_['tenure'].values.reshape(-1, 1))
        return df_

    def impute_built_year_unify(sub_df) -> pd.DataFrame:
        unique_year = sub_df['built_year'].unique()
        if len(unique_year)==2:
            sub_df['built_year'] = [x for x in unique_year if str(x) != 'nan'][0]
        return sub_df
    
    @staticmethod
    def preprocess_built_year(df) -> pd.DataFrame:
        df_ = df.copy()
        # Imputation
        """ Using property_type, lat, lng to allocate built-year groups
            compromised resolution to 2 decimal will extend the number of distinct year + missing sample pairs 
            (high confidence to share the same built year)
        """
        df_[['lat_2d', 'lng_2d']] = df_[['lat', 'lng']].round(2)
        """ low resolution pairs: 'property_type','lat_2d', 'lng_2d'
        """
        df_ = df_.groupby(['property_type','lat_2d', 'lng_2d'])\
            .apply(DataPreprocessor.impute_built_year_1)
        """ higher resolution pairs: 'property_type','lat_2d', 'lng_2d','furnishing' 
        """
        df_ = df_.groupby(['property_type','lat_2d', 'lng_2d','subzone','furnishing'])\
            .apply(DataPreprocessor.impute_built_year_1)
        df_.drop(columns=['lat_2d', 'lng_2d'], inplace=True)
        return df_