import numpy as np
import pandas as pd

class data_preprocessor:
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