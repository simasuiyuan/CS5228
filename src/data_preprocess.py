import numpy as np
import pandas as pd
from time import time
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pickle
import pathlib
from tqdm import tqdm
from math import sin, cos, sqrt, atan2, radians

WORKING_DIR = pathlib.Path(__file__).parent.parent.resolve()
# depends on if model can handle NaN or not
KEEP_UNCERTAIN = False

ENCODER = {
    "property_type_encoder": OrdinalEncoder(),
    "tenure_encoder": OneHotEncoder(),
    "total_level_encoder": OrdinalEncoder(),
    "floor_level_encoder": OneHotEncoder(),
    "furnishing_encoder": OneHotEncoder(),
    "planning_area_encoder": OrdinalEncoder(),
    "CR_encoder": OrdinalEncoder(),
    "IEBP_encoder": OrdinalEncoder(),
    "BN_encoder": OrdinalEncoder(),
    "IHL_encoder": OrdinalEncoder(),
    "pri_sch_encoder": OrdinalEncoder(),
    "sec_sch_encoder": OrdinalEncoder(),
    "shopping_mall_encoder": OrdinalEncoder(),
}

class DataPreprocessor:
    @staticmethod
    def remove_price_outlier(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        if not test:
            df_.drop(df_[zscore(df_['price']) > 3].index, inplace=True)
            # df_.drop(df_[zscore(df_['lat']) > 2].index, inplace=True)
            # df_.drop(df_[~((zscore(train_df_clean['lng']) > -1) & (zscore(train_df_clean['lng']) < 1))].index, inplace=True)
        return df_

    @staticmethod
    def remove_duplicates(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        if not test:
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
    def preprocess_title(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        """get property_type form title
        """
        df_['title_property_type'] = df_['title'].str.split(' for').str[0].str.split('bed').str[-1].str.strip()
        """get num bed form title
        """
        df_['title_n_beds'] = df_['title'].str.split(' for').str[0].str.split('bed').str[0].str.strip()
        df_.loc[df_['title_n_beds'].str.contains("studio"), 'title_n_beds'] = '1'
        """ check is all sale
        """
        #print(df_['title'].str.split('for ').str[-1].str.split(' ').str[0].unique())
        """ get address form title
        """
        df_['title_address'] = df_['title'].str.split('in ').str[-1]
        return df_

    @staticmethod
    def preprocess_lat_lng(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        lat_lng_impute_dict = {
            "1953": (1.3165065395052853, 103.85747739945413),
            "m5": (1.2959352730861813, 103.82885982666735), 
            "ness": (1.313556425343857, 103.88705437671443),
            "pollen & bleu": (1.3138003193645669, 103.80676998433876)
        }
        for key, val in lat_lng_impute_dict.items():
            df.loc[df["title_address"]==key, ["lat", "lng"]] = val
        return df

    @staticmethod
    def preprocess_property_type(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        """ convert to small letters
        """
        df_['property_type_clean'] = df_['property_type'].str.lower()

        """ drop hdb rooms: this feature will be reflected on num_beds	num_baths
        """
        df_.loc[df_['property_type_clean'].str.contains('hdb') & (df_['property_type_clean'] != 'hdb executive'), 'property_type_clean'] = 'hdb'
        
        if not test:
            cat_order = df_.groupby('property_type_clean').median().sort_values('price').index.to_list()
            property_type_encoder = ENCODER["property_type_encoder"]
            property_type_encoder.set_params(categories=[cat_order])
            property_type_encoder.fit(df_['property_type_clean'].values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/'property_type_encoder.sav', 'wb') as f:
                pickle.dump(property_type_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/'property_type_encoder.sav', 'rb') as f:
                property_type_encoder = pickle.load(f)

        property_type_cat = property_type_encoder.transform(df_['property_type_clean'].values.reshape(-1, 1))
        if type(property_type_cat)!=np.ndarray: property_type_cat = property_type_cat.toarray()
        for feature in range(property_type_cat.shape[1]):
            df_[f'property_type_cat_{feature}']  = property_type_cat[:, feature]
        return df_

    @staticmethod
    def preprocess_tenure(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        """ ref https://www.theorigins.com.sg/post/freehold-vs-leasehold-condo-is-99-years-really-enough
        """
        df_.loc[df_['tenure'].isna(), 'tenure'] = 'others'
        df_.loc[df_['tenure'].str.contains(r'^1[0-9]{2}-year leasehold$'), 'tenure'] = '103/110-year leasehold'
        df_.loc[df_['tenure'].str.contains(r'^9[0-9]{2}-year leasehold$'), 'tenure'] = '999-year leasehold'
        df_.loc[df_['property_type']=='hdb', 'tenure'] = '99-year leasehold'
        
        if not test:
            cat_order = df_.groupby('tenure').median().sort_values('price').index.to_list()
            tenure_encoder = ENCODER["tenure_encoder"]
            tenure_encoder.set_params(categories=[cat_order])
            tenure_encoder.fit(df_['tenure'].values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/'tenure_encoder.sav', 'wb') as f:
                pickle.dump(tenure_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/'tenure_encoder.sav', 'rb') as f:
                tenure_encoder = pickle.load(f)

        tenure_cat = tenure_encoder.transform(df_['tenure'].values.reshape(-1, 1))
        if type(tenure_cat)!=np.ndarray: tenure_cat = tenure_cat.toarray()
        for feature in range(tenure_cat.shape[1]):
            df_[f'tenure_cat_{feature}']  = tenure_cat[:, feature]
        return df_

    def impute_built_year_unify(sub_df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        unique_year = sub_df['built_year'].unique()
        if len(unique_year)==2:
            sub_df['built_year'] = [x for x in unique_year if str(x) != 'nan'][0]
        return sub_df
    
    def impute_built_year_uncertain(sub_df) -> pd.DataFrame:
        unique_year = sub_df['built_year'].unique()
        if (np.isnan(np.sum(unique_year)))& (len(unique_year) > 1):
            sub_df['built_year'] = int(np.median([x for x in unique_year if str(x) != 'nan']))
        return sub_df
    
    @staticmethod
    def preprocess_built_year(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        """ unfinished!!!!!!!!!!
        """
        df_ = df.copy()
        # Imputation
        """ Using property_type, lat, lng to allocate built-year groups
            compromised resolution to 2 decimal will extend the number of distinct year + missing sample pairs 
            (high confidence to share the same built year)
        """
        df_ = df_.groupby(['lat', 'lng'], dropna=False).apply(DataPreprocessor.impute_built_year_unify)
        df_ = df_.groupby(['lat', 'lng','property_type_clean'], dropna=False).apply(DataPreprocessor.impute_built_year_unify)
        # """ get hdb block number form title_address 
        # """
        df_.loc[df_['property_type_clean']=='hdb', "block_number"] = df_[df_['property_type_clean']=='hdb']['title_address'].str.split(' ').str[0].str.extract(r'(\d+)', expand=False)
        df_ = df_.groupby(['lat', 'lng','property_type_clean', 'block_number'], dropna=False).apply(DataPreprocessor.impute_built_year_unify)
        if not uncertain:
            df_['lat_lowres'] = df_['lat'].round(3)
            df_['lng_lowres'] = df_['lng'].round(3)
            df_ = df_.groupby(['lat_lowres', 'lng_lowres', 'property_type_clean'], dropna=False).apply(DataPreprocessor.impute_built_year_unify)
            df_ = df_.groupby(['lat_lowres', 'lng_lowres', 'property_type_clean'], dropna=False).apply(DataPreprocessor.impute_built_year_uncertain)
        return df_

    def impute_num_beds_uncertain(sub_df) -> pd.DataFrame:
        num_beds = sub_df['num_beds'].unique()
        if np.isnan(np.sum(num_beds)):
            sub_df['num_beds'] = int(np.median([x for x in num_beds if str(x) != 'nan']))
        return sub_df

    @staticmethod
    def preprocess_num_beds(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        # Imputation
        """ Using number of beds extracted from title to impute
        """
        df_.loc[(df_["num_beds"].isna()) & (df_["title_n_beds"].str.isdigit()), "num_beds"] = \
            df_.loc[(df_["num_beds"].isna()) & (df_["title_n_beds"].str.isdigit()), "title_n_beds"].astype(float)

        """ => still has some fuckers contain NaN => @title_n_beds == hdb flat
        use size_sqft to take meadian (uncertain!)
        """
        if not uncertain:
            df_ = df_.groupby(['size_sqft'], dropna=False).apply(DataPreprocessor.impute_num_beds_uncertain)
        return df_

    def impute_num_baths_unify(sub_df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        unique_num_baths = sub_df['num_baths'].unique()
        if len(unique_num_baths)==2:
            sub_df['num_baths'] = [x for x in unique_num_baths if str(x) != 'nan'][0]
        return sub_df
    
    def impute_num_baths_uncertain(sub_df) -> pd.DataFrame:
        unique_num_baths = sub_df['num_baths'].unique()
        if np.isnan(np.sum(unique_num_baths)) & (len(unique_num_baths)>1):
            sub_df['num_baths'] = int(np.median([x for x in unique_num_baths if str(x) != 'nan']))
        return sub_df

    @staticmethod
    def preprocess_num_baths(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        # Imputation
        """ High confidence => same type, size and number of beds => same number bath
        """
        df_ = df_.groupby(['property_type_clean','size_sqft','num_beds'], dropna=False).apply(DataPreprocessor.impute_num_baths_unify)

        """ => still has some fuckers contain NaN 
        """

        if not uncertain:
            df_ = df_.groupby(['property_type_clean', 'block_number', 'size_sqft','num_beds', 'lat', 'lng'], dropna=False).apply(DataPreprocessor.impute_num_baths_uncertain)
            df_ = df_.groupby(['property_type_clean', 'block_number', 'size_sqft', 'num_beds'], dropna=False).apply(DataPreprocessor.impute_num_baths_uncertain)
            df_ = df_.groupby(['property_type_clean', 'size_sqft', 'num_beds'], dropna=False).apply(DataPreprocessor.impute_num_baths_uncertain)
            def __house_size_cat(row):
                size = row['size_sqft']
                if size < 500:
                    return "small"
                elif size>= 500 & size< 1000:
                    return "medium"
                elif size>= 1000 & size <2000:
                    return "large"
                elif size>=2000 & size < 3000:
                    return "super"
                elif size>= 3000 & size < 4000:
                    return "super ass rich"
                else:
                    return "_"
            df_['size_sqft_cat'] = df_.apply(__house_size_cat, axis = 1)
            df_ = df_.groupby(['property_type_clean', 'size_sqft_cat', 'num_beds'], dropna=False).apply(DataPreprocessor.impute_num_baths_uncertain)
            df_.drop('size_sqft_cat', axis=1,inplace=True)
        return df_

    def impute_size_sqft_unify(sub_df) -> pd.DataFrame:
        unique_size_sqft = sub_df['size_sqft'].unique()
        if len(unique_size_sqft)==2:
            sub_df['size_sqft'] = [x for x in unique_size_sqft if x != 0][0]
        return sub_df
        
    @staticmethod
    def preprocess_size_sqft(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        if not test:
            df_.drop(df_[zscore(df_['size_sqft']) > 5].index, inplace=True)
        df_.loc[df_["size_sqft"] <= 200, "size_sqft"] = (df_.loc[df_["size_sqft"] <= 200, "size_sqft"] * 10.76391).astype(int)
        df_ = df_.groupby(['title','property_type', 'num_beds', 'num_baths'], dropna=False).apply(DataPreprocessor.impute_size_sqft_unify)
        return df_

    def _floor_level_refinement(row, floor_lvl_type):
        property_type = row["property_type_clean"]
        if property_type not in floor_lvl_type:
            return "no_level"
        else:
            return row["floor_level"]
        # elif str(original_floor_level) == 'nan':
        #     return 0, 1 
        # else:
        #      level = re.sub(r'\(.+\)', "", original_floor_level).strip()
        #      total_level = re.find(r'\(\d+ total\)').
            
    @staticmethod
    def preprocess_floor_level(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        FLOOR_LEVEL_TYPE = ["condo", "apartment", "executive condo", "hdb", "hdb executive"]
        df_ = df.copy()
        df_["floor_level"] = df_.apply(lambda sub_df: DataPreprocessor._floor_level_refinement(sub_df, FLOOR_LEVEL_TYPE), axis=1)
        
        df_["floor_level_cat"] = df_["floor_level"].str.split(' ').str[0]
        df_["total_level_cat"] = df_["floor_level"].str.split(' ').str[1].str.replace(r'\(', '', regex=True)

        if not test:
            total_level_cat_order = np.sort(df_["total_level_cat"].astype(float).unique())
            total_level_cat_order = [str(x) for x in total_level_cat_order]
            total_level_encoder = ENCODER["total_level_encoder"]
            total_level_encoder.set_params(
                categories=[total_level_cat_order], 
                handle_unknown='use_encoded_value', 
                unknown_value=len(total_level_cat_order))
            total_level_encoder.fit(df_['total_level_cat'].astype(float).astype(str).values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/'total_level_encoder.sav', 'wb') as f:
                pickle.dump(total_level_encoder, f)
            
            floor_level_cat_order = ['ground', 'low', 'mid', 'high', 'top', 'penthouse', 'no_level', 'nan']
            floor_level_encoder = ENCODER["floor_level_encoder"]
            floor_level_encoder.set_params(categories=[floor_level_cat_order])
            floor_level_encoder.fit(df_['floor_level_cat'].astype(str).values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/'floor_level_encoder.sav', 'wb') as f:
                pickle.dump(floor_level_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/'total_level_encoder.sav', 'rb') as f:
                total_level_encoder = pickle.load(f)
            with open(WORKING_DIR/'lib'/'floor_level_encoder.sav', 'rb') as f:
                floor_level_encoder = pickle.load(f)
        
        total_level_cat = total_level_encoder.transform(df_['total_level_cat'].astype(float).astype(str).values.reshape(-1, 1))
        if type(total_level_cat)!=np.ndarray: total_level_cat = total_level_cat.toarray()
        for feature in range(total_level_cat.shape[1]):
            df_[f'total_level_cat_{feature}']  = total_level_cat[:, feature]

        floor_level_cat = floor_level_encoder.transform(df_['floor_level_cat'].astype(str).values.reshape(-1, 1))
        if type(floor_level_cat)!=np.ndarray: floor_level_cat = floor_level_cat.toarray()
        for feature in range(floor_level_cat.shape[1]):
            df_[f'floor_level_cat_{feature}']  = floor_level_cat[:, feature]
        
        df_ = df_.drop(["floor_level_cat","total_level_cat"], axis=1,inplace=False)
        return df_
    
    @staticmethod
    def preprocess_furnishing(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        df_.loc[df_["furnishing"]=='na', "furnishing"] = 'unspecified'
        if not test:
            furnishing_cat_order = ['unfurnished','partial','fully','unspecified']
            furnishing_encoder = ENCODER["furnishing_encoder"]
            furnishing_encoder.set_params(categories=[furnishing_cat_order])
            furnishing_encoder.fit(df_['furnishing'].astype(str).values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/'furnishing_encoder.sav', 'wb') as f:
                pickle.dump(furnishing_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/'furnishing_encoder.sav', 'rb') as f:
                furnishing_encoder = pickle.load(f)

        furnishing_cat = furnishing_encoder.transform(df_['furnishing'].astype(str).values.reshape(-1, 1))
        if type(furnishing_cat)!=np.ndarray: furnishing_cat = furnishing_cat.toarray()
        for feature in range(furnishing_cat.shape[1]):
            df_[f'furnishing_cat_{feature}']  = furnishing_cat[:, feature]
        return df_

    def extract_features_available_unit_types(row):
        available_unit_types = row['available_unit_types_temp']
        if available_unit_types:
            number_of_types = len(available_unit_types)
            has_studio = 1 if 'studio' in available_unit_types else 0
            num_beds = [int(x.strip()) for x in available_unit_types if x.strip().isdigit()]
            if len(num_beds)>0:
                min_br_available = min(num_beds)
                max_br_available = max(num_beds)
            else:
                min_br_available, max_br_available = 0,0
            return pd.Series({
                "number_of_types_available": number_of_types,
                "has_studio":has_studio,
                "min_br_available":min_br_available,
                "max_br_available":max_br_available
            })
        else:
            return pd.Series({
                "number_of_types_available": np.nan,
                "has_studio":np.nan,
                "min_br_available":np.nan,
                "max_br_available":np.nan
            })

    @staticmethod
    def preprocess_available_unit_types(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        df_["available_unit_types_temp"] = df_["available_unit_types"].astype(str).str.replace('br','').str.split(',')
        features_df = df_.apply(DataPreprocessor.extract_features_available_unit_types, axis=1)
        df_ = df_.merge(features_df, left_index=True, right_index=True)
        df_.drop('available_unit_types_temp', axis=1,inplace=True)
        return df_
    
    def impute_planning_area(df):
        planning_area_impute_dict = {
            "1953": "serangoon",
            "m5": "tanglin", 
            "ness": "geylang",
            "pollen & bleu": "bukit timah"
        }
        for key, val in planning_area_impute_dict.items():
            df.loc[df["title_address"]==key, "planning_area"] = val
        return df
    @staticmethod
    def preprocess_planning_area(df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        df_ = DataPreprocessor.impute_planning_area(df_)
        if not test:
            planning_area_list = list(df_['planning_area'].unique())
            planning_area_list.append('nan')
            planning_area_encoder = ENCODER["planning_area_encoder"]
            planning_area_encoder.set_params(categories=[planning_area_list], handle_unknown='use_encoded_value', unknown_value=len(planning_area_list))
            planning_area_encoder.fit(df_['planning_area'].astype(str).values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/'planning_area_encoder.sav', 'wb') as f:
                pickle.dump(planning_area_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/'planning_area_encoder.sav', 'rb') as f:
                planning_area_encoder = pickle.load(f)
        planning_area_cat = planning_area_encoder.transform(df_['planning_area'].astype(str).values.reshape(-1, 1))
        if type(planning_area_cat)!=np.ndarray: planning_area_cat = planning_area_cat.toarray()
        for feature in range(planning_area_cat.shape[1]):
            df_[f'planning_area_cat_{feature}']  = planning_area_cat[:, feature]
        return df_

    @staticmethod
    def timer_func(func):
        # This function shows the execution time of 
        # the function object passed
        def wrap_func(*args, **kwargs):
            t1 = time()
            result = func(*args, **kwargs)
            t2 = time()
            log_msg = f'Function {func.__name__!r} executed in {(t2-t1):.4f}s'
            return result, log_msg
        return wrap_func
    
    @staticmethod
    def data_preprocessing_v1(
        df: pd.DataFrame, 
        test:bool=False, 
        uncertain:bool=False, 
        drop_na:bool=False, 
        remove_original_attributes:bool=True
    ) -> pd.DataFrame:
        df_clean = df.copy()
        drop_attributes = {'listing_id'}

        preprocessing_pipeline = [
            DataPreprocessor.remove_price_outlier, # Excessive outliers
            DataPreprocessor.remove_duplicates, # Duplicated records
            DataPreprocessor.preprocess_title, # extract features form title
            DataPreprocessor.preprocess_lat_lng, # correct lat and lng
            DataPreprocessor.preprocess_property_type, # property_type
            DataPreprocessor.preprocess_tenure, # tenure
            DataPreprocessor.preprocess_built_year, # built_year -  unfinished!
            DataPreprocessor.preprocess_num_beds, # num_beds
            DataPreprocessor.preprocess_num_baths, # num_baths
            DataPreprocessor.preprocess_size_sqft, # size_sqft
            DataPreprocessor.preprocess_floor_level, # floor_level
            DataPreprocessor.preprocess_furnishing, # furnishing
            DataPreprocessor.preprocess_available_unit_types, # available_unit_types
            DataPreprocessor.preprocess_planning_area, # planning_area
        ]

        @DataPreprocessor.timer_func
        def executor(func, *args, **kwargs):
            return func(*args, **kwargs)

        p_bar = tqdm(preprocessing_pipeline)
        log_msg = "Pre-processing Start"
        for preprocess_func in p_bar:
            p_bar.set_description(f"Processed function: {log_msg} & Processing function: {preprocess_func.__name__}")
            df_clean, log_msg = executor(preprocess_func, df_clean, test=test, uncertain=uncertain)
            log_msg = log_msg.replace("executor", preprocess_func.__name__)
            

        if remove_original_attributes:
            drop_attributes.add('title')
            drop_attributes.add('title_property_type')
            drop_attributes.add('title_n_beds')
            drop_attributes.add('title_address')
            drop_attributes.add('address')
            drop_attributes.add('property_name')
            drop_attributes.add('property_type')
            drop_attributes.add('property_type_clean')
            drop_attributes.add('tenure')
            drop_attributes.add('block_number')
            drop_attributes.add('floor_level')
            drop_attributes.add('lat_lowres')
            drop_attributes.add('lng_lowres')
            drop_attributes.add('subzone')
            drop_attributes.add('available_unit_types')
            drop_attributes.add('total_num_units')
            drop_attributes.add('furnishing')
            drop_attributes.add('property_details_url')
            drop_attributes.add('planning_area')
            drop_attributes.add('elevation')
            drop_attributes.add('CR')
            drop_attributes.add('IEBP')
            drop_attributes.add('BN')
            drop_attributes.add('IHL')
            df_clean = df_clean.drop(drop_attributes, axis=1,inplace=False).reset_index(drop=True)
        
        if drop_na:
            df_clean = df_clean.dropna()

        return df_clean

    """ Auxillary Dataset
    """
    @staticmethod
    def compute_dist(lat1:float, lng1:float, lat2:float, lng2:float)->float:
        # approximate radius of earth in km
        R = 6373.0
        lat1 = radians(lat1)
        lng1 = radians(lng1)
        lat2 = radians(lat2)
        lng2 = radians(lng2)
        dlng = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c
    
    @staticmethod
    def _distance_to_closest_centres(row, sg_commercial_centres_df):
        shortest_distances = {
            'CR_dist': np.inf,
            'CR': None,
            'IEBP_dist': np.inf,
            'IEBP': None,
            'BN_dist': np.inf,
            'BN': None,
            'IHL_dist': np.inf,
            'IHL': None,
        }
        for type in sg_commercial_centres_df["type"].unique():
            available_centers = sg_commercial_centres_df.loc[sg_commercial_centres_df["type"]==type, ['lat','lng','name']]
            available_centers['dist'] = available_centers.apply(lambda available_center: DataPreprocessor.compute_dist(row['lat'], row['lng'], available_center['lat'], available_center['lng']), axis=1)
            shortest_distances[f"{type}_dist"] =  available_centers["dist"].min()
            shortest_distances[type] =  available_centers.loc[available_centers["dist"]==available_centers["dist"].min(),'name'].values[0]
        return shortest_distances

    @staticmethod
    def preprocess_commercial_center(df, sg_commercial_centres_df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        sg_commercial_centres_df.loc[sg_commercial_centres_df['type']=='IEPB', 'type'] = 'IEBP'
        unique_location = df_[['lat', 'lng']].drop_duplicates().reset_index(drop=True)
        closest_commercial_centers = unique_location.merge(
            pd.DataFrame(
                list(
                    unique_location.apply(lambda row: DataPreprocessor._distance_to_closest_centres(row, sg_commercial_centres_df), axis=1)
                )), 
            left_index=True, 
            right_index=True
        )
        df_ = df_.merge(closest_commercial_centers, left_on=['lat','lng'], right_on=['lat','lng'])

        for __type in sg_commercial_centres_df["type"].unique():
            print(__type)
            if not test:
                cat_order = df_.groupby(__type).median().sort_values('price').index.to_list()
                commercial_encoder = ENCODER[f"{__type}_encoder"]
                commercial_encoder.set_params(categories=[cat_order])
                commercial_encoder.fit(df_[__type].values.reshape(-1, 1))
                with open(WORKING_DIR/'lib'/f'{__type}_encoder.sav', 'wb') as f:
                    pickle.dump(commercial_encoder, f)
            else:
                with open(WORKING_DIR/'lib'/f'{__type}_encoder.sav', 'rb') as f:
                    commercial_encoder = pickle.load(f)

            commercial_cat = commercial_encoder.transform(df_[__type].values.reshape(-1, 1))
            if type(commercial_cat)!=np.ndarray: commercial_cat = commercial_cat.toarray()
            for feature in range(commercial_cat.shape[1]):
                df_[f'{__type}_cat_{feature}']  = commercial_cat[:, feature]
        return df_

    @staticmethod
    def _distance_and_closest_community(row, ref_df: pd.DataFrame, key: str, radius=0.5):
        ref_keys = {
            f'{key}_dist': np.inf,
            f'{key}_name': None,
            f'number_of_close_{key}':0
        }
        available_items = ref_df.loc[:, ['lat','lng','name']]
        available_items['dist'] = available_items.apply(lambda available_item: DataPreprocessor.compute_dist(row['lat'], row['lng'], available_item['lat'], available_item['lng']), axis=1)
        ref_keys[f'{key}_dist'] =  available_items["dist"].min()
        ref_keys[f'{key}_name'] = available_items.loc[available_items["dist"]==available_items["dist"].min(),'name'].values[0]
        ref_keys[f'number_of_close_{key}'] = available_items.loc[available_items["dist"] <= radius].shape[0]
        return ref_keys

    @staticmethod
    def preprocess_primary_school(df, ref_df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        
        unique_location = df_[['lat', 'lng']].drop_duplicates().reset_index(drop=True)
        key = 'pri_sch'
        closest_primary_school = unique_location.merge(
            pd.DataFrame(
                list(
                    unique_location.apply(lambda row: DataPreprocessor._distance_and_closest_community(
                        row, 
                        ref_df,
                        key=key
                    ), axis=1)
                )), 
            left_index=True, 
            right_index=True
        )
        df_ = df_.merge(closest_primary_school, left_on=['lat','lng'], right_on=['lat','lng'])

        if not test:
            cat_order = df_.groupby(f'{key}_name').median().sort_values('price').index.to_list()
            pri_sch_encoder = ENCODER[f"{key}_encoder"]
            pri_sch_encoder.set_params(categories=[cat_order])
            pri_sch_encoder.fit(df_[f'{key}_name'].values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/f'{key}_encoder.sav', 'wb') as f:
                pickle.dump(pri_sch_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/f'{key}_encoder.sav', 'rb') as f:
                pri_sch_encoder = pickle.load(f)

        pri_sch_cat = pri_sch_encoder.transform(df_[f'{key}_name'].values.reshape(-1, 1))
        if type(pri_sch_cat)!=np.ndarray: pri_sch_cat = pri_sch_cat.toarray()
        for feature in range(pri_sch_cat.shape[1]):
            df_[f'{key}_cat_{feature}']  = pri_sch_cat[:, feature]
        return df_

    @staticmethod
    def preprocess_secondary_school(df, ref_df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        
        unique_location = df_[['lat', 'lng']].drop_duplicates().reset_index(drop=True)
        key = 'sec_sch'
        closest_secondary_school = unique_location.merge(
            pd.DataFrame(
                list(
                    unique_location.apply(lambda row: DataPreprocessor._distance_and_closest_community(
                        row, 
                        ref_df,
                        key=key
                    ), axis=1)
                )), 
            left_index=True, 
            right_index=True
        )
        df_ = df_.merge(closest_secondary_school, left_on=['lat','lng'], right_on=['lat','lng'])

        if not test:
            cat_order = df_.groupby(f'{key}_name').median().sort_values('price').index.to_list()
            sec_sch_encoder = ENCODER[f"{key}_encoder"]
            sec_sch_encoder.set_params(categories=[cat_order])
            sec_sch_encoder.fit(df_[f'{key}_name'].values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/f'{key}_encoder.sav', 'wb') as f:
                pickle.dump(sec_sch_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/f'{key}_encoder.sav', 'rb') as f:
                sec_sch_encoder = pickle.load(f)

        sec_sch_cat = sec_sch_encoder.transform(df_[f'{key}_name'].values.reshape(-1, 1))
        if type(sec_sch_cat)!=np.ndarray: sec_sch_cat = sec_sch_cat.toarray()
        for feature in range(sec_sch_cat.shape[1]):
            df_[f'{key}_cat_{feature}']  = sec_sch_cat[:, feature]
        return df_

    @staticmethod
    def preprocess_shopping_mall(df, ref_df, test=False, uncertain: bool=KEEP_UNCERTAIN) -> pd.DataFrame:
        df_ = df.copy()
        
        unique_location = df_[['lat', 'lng']].drop_duplicates().reset_index(drop=True)
        key = 'shopping_mall'
        closest_shopping_mall = unique_location.merge(
            pd.DataFrame(
                list(
                    unique_location.apply(lambda row: DataPreprocessor._distance_and_closest_community(
                        row, 
                        ref_df,
                        key=key
                    ), axis=1)
                )), 
            left_index=True, 
            right_index=True
        )
        df_ = df_.merge(closest_shopping_mall, left_on=['lat','lng'], right_on=['lat','lng'])

        if not test:
            cat_order = df_.groupby(f'{key}_name').median().sort_values('price').index.to_list()
            shopping_mall_encoder = ENCODER[f"{key}_encoder"]
            shopping_mall_encoder.set_params(categories=[cat_order])
            shopping_mall_encoder.fit(df_[f'{key}_name'].values.reshape(-1, 1))
            with open(WORKING_DIR/'lib'/f'{key}_encoder.sav', 'wb') as f:
                pickle.dump(shopping_mall_encoder, f)
        else:
            with open(WORKING_DIR/'lib'/f'{key}_encoder.sav', 'rb') as f:
                shopping_mall_encoder = pickle.load(f)

        shopping_mall_cat = shopping_mall_encoder.transform(df_[f'{key}_name'].values.reshape(-1, 1))
        if type(shopping_mall_cat)!=np.ndarray: shopping_mall_cat = shopping_mall_cat.toarray()
        for feature in range(shopping_mall_cat.shape[1]):
            df_[f'{key}_cat_{feature}']  = shopping_mall_cat[:, feature]
        return df_

    # @staticmethod
    # def data_preprocessing_v2(
    #     df: pd.DataFrame, 
    #     test:bool=False, 
    #     uncertain:bool=False, 
    #     drop_na:bool=False, 
    #     remove_original_attributes:bool=True
    # ) -> pd.DataFrame:
    #     df_clean = df.copy()
    #     drop_attributes = {'listing_id'}

    #     preprocessing_pipeline = [
    #         DataPreprocessor.remove_price_outlier, # Excessive outliers
    #         DataPreprocessor.remove_duplicates, # Duplicated records
    #         DataPreprocessor.preprocess_title, # extract features form title
    #         DataPreprocessor.preprocess_lat_lng, # correct lat and lng
    #         DataPreprocessor.preprocess_property_type, # property_type
    #         DataPreprocessor.preprocess_tenure, # tenure
    #         DataPreprocessor.preprocess_built_year, # built_year -  unfinished!
    #         DataPreprocessor.preprocess_num_beds, # num_beds
    #         DataPreprocessor.preprocess_num_baths, # num_baths
    #         DataPreprocessor.preprocess_size_sqft, # size_sqft
    #         DataPreprocessor.preprocess_floor_level, # floor_level
    #         DataPreprocessor.preprocess_furnishing, # furnishing
    #         DataPreprocessor.preprocess_available_unit_types, # available_unit_types
    #         DataPreprocessor.preprocess_planning_area, # planning_area
    #     ]

    #     @DataPreprocessor.timer_func
    #     def executor(func, *args, **kwargs):
    #         return func(*args, **kwargs)

    #     p_bar = tqdm(preprocessing_pipeline)
    #     log_msg = "Pre-processing Start"
    #     for preprocess_func in p_bar:
    #         p_bar.set_description(f"Processed function: {log_msg} & Processing function: {preprocess_func.__name__}")
    #         df_clean, log_msg = executor(preprocess_func, df_clean, test=test, uncertain=uncertain)
    #         log_msg = log_msg.replace("executor", preprocess_func.__name__)
            

    #     if remove_original_attributes:
    #         drop_attributes.add('title')
    #         drop_attributes.add('title_property_type')
    #         drop_attributes.add('title_n_beds')
    #         drop_attributes.add('title_address')
    #         drop_attributes.add('address')
    #         drop_attributes.add('property_name')
    #         drop_attributes.add('property_type')
    #         drop_attributes.add('property_type_clean')
    #         drop_attributes.add('tenure')
    #         drop_attributes.add('block_number')
    #         drop_attributes.add('floor_level')
    #         drop_attributes.add('lat_lowres')
    #         drop_attributes.add('lng_lowres')
    #         drop_attributes.add('subzone')
    #         drop_attributes.add('available_unit_types')
    #         drop_attributes.add('total_num_units')
    #         drop_attributes.add('furnishing')
    #         drop_attributes.add('property_details_url')
    #         drop_attributes.add('planning_area')
    #         drop_attributes.add('elevation')
    #         drop_attributes.add('CR')
    #         drop_attributes.add('IEBP')
    #         drop_attributes.add('BN')
    #         drop_attributes.add('IHL')
    #         df_clean = df_clean.drop(drop_attributes, axis=1,inplace=False).reset_index(drop=True)
        
    #     if drop_na:
    #         df_clean = df_clean.dropna()

    #     return df_clean