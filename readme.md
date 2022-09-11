# CS5228 project

## pre-request for working locally
1. install kaggle to env: pip install kaggle
2. follow this: https://adityashrm21.github.io/Setting-Up-Kaggle/

## project file structure
~~~
project
    |-  input
    |-  lib
    |-  working
    |-  src
        |- data_preprocess
~~~
## download dataset
~~~
kaggle competitions download -c cs5228-2022-semester-1-final-project -p input
~~~

## Task1: pre-process attributes
### add & modify your pre-process method under:
    ~~~
    src/data_preprocess

    class DataPreprocessor:
        @staticmethod
        def preprocess_{feature_name}(df) -> pd.DataFrame:
            df_ = df.copy()
            # your algorithms
            return df_
    ~~~

### objectives
1. handling missing data
2. transform cat data
3. drop the useless columns