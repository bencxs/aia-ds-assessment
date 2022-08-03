import pandas as pd
import numpy as np
import os

DATA_DIR = '../assessment_data/'
ARTIFACTS_DIR = '../artifacts/'

def create_base():

    def main_need(row):
        if row['S_BASE'] == 'POLA':
            return 'Accident'
        elif row['D'] != None:
            return row['D']
        elif row['E'] != None:
            return row['E']
        elif row['B'] != None and row['B'] != 'Medical':
            return row['B']
        elif row['B'] != None and row['B'] == 'Medical':
            return 'Other Medical'
        elif row['C'] != None and row['C'] != 'Medical':
            return row['C']
        elif row['C'] != None and row['C'] == 'Medical':
            return 'Other Medical'
        
    def product_category(row):
        if row['PROD_CAT'] != None:
            return row['PROD_CAT']
        elif row['PROD_SUBCAT_IND_1'] == 1 and (
            row['PROD_SUBCAT_IND_2'] == 1 or row['MAIN_NEED'] in ['Medical', 'Other Medical', 'Critical Illness']
            ):
            return 'Cat 2'
        else:
            return 'Cat 3'

    df_main = pd.read_csv(os.path.join(DATA_DIR, 'MAIN.csv'), 
                        dtype={
                            'POLICY_NO': 'object',
                            'AGT_CD': 'object',
                            'S_BASE': 'object',
                            'RIDER_CD': 'object',
                            'COVERAGE_TYPE_IND': 'object',
                            'INS_UNIQUE_ID': 'object',
                            'PROD_CAT': 'object',
                            'PRODUCT_CD': 'object',
                            'ANP_RIDER': np.float32,
                            'VIT_MBR_IND': np.int8,
                            'PROD_SUBCAT_IND_1': np.int8,
                            'PROD_SUBCAT_IND_2': np.int8,
                            })
    df_main['PURCHASE_DT'] = pd.to_datetime(df_main['PURCHASE_DT'], format='%Y-%m-%d')

    df_agent = pd.read_csv(os.path.join(DATA_DIR, 'AGENT.csv'),
                        dtype={
                            'AGT_CHANNEL': 'object',
                            'AGT_STATUS': 'object',
                            'AGT_CD': 'object',
                        })

    df_b = pd.read_csv(os.path.join(DATA_DIR, 'B.csv'),
                    dtype={
                            'PURPOSE_CODE': 'object',
                            'CHAR_VALUE': 'object',
                            'CHAR_VALUE2': 'object',
                    })

    df_c = pd.read_csv(os.path.join(DATA_DIR, 'C.csv'),
                        dtype={
                            'PURPOSE_CODE': 'object',
                            'CHAR_VALUE': 'object',
                            'CHAR_VALUE2': 'object',
                    })

    df_d = pd.read_csv(os.path.join(DATA_DIR, 'D.csv'),
                        dtype={
                            'PURPOSE_CODE': 'object',
                            'CHAR_VALUE': 'object',
                            'CHAR_VALUE2': 'object',
                    })

    df_e = pd.read_csv(os.path.join(DATA_DIR, 'E.csv'),
                        dtype={
                            'PURPOSE_CODE': 'object',
                            'CHAR_VALUE': 'object',
                            'CHAR_VALUE2': 'object',
                    })

    # Filter to Inforce and Agency agent base
    df_agent_1 = df_agent[(df_agent['AGT_STATUS'] == 'INFORCE') & (df_agent['AGT_CHANNEL'] == 'AGENCY')]

    # Join with df_main
    df_1 = df_main.merge(df_agent_1, on='AGT_CD', how='inner')

    # Join with all supplementary datasets
    df_2 = df_1.merge(df_b,
                        left_on=['PRODUCT_CD', 'S_BASE'],
                        right_on=['PURPOSE_CODE', 'CHAR_VALUE2'],
                        how='left')
    df_2.drop(['PURPOSE_CODE', 'CHAR_VALUE2'], axis=1, inplace=True)
    df_2.rename(columns={'CHAR_VALUE': 'B'}, inplace=True)

    df_2 = df_2.merge(df_c,
                        left_on=['PRODUCT_CD'],
                        right_on=['PURPOSE_CODE'],
                        how='left')
    df_2.drop(['PURPOSE_CODE', 'CHAR_VALUE2'], axis=1, inplace=True)
    df_2.rename(columns={'CHAR_VALUE': 'C'}, inplace=True)

    df_2 = df_2.merge(df_d,
                        left_on=['PRODUCT_CD', 'S_BASE'],
                        right_on=['PURPOSE_CODE', 'CHAR_VALUE2'],
                        how='left')
    df_2.drop(['PURPOSE_CODE', 'CHAR_VALUE2'], axis=1, inplace=True)
    df_2.rename(columns={'CHAR_VALUE': 'D'}, inplace=True)

    df_2 = df_2.merge(df_e,
                        left_on=['PRODUCT_CD'],
                        right_on=['PURPOSE_CODE'],
                        how='left')
    df_2.drop(['PURPOSE_CODE', 'CHAR_VALUE2'], axis=1, inplace=True)
    df_2.rename(columns={'CHAR_VALUE': 'E'}, inplace=True)

    # Main Need
    df_2 = df_2.replace(np.nan, None)
    cols = ['B', 'C', 'D', 'E']
    df_2[cols] = df_2[cols].apply(lambda x: x.str.strip())
    df_2['MAIN_NEED'] = df_2.apply(lambda x: main_need(x), axis=1)

    # Product Category
    df_2['PROD_CAT'] = df_2.apply(lambda x: product_category(x), axis=1)
    
    # For validation
    df_2.to_csv(os.path.join(ARTIFACTS_DIR, 'df_base.csv'), index=False)
    
    return df_2

if __name__ == '__main__':
    create_base()