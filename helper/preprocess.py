import pandas as pd
import numpy as np
import os

DATA_DIR = '../assessment_data/'
ARTIFACTS_DIR = '../artifacts/'

def preprocess_agent_level(df_base_1):
    df_agt_1 = (df_base_1.groupby(['AGT_CD', 'S_BASE'])
        .agg(
            N_POLICIES=('POLICY_NO', 'size'),
            TOTAL_ANP=('ANP_RIDER', 'sum'),
            )
        .reset_index()
    )
    df_agt_1 = df_agt_1.pivot(index='AGT_CD', columns='S_BASE', values=['N_POLICIES', 'TOTAL_ANP']).reset_index()

    # Flatten pivot columns
    df_agt_1.columns = ['_'.join(col).rstrip('_') for col in df_agt_1.columns.to_flat_index()]

    # Derive additional features
    df_agt_1['TOTAL_POLICIES'] = df_agt_1[['N_POLICIES_LA', 'N_POLICIES_LAT', 'N_POLICIES_POLA']].sum(axis=1)
    df_agt_1['TOTAL_ANP'] = df_agt_1[['TOTAL_ANP_LA', 'TOTAL_ANP_LAT', 'TOTAL_ANP_POLA']].sum(axis=1)
    df_agt_1['TICKET_SIZE'] = df_agt_1['TOTAL_ANP'] / df_agt_1['TOTAL_POLICIES']
    df_agt_1['COUNT_RATIO_LA'] = df_agt_1['N_POLICIES_LA'] / df_agt_1['TOTAL_POLICIES']
    df_agt_1['COUNT_RATIO_LAT'] = df_agt_1['N_POLICIES_LAT'] / df_agt_1['TOTAL_POLICIES']
    df_agt_1['COUNT_RATIO_POLA'] = df_agt_1['N_POLICIES_POLA'] / df_agt_1['TOTAL_POLICIES']
    df_agt_1['ANP_RATIO_LA'] = df_agt_1['TOTAL_ANP_LA'] / df_agt_1['TOTAL_ANP']
    df_agt_1['ANP_RATIO_LAT'] = df_agt_1['TOTAL_ANP_LAT'] / df_agt_1['TOTAL_ANP']
    df_agt_1['ANP_RATIO_POLA'] = df_agt_1['TOTAL_ANP_POLA'] / df_agt_1['TOTAL_ANP']
    df_agt_1['COUNT_SBASE_TYPE'] = df_agt_1[['N_POLICIES_LA', 'N_POLICIES_LAT', 'N_POLICIES_POLA']].count(axis=1)

    df_agt_2 = (df_base_1.groupby(['AGT_CD', 'COVERAGE_TYPE_IND'])
            .agg(
                N_LINE_ITEM=('POLICY_NO', 'size'),
                TOTAL_ANP=('ANP_RIDER', 'sum'),
                )
            .reset_index()
    )
    df_agt_2 = df_agt_2.pivot(index='AGT_CD', columns='COVERAGE_TYPE_IND', values=['N_LINE_ITEM', 'TOTAL_ANP']).reset_index()

    # Flatten pivot columns
    df_agt_2.columns = ['_'.join(col).rstrip('_') for col in df_agt_2.columns.to_flat_index()]

    # Derive additional features
    df_agt_2['R_B_COUNT_RATIO'] = df_agt_2['N_LINE_ITEM_R'] / df_agt_2['N_LINE_ITEM_B']
    df_agt_2['R_B_ANP_RATIO'] = df_agt_2['TOTAL_ANP_R'] / df_agt_2['TOTAL_ANP_B']


    df_agt_3 = (df_base_1.groupby(['AGT_CD', 'PROD_CAT'])
            .agg(
                N_LINE_ITEM=('POLICY_NO', 'size'),
                TOTAL_ANP=('ANP_RIDER', 'sum'),
                )
            .reset_index()
    )
    df_agt_3 = df_agt_3.pivot(index='AGT_CD', columns='PROD_CAT', values=['N_LINE_ITEM', 'TOTAL_ANP']).reset_index()

    # Flatten pivot columns
    df_agt_3.columns = ['_'.join(col).rstrip('_') for col in df_agt_3.columns.to_flat_index()]

    # Derive additional features
    df_agt_3['TOTAL_LINE_ITEM'] = df_agt_3[['N_LINE_ITEM_Cat 1', 'N_LINE_ITEM_Cat 2', 'N_LINE_ITEM_Cat 3']].sum(axis=1)
    df_agt_3['TOTAL_LINE_ITEM_ANP'] = df_agt_3[['TOTAL_ANP_Cat 1', 'TOTAL_ANP_Cat 2', 'TOTAL_ANP_Cat 3']].sum(axis=1)
    df_agt_3['COUNT_RATIO_CAT1'] = df_agt_3['N_LINE_ITEM_Cat 1'] / df_agt_3['TOTAL_LINE_ITEM']
    df_agt_3['COUNT_RATIO_CAT2'] = df_agt_3['N_LINE_ITEM_Cat 2'] / df_agt_3['TOTAL_LINE_ITEM']
    df_agt_3['COUNT_RATIO_CAT3'] = df_agt_3['N_LINE_ITEM_Cat 3'] / df_agt_3['TOTAL_LINE_ITEM']
    df_agt_3['ANP_RATIO_CAT1'] = df_agt_3['TOTAL_ANP_Cat 1'] / df_agt_3['TOTAL_LINE_ITEM_ANP']
    df_agt_3['ANP_RATIO_CAT2'] = df_agt_3['TOTAL_ANP_Cat 2'] / df_agt_3['TOTAL_LINE_ITEM_ANP']
    df_agt_3['ANP_RATIO_CAT3'] = df_agt_3['TOTAL_ANP_Cat 3'] / df_agt_3['TOTAL_LINE_ITEM_ANP']
    df_agt_3['COUNT_PRODCAT_TYPE'] = df_agt_3[['N_LINE_ITEM_Cat 1', 'N_LINE_ITEM_Cat 2', 'N_LINE_ITEM_Cat 3']].count(axis=1)

    df_agt_3.drop(['TOTAL_LINE_ITEM', 'TOTAL_LINE_ITEM_ANP'], axis=1, inplace=True)
    
    df_base_vit = df_base_1[df_base_1['VIT_MBR_IND'] == 1]
    df_agt_4 = (df_base_vit.groupby(['AGT_CD'])
            .agg(
                N_VIT_POLICIES=('POLICY_NO', 'nunique'),
                )
            .reset_index()
    )
    
    df_base_1['MAIN_NEED'].fillna('N/A', inplace=True)
    df_agt_5 = (df_base_1.groupby(['AGT_CD', 'MAIN_NEED'])
            .agg(
                N_LINE_ITEM=('POLICY_NO', 'size'),
                TOTAL_ANP=('ANP_RIDER', 'sum'),
                )
            .reset_index()
    )
    df_agt_5 = df_agt_5.pivot(index='AGT_CD', columns='MAIN_NEED', values=['N_LINE_ITEM', 'TOTAL_ANP']).reset_index()

    # Flatten pivot columns
    df_agt_5.columns = ['_'.join(col).rstrip('_') for col in df_agt_5.columns.to_flat_index()]
    
    # Derive additional features
    df_agt_5['COUNT_MAINNEED_TYPE'] = (df_agt_5[[
        'N_LINE_ITEM_Accident', 'N_LINE_ITEM_Critical Illness',
       'N_LINE_ITEM_Disability', 'N_LINE_ITEM_Endowment', 'N_LINE_ITEM_Income',
       'N_LINE_ITEM_Life', 'N_LINE_ITEM_Medical',
       'N_LINE_ITEM_Other Medical', 'N_LINE_ITEM_Payor/Waiver',
       'N_LINE_ITEM_Savers']]).count(axis=1)

    # Merge all agent level df into one
    df_agt_f = df_agt_1.merge(df_agt_2, on='AGT_CD', how='left')
    df_agt_f = df_agt_f.merge(df_agt_3, on='AGT_CD', how='left')
    df_agt_f = df_agt_f.merge(df_agt_4, on='AGT_CD', how='left')
    df_agt_f = df_agt_f.merge(df_agt_5, on='AGT_CD', how='left')
    
    df_agt_f.to_csv(os.path.join(ARTIFACTS_DIR, 'df_agt_f.csv'), index=False)
    
    return df_agt_f

if __name__ == '__main__':
    preprocess_agent_level()