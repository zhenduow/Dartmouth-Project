import datetime
import time
import statistics as stat
import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np

DIAGNOSES_PATH = ".\\IRB_90679_Chapman_AMI_diagnoses_12132018.txt"
LABS_PATH = ".\\IRB_90679_Chapman_AMI_labs_12132018.txt"
MED_ADMIN_PATH = ".\\IRB_90679_Chapman_AMI_med_admin_12132018.txt"
MED_ORDERS_PATH = ".\\IRB_90679_Chapman_AMI_med_orders_12132018.txt"
PROCEDURES_PATH = ".\\IRB_90679_Chapman_AMI_procedures_12132018.txt"
VISITS_W_PROV_TYPE_PATH = ".\\IRB_90679_Chapman_AMI_visits_w_prov_type_12132018.txt"
DEMOGRAPHICS_PATH = ".\\IRB_90679_Chapman_AMI_demographics_12132018.txt"
TARGET_PATH = '.\\target.csv'

ACUTE_MYOCARDIAL_INFARCTION_CODE = ['410.00','410.01','410.10','410.11','410.20','410.21','410.30','410.31',
                                    '410.40','410.41','410.50','410.51','410.60','410.61','410.70','410.71',
                                    '410.80','410.81','410.90','410.91','I21.09','I21.11','I21.19','I21.29',
                                    'I21.3','I21.4']

CHEST_PAIN_CODE = ['786.5', 'R07.9']
CARDIAC_ARREST_CODE = ['427.5', 'I46.9']
CLOPIDOGREL_NAMES = ['Plavix',
                     'PLAVIX 75 MG PO TABLET',
                     'PLAVIX PO',
                     'Plavix 300 mg TAB UD [CLOP300TU]',
                     'CLOPIDOGREL BISULFATE 75 MG PO TABLET',
                     'CLOPIDOGREL BISULFATE 37.5 MG PO HALF TAB (INPATIENT USE ONLY)',
                     'CLOPIDOGREL BISULFATE 300 MG PO TABLET',
                     'CLOPIDOGREL BISULFATE PO',
                     'clopidogrel'
                     ]

AMI_LOCATION = {
    '410.00': 'Anterolateral wall',
    '410.01': 'Anterolateral wall',
    '410.10': 'Other anterior wall',
    'I21.09': 'Other anterior wall',
    '410.11': 'Anterior wall',
    'I21.0': 'Anterior wall',
    'I21.02': 'Anterior wall',
    '410.20': 'Inferolateral wall',
    '410.21': 'Inferolateral wall',
    '410.30': 'Inferoposterior wall',
    '410.31': 'Inferoposterior wall',
    'I21.11': 'Inferoposterior wall',
    '410.40': 'Other inferior wall',
    '410.41': 'Other inferior wall',
    'I21.19': 'Other inferior wall',
    '410.50': 'Other lateral wall',
    '410.51': 'Other lateral wall',
    '410.60': 'True posterior wall infarction initial episode of care',
    '410.61': 'True posterior wall infarction initial episode of care',
    '410.70': 'Subendocardial infarction',
    '410.71': 'Subendocardial infarction',
    'I21.4' : 'Subendocardial infarction',
    '410.80': 'Other specified sites',
    '410.81': 'Other specified sites',
    'I21.29': 'Other specified sites',
    'I21.2' : 'Other specified sites',
    '410.90' : 'Unspecified site',
    '410.91' : 'Unspecified site',
    'I21.3' : 'Unspecified site'
}

PRIOR_SEPSIS_30D_CODE = ['995.91', 'A41.9']
PRIOR_HYPERKALEMIA_30D_CODE = ['276.7', 'E87.5']
PRIOR_HYPOKALEMIA_30D_CODE = ['276.8', 'E87.6']
PRIOR_HYPERVOLEMIA_30D_CODE = ['276.61', 'E87.71']
PRIOR_AKF_30D_CODE = ['584', 'N17', 'N17.0', 'N17.1', 'N17.2', 'N17.8', 'N17.9']
PRIOR_UTI_30D_CODE = ['599', 'N39.0']
PRIOR_LONGTERM_ANTICOAGULANTS_30D_CODE = ['V58.61', 'Z79.01']
PRIOR_DIS_MAGN_METAB_90D_CODE = ['275.2', 'E83.40', 'E83.41', 'E83.42', 'E83.49']
PRIOR_LVEF_90D_CODE = ['428.1', 'I50.1']
PRIOR_CARDIAC_DEVICE_90D_CODE = ['V45.00', 'Z95.9']

REVASCULARIZATION_CODE = ['36.19','36.31','92937','92941','92943','92944']
ONE_VESSEL_CODE = ['92937','92941','92943']
ADDITIONAL_VESSEL_CODE = '92944'

CHEST_PAIN_NAMES = ['CHEST PAIN ON BREATHING', 'CHEST PAIN UNSPECIFIED', 'OTHER CHEST PAIN']
FAMILY_DEPRESSION_CODE = ['Z81.8', 'V17.0']

ECHOCARDIOGRAPHY_CODE = ['INTRACARDIAC ECHOCARDIOGRAPHY', 'Catheter, intracardiac echocardiography']

CHF_CODE = ['CHRONIC COMBINED SYSTOLIC AND DIASTOLIC CHF',
            'ACUTE ON CHRONIC COMB SYSTOLIC & DIASTOLIC CHF',
            'UNSPECIFIED COMBINED SYSTOLIC & DIASTOLIC CHF',
            'ACUTE COMBINED SYSTOLIC AND DIASTOLIC CHF']

def get_date_diff(discharge_date, admit_date):
    '''
    give two date string in the format YYYY-MM-DD, return the difference of first - second
    :param discharge_date: discharge_date
    :param admit_date: admit_date
    :return: The time difference
    '''
    try:
        month_d, day_d, year_d = discharge_date.split('/')
    except:
        try:
            month_d, day_d, year_d = discharge_date.split('-')
        except:
            return -1
    try:
        month_a, day_a, year_a = admit_date.split('/')
    except:
        try:
            month_a, day_a, year_a = admit_date.split('-')
        except:
            return -1
    d_date = datetime.date(int(year_d), int(month_d), int(day_d))
    a_date = datetime.date(int(year_a), int(month_a), int(day_a))
    d = d_date - a_date
    return d.days # less than one day does not count as one day

def get_LOS5_FLAG(LOS):
    '''
    If the length of stay >=5,
    then this value is set to one.
    Otherwise, this value is set to zero.
    :param LOS:
    :return:
    '''
    return 1 if LOS>5 else 0

def datecmp(date1, date2):
    '''
    compare two dates, if the first one is earlier than the second, return true, else return false
    :param date1:
    :param date2:
    :return:
    '''
    try:
        month_1, day_1, year_1 = date1.split('/')
    except:
        try:
            month_1, day_1, year_1 = date1.split('-')
        except:
            return False
    try:
        month_2, day_2, year_2 = date2.split('/')
    except:
        try:
            month_2, day_2, year_2 = date2.split('-')
        except:
            return False
    date_1 = datetime.date(int(year_1), int(month_1), int(day_1))
    date_2 = datetime.date(int(year_2), int(month_2), int(day_2))
    return (date_1 - date_2).days < 0

def remove_quotes(content):
    '''
    Remove quotation marks
    :param content:
    :return:
    '''
    text = str(content)
    return text.replace('"','')

def get_one_value_by_foreign_key(df, foreign_key, foreign_value, target_key):
    '''
    find the single value of target_key in df with foreigh_key equals foreign_value
    :param df:
    :param foreign_key:
    :param foreign_value:
    :param target_key:
    :return:
    '''
    return df[df[foreign_key] == foreign_value][target_key].values[0]\
        if len(df[df[foreign_key] == foreign_value]) > 0 else ''

def get_values_by_foreign_key(df, foreign_key, foreign_value, target_key):
    '''
    find all values of target_key in df with foreigh_key equals foreign_value
    :param df:
    :param foreign_key:
    :param foreign_value:
    :param target_key:
    :return:
    '''
    return list(df[df[foreign_key] == foreign_value][target_key].values)\
        if len(df[df[foreign_key] == foreign_value]) > 0 else ''

def try_float(s):
    try:
        return float(s)
    except:
        return 0

def DEMOGRAPHICS(target_df, demographics_df, diagnoses_df):
    '''
    Fill the demographics information for each patient.
    :param target_df:
    :param demographics_df:
    :return:
    '''
    # initialize table columns
    target_df['MRN'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['GENDER'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'PAT_GENDER')
                           for patient in target_df['PERSON_ID'].values]
    target_df['RACE'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['ETHNICITY'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['SSN'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'SSN')
                        for patient in target_df['PERSON_ID'].values]
    target_df['ZIPCODE'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'ZIP')
                            for patient in target_df['PERSON_ID'].values]
    target_df['FIRST_NAME'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'FIRST_NAME')
                               for patient in target_df['PERSON_ID'].values]
    target_df['LAST_NAME'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'LAST_NAME')
                              for patient in target_df['PERSON_ID'].values]
    target_df['MIDDLE_NAME'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'MIDDLE_NAME')
                                for patient in target_df['PERSON_ID'].values]
    target_df['DOB'] = [get_one_value_by_foreign_key(demographics_df, 'PAT_ID', patient, 'DOB')
                        for patient in target_df['PERSON_ID'].values]
    target_df['PRIM_DIAG'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['ADMIT_DATE'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['DISCHARGE_DATE'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['INDEX_ADMIT_DATE'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['INDEX_DISCHARGE_DATE'] = ['' for patient in target_df['PERSON_ID'].values]
    target_df['VISIT_OCCURRENCE_ID'] = ['' for patient in target_df['PERSON_ID'].values]

    # get prim_diagnoses code version
    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_visits_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        all_code_versions = list(all_visits_df['CODE_VERSION'].values)
        target_df.at[iter, 'PRIM_DIAG'] = 'ICD9CM' if 'ICD9CM' in all_code_versions else 'ICD-10-CM'

    # get dates and visit_no
    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_visits_df = demographics_df[demographics_df['PAT_ID'] == pat_id]
        all_adm_dates = list(all_visits_df['AMI_ADM_DATE'].values)
        all_dsch_dates = list(all_visits_df['AMI_DSCH_DATE'].values)
        all_visit_no = list(all_visits_df['VISIT_NO'].values)

        # get the earlist adm and dsch date
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_dsch_dates,all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m-%d-%Y'))
            adm_date, dsch_date,visit = date_pairs[0]
            target_df.at[iter, 'ADMIT_DATE'] = adm_date
            target_df.at[iter, 'DISCHARGE_DATE'] = dsch_date
            target_df.at[iter, 'INDEX_ADMIT_DATE'] = adm_date
            target_df.at[iter, 'INDEX_DISCHARGE_DATE'] = dsch_date
            target_df.at[iter, 'VISIT_OCCURRENCE_ID'] = visit

    return target_df

def PRIOR_MONTH_DIAGNOSIS(target_df, diagnoses_df):
    '''
    generate PRIOR_MONTH_DIAGNOSIS variables
    :param target_df:
    :param diagnose_df:
    :return:
    '''
    # initialize table columns with default values
    target_df['PRIOR_SEPSIS_30D'] = [0] * len(target_df)
    target_df['PRIOR_HYPERKALEMIA_30D'] = [0] * len(target_df)
    target_df['PRIOR_HYPOKALEMIA_30D'] = [0] * len(target_df)
    target_df['PRIOR_HYPERVOLEMIA_30D'] = [0] * len(target_df)
    target_df['PRIOR_AKF_30D'] = [0] * len(target_df)
    target_df['PRIOR_UTI_30D'] = [0] * len(target_df)
    target_df['PRIOR_LONGTERM_ANTICOAGULANTS_30D'] = [0] * len(target_df)
    target_df['PRIOR_SEPSIS_90D'] = [0] * len(target_df)
    target_df['PRIOR_DIS_MAGN_METAB_90D'] = [0] * len(target_df)
    target_df['PRIOR_HYPOKALEMIA_90D'] = [0] * len(target_df)
    target_df['PRIOR_LVEF_90D'] = [0] * len(target_df)
    target_df['PRIOR_AKF_90D'] = [0] * len(target_df)
    target_df['PRIOR_CARDIAC_DEVICE_90D'] = [0] * len(target_df)

    # for each patient in the target table
    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_visits_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_visits_df[all_visits_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        PRIOR_SEPSIS_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_SEPSIS_30D_CODE)]
        PRIOR_HYPERKALEMIA_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_HYPERKALEMIA_30D_CODE)]
        PRIOR_HYPOKALEMIA_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_HYPOKALEMIA_30D_CODE)]
        PRIOR_HYPERVOLEMIA_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_HYPERVOLEMIA_30D_CODE)]
        PRIOR_AKF_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_AKF_30D_CODE)]
        PRIOR_UTI_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_UTI_30D_CODE)]
        PRIOR_LONGTERM_ANTICOAGULANTS_30D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_LONGTERM_ANTICOAGULANTS_30D_CODE)]
        PRIOR_DIS_MAGN_METAB_90D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_DIS_MAGN_METAB_90D_CODE)]
        PRIOR_LVEF_90D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_DIS_MAGN_METAB_90D_CODE)]
        PRIOR_CARDIAC_DEVICE_90D_df = all_visits_df[all_visits_df['CODE'].isin(PRIOR_DIS_MAGN_METAB_90D_CODE)]

        if len(ACUTE_MYOCARDIAL_INFARCTION_df) > 0:
            # get the earlist ACUTE_MYOCARDIAL_VISIT date
            all_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
            all_dates.sort(key=lambda all_dates: time.strptime(all_dates, '%m/%d/%Y'))
            earliest_date = all_dates[0]

            # check if the date of the visit type we care is within 30 days
            # repeat for all visit type and 90 days.
            for date in list(PRIOR_SEPSIS_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if  time_diff <= 30 and time_diff >= 0:
                    target_df.at[iter,'PRIOR_SEPSIS_30D'] = 1
                if  time_diff <= 90 and time_diff >= 0:
                    target_df.at[iter,'PRIOR_SEPSIS_90D'] = 1

            for date in list(PRIOR_HYPERKALEMIA_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 30 and time_diff >=0 :
                    target_df.at[iter, 'PRIOR_HYPERKALEMIA_30D'] = 1

            for date in list(PRIOR_HYPOKALEMIA_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if 30 >= time_diff >= 0:
                    target_df.at[iter, 'PRIOR_HYPOKALEMIA_30D'] = 1
                if time_diff <= 90 and time_diff >= 0:
                    target_df.at[iter, 'PRIOR_HYPOKALEMIA_90D'] = 1

            for date in list(PRIOR_HYPERVOLEMIA_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <=30 and time_diff >= 0:
                    target_df.at[iter, 'PRIOR_HYPERVOLEMIA_30D'] = 1

            for date in list(PRIOR_AKF_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 30 and time_diff >= 0:
                    target_df.at[iter, 'PRIOR_AKF_30D'] = 1
                if time_diff <= 90 and time_diff >= 0:
                    target_df.at[iter, 'PRIOR_AKF_90D'] = 1

            for date in list(PRIOR_UTI_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 30 and time_diff >= 0:
                    target_df.at[iter, 'PRIOR_UTI_30D'] = 1

            for date in list(PRIOR_LONGTERM_ANTICOAGULANTS_30D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 30 and time_diff >=0 :
                    target_df.at[iter, 'PRIOR_LONGTERM_ANTICOAGULANTS_30D'] = 1

            for date in list(PRIOR_DIS_MAGN_METAB_90D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 90 and time_diff >=0 :
                    target_df.at[iter, 'PRIOR_DIS_MAGN_METAB_90D'] = 1

            for date in list(PRIOR_LVEF_90D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 90 and time_diff >=0 :
                    target_df.at[iter, 'PRIOR_LVEF_90D'] = 1

            for date in list(PRIOR_CARDIAC_DEVICE_90D_df['ADM_DATE'].values):
                time_diff = get_date_diff(earliest_date, date)
                if time_diff <= 90 and time_diff >=0 :
                    target_df.at[iter, 'PRIOR_CARDIAC_DEVICE_90D'] = 1

    return target_df

def HOSPITAL_SCORE(target_df, procedures_df, diagnoses_df, visits_df, labs_df):
    '''
    Compute hospital score and make a new column for it in target_df
    :param target_df:
    :param diagnose_df:
    :return:
    '''

    def get_ONCOLOGY_SERVICE_FLAG(code_list):
        for code in code_list:
            #print(code)
            if '140' <= str(code) < '240':
                return 1
            elif 'C00' <= str(code) < 'C97':
                return 1
            elif 'D00' <= str(code) < 'D50':
                return 1
            elif 'C7A' <= str(code) < 'C7C':
                return 1
            elif 'D3A' <= str(code) < 'D3B':
                return 1
        return 0

    # initialize table columns with default values
    target_df['LOS'] = [0] * len(target_df)
    target_df['LOS5_FLAG'] = [0] * len(target_df)
    target_df['PROCEDURE_FLAG'] = [0] * len(target_df)
    target_df['PRIOR_YEAR_ADMISSIONS_COUNT'] = [0] * len(target_df)
    target_df['NONELECTIVE_ADMISSION_FLAG'] = [0] * len(target_df)
    target_df['ONCOLOGY_SERVICE_FLAG'] = [0] * len(target_df)
    target_df['HEMOGLOBIN_LEVEL_LAST_12_FLAG'] = [0] * len(target_df)
    target_df['SODIUM_LEVEL_LAST_135_FLAG'] = [0] * len(target_df)
    target_df['HOSPITAL_SCORE'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_visits_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_visits_df[all_visits_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_no
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates,all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]
            target_df.at[iter, 'VISIT_NO'] = earliest_visit_no

            # compute LOS and LOS5_FLAG
            earliest_dsch_date = get_one_value_by_foreign_key(visits_df, 'VISIT_NO', earliest_visit_no, 'DSCH_DATE')
            target_df.at[iter, 'LOS'] = get_date_diff(earliest_dsch_date, earliest_adm_date) + 1
            target_df.at[iter, 'LOS5_FLAG'] = get_LOS5_FLAG(target_df.at[iter, 'LOS'])

            # compute PROCEDURE_FLAG
            all_procedures_df = procedures_df[procedures_df['PAT_ID'] == pat_id]
            procedure_flag = False
            for p_iter, p_row in all_procedures_df.iterrows():
                procedure_flag |= datecmp(earliest_adm_date, p_row['PROC_DT']) and datecmp(p_row['PROC_DT'], earliest_dsch_date)
            target_df.at[iter, 'PROCEDURE_FLAG'] = 1 if procedure_flag else 0

            # compute PRIOR_YEAR_ADMISSIONS_COUNT
            total_prior_year_admission = 0
            all_admissions_df = visits_df[visits_df['PAT_ID'] == pat_id]
            for adm_iter, adm_row in all_admissions_df.iterrows():
                if 0 < get_date_diff(earliest_adm_date, adm_row['DSCH_DATE']) <= 365:
                    total_prior_year_admission += 1
            target_df.at[iter, 'PRIOR_YEAR_ADMISSIONS_COUNT'] = total_prior_year_admission

            # compute NONELECTIVE_ADMISSION_FLAG
            pat_visits = visits_df[visits_df['PAT_ID'] == pat_id]
            emergency_visit_dates = get_values_by_foreign_key(pat_visits, 'PAT_CLASS', 'EMERGENCY', 'ADM_DATE')
            n_flag = 0
            for date in emergency_visit_dates:
                if 0 <= get_date_diff(earliest_adm_date, date) <= 1:
                    n_flag = 1
            target_df.at[iter, 'NONELECTIVE_ADMISSION_FLAG'] = n_flag

            # compute ONCOLOGY_SERVICE_FLAG
            target_df.at[iter, 'ONCOLOGY_SERVICE_FLAG'] = get_ONCOLOGY_SERVICE_FLAG(
                set(get_values_by_foreign_key(diagnoses_df, 'VISIT_NO', earliest_visit_no, 'CODE'))
            )

            # compute HEMOGLOBIN_LEVEL_LAST_12_FLAG
            all_labs_df = labs_df[labs_df['VISIT_NO'] == earliest_visit_no]
            all_hemoglobin_df = all_labs_df[all_labs_df['ITEM'] == 'Hemoglobin']
            all_hemoglobin_dates = list(all_hemoglobin_df['OBS_DTM'].values)
            all_hemoglobin_values = list(all_hemoglobin_df['OBS_VALUE'].values)
            hemoglobin_dates_values = list(zip(all_hemoglobin_dates, all_hemoglobin_values))
            hemoglobin_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(hemoglobin_dates_values) > 0:
                last_hemoglobin_date, last_hemoglobin_value = hemoglobin_dates_values[-1]
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_LAST_12_FLAG'] = 1 if try_float(last_hemoglobin_value) < 12 else 0
            else:
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_LAST_12_FLAG'] = 0

            # compute SODIUM_LEVEL_LAST_135_FLAG
            all_labs_df = labs_df[labs_df['VISIT_NO'] == earliest_visit_no]
            all_sodium_df = all_labs_df[all_labs_df['ITEM'] == 'Sodium, Serum or Plasma']
            all_sodium_dates = list(all_sodium_df['OBS_DTM'].values)
            all_sodium_values = list(all_sodium_df['OBS_VALUE'].values)
            sodium_dates_values = list(zip(all_sodium_dates, all_sodium_values))
            sodium_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(sodium_dates_values) > 0:
                last_sodium_date, last_sodium_value = sodium_dates_values[-1]
                target_df.at[iter, 'SODIUM_LEVEL_LAST_135_FLAG'] = 1 if try_float(last_sodium_value) < 135 else 0
            else:
                target_df.at[iter, 'SODIUM_LEVEL_LAST_135_FLAG'] = 0

        target_df.at[iter,'HOSPITAL_SCORE'] = 2 * target_df.at[iter, 'LOS5_FLAG']\
                                              + target_df.at[iter, 'PROCEDURE_FLAG']\
                                              + 2 * (float(target_df.at[iter, 'PRIOR_YEAR_ADMISSIONS_COUNT']) > 2)\
                                              + 3 * (float(target_df.at[iter, 'PRIOR_YEAR_ADMISSIONS_COUNT']) > 5)\
                                              + target_df.at[iter, 'NONELECTIVE_ADMISSION_FLAG']\
                                              + 2 * target_df.at[iter, 'ONCOLOGY_SERVICE_FLAG']\
                                              + target_df.at[iter, 'HEMOGLOBIN_LEVEL_LAST_12_FLAG']\
                                              + target_df.at[iter, 'SODIUM_LEVEL_LAST_135_FLAG']

    return target_df

def LABORATORIES(target_df, diagnoses_df, labs_df):
    '''
    Laboratory statistics
    :return:
    '''
    # initializing columns
    target_df['SODIUM_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['SODIUM_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['SODIUM_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['SODIUM_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['SODIUM_LEVEL_LAST'] = [0.0] * len(target_df)
    target_df['SODIUM_LEVEL_AVG_136_FLAG'] = [0] * len(target_df)

    target_df['CALCIUM_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['CALCIUM_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['CALCIUM_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['CALCIUM_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['CALCIUM_LEVEL_LAST'] = [0.0] * len(target_df)
    target_df['CALCIUM_LEVEL_AVG_86_FLAG'] = [0] * len(target_df)

    target_df['CREATININE_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['CREATININE_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['CREATININE_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['CREATININE_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['CREATININE_LEVEL_LAST'] = [0.0] * len(target_df)

    target_df['HEMOGLOBIN_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['HEMOGLOBIN_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['HEMOGLOBIN_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['HEMOGLOBIN_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['HEMOGLOBIN_LEVEL_LAST'] = [0.0] * len(target_df)

    target_df['CKI_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['CKI_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['CKI_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['CKI_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['CKI_LEVEL_LAST'] = [0.0] * len(target_df)

    target_df['CKT_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['CKT_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['CKT_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['CKT_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['CKT_LEVEL_LAST'] = [0.0] * len(target_df)

    target_df['BNP_LEVEL_AVG'] = [0.0] * len(target_df)
    target_df['BNP_LEVEL_MIN'] = [0.0] * len(target_df)
    target_df['BNP_LEVEL_MAX'] = [0.0] * len(target_df)
    target_df['BNP_LEVEL_FIRST'] = [0.0] * len(target_df)
    target_df['BNP_LEVEL_LAST'] = [0.0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get earliest AMI admission
        all_visits_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_visits_df[all_visits_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_no
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

            # get the earliest admission labs
            adm_labs_df = labs_df[labs_df['VISIT_NO'] == earliest_visit_no]

            # get all the sodium records
            sodium_df = adm_labs_df[adm_labs_df['ITEM'] == 'Sodium, Serum or Plasma']
            sodium_values = [try_float(x) for x in list(sodium_df['OBS_VALUE'].values)]
            sodium_dates = list(sodium_df['OBS_DTM'].values)
            sodium_dates_values = list(zip(sodium_dates, sodium_values))
            sodium_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(sodium_values) > 0:
                target_df.at[iter, 'SODIUM_LEVEL_AVG'] = stat.mean(sodium_values)
                target_df.at[iter, 'SODIUM_LEVEL_MIN'] = min(sodium_values)
                target_df.at[iter, 'SODIUM_LEVEL_MAX'] = max(sodium_values)
                _, first_sodium_value = sodium_dates_values[0]
                target_df.at[iter, 'SODIUM_LEVEL_FIRST'] = first_sodium_value
                _, last_sodium_value = sodium_dates_values[-1]
                target_df.at[iter, 'SODIUM_LEVEL_LAST'] = last_sodium_value
                target_df.at[iter, 'SODIUM_LEVEL_AVG_136_FLAG'] = 1 if stat.mean(sodium_values) < 136 else 0

            # get all the calcium records
            calcium_df = adm_labs_df[adm_labs_df['ITEM'] == 'CALCIUM']
            calcium_values = [try_float(x) for x in list(calcium_df['OBS_VALUE'].values)]
            calcium_dates = list(calcium_df['OBS_DTM'].values)
            calcium_dates_values = list(zip(calcium_dates, calcium_values))
            calcium_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(calcium_values) > 0:
                target_df.at[iter, 'CALCIUM_LEVEL_AVG'] = stat.mean(calcium_values)
                target_df.at[iter, 'CALCIUM_LEVEL_MIN'] = min(calcium_values)
                target_df.at[iter, 'CALCIUM_LEVEL_MAX'] = max(calcium_values)
                _, first_calcium_value = calcium_dates_values[0]
                target_df.at[iter, 'CALCIUM_LEVEL_FIRST'] = first_calcium_value
                _, last_calcium_value = calcium_dates_values[-1]
                target_df.at[iter, 'CALCIUM_LEVEL_LAST'] = last_calcium_value
                target_df.at[iter, 'CALCIUM_LEVEL_AVG_86_FLAG'] = 1 if stat.mean(calcium_values) < 8.6 else 0

            # get all the creatinine records
            creatinine_df = adm_labs_df[adm_labs_df['ITEM'] == 'CREATININE']
            creatinine_values = [try_float(x) for x in list(creatinine_df['OBS_VALUE'].values)]
            creatinine_dates = list(creatinine_df['OBS_DTM'].values)
            creatinine_dates_values = list(zip(creatinine_dates, creatinine_values))
            creatinine_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(creatinine_values) > 0:
                target_df.at[iter, 'CREATININE_LEVEL_AVG'] = stat.mean(creatinine_values)
                target_df.at[iter, 'CREATININE_LEVEL_MIN'] = min(creatinine_values)
                target_df.at[iter, 'CREATININE_LEVEL_MAX'] = max(creatinine_values)
                _, first_creatinine_value = creatinine_dates_values[0]
                target_df.at[iter, 'CREATININE_LEVEL_FIRST'] = first_creatinine_value
                _, last_creatinine_value = creatinine_dates_values[-1]
                target_df.at[iter, 'CREATININE_LEVEL_LAST'] = last_creatinine_value

            # get all the hemoglobin records
            hemoglobin_df = adm_labs_df[adm_labs_df['ITEM'] == 'Hemoglobin']
            hemoglobin_values = [try_float(x) for x in list(hemoglobin_df['OBS_VALUE'].values)]
            hemoglobin_dates = list(hemoglobin_df['OBS_DTM'].values)
            hemoglobin_dates_values = list(zip(hemoglobin_dates, hemoglobin_values))
            hemoglobin_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(hemoglobin_values) > 0:
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_AVG'] = stat.mean(hemoglobin_values)
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_MIN'] = min(hemoglobin_values)
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_MAX'] = max(hemoglobin_values)
                _, first_hemoglobin_value = hemoglobin_dates_values[0]
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_FIRST'] = first_hemoglobin_value
                _, last_hemoglobin_value = hemoglobin_dates_values[-1]
                target_df.at[iter, 'HEMOGLOBIN_LEVEL_LAST'] = last_hemoglobin_value

            # get all the ck Isoenzyme records
            cki_df = adm_labs_df[adm_labs_df['ITEM'] == 'Creatine Kinase, Isoenzyme MB']
            cki_values = [try_float(x) for x in list(cki_df['OBS_VALUE'].values)]
            cki_dates = list(cki_df['OBS_DTM'].values)
            cki_dates_values = list(zip(cki_dates, cki_values))
            cki_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(cki_values) > 0:
                target_df.at[iter, 'CKI_LEVEL_AVG'] = stat.mean(cki_values)
                target_df.at[iter, 'CKI_LEVEL_MIN'] = min(cki_values)
                target_df.at[iter, 'CKI_LEVEL_MAX'] = max(cki_values)
                _, first_cki_value = cki_dates_values[0]
                target_df.at[iter, 'CKI_LEVEL_FIRST'] = first_cki_value
                _, last_cki_value = cki_dates_values[-1]
                target_df.at[iter, 'CKI_LEVEL_LAST'] = last_cki_value


            # get all the ck total records
            ckt_df = adm_labs_df[adm_labs_df['ITEM'] == 'Creatine Kinase, Total, Ser/Pla']
            ckt_values = [try_float(x) for x in list(ckt_df['OBS_VALUE'].values)]
            ckt_dates = list(ckt_df['OBS_DTM'].values)
            ckt_dates_values = list(zip(ckt_dates, ckt_values))
            ckt_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(ckt_values) > 0:
                target_df.at[iter, 'CKT_LEVEL_AVG'] = stat.mean(ckt_values)
                target_df.at[iter, 'CKT_LEVEL_MIN'] = min(ckt_values)
                target_df.at[iter, 'CKT_LEVEL_MAX'] = max(ckt_values)
                _, first_ckt_value = ckt_dates_values[0]
                target_df.at[iter, 'CKT_LEVEL_FIRST'] = first_ckt_value
                _, last_ckt_value = ckt_dates_values[-1]
                target_df.at[iter, 'CKT_LEVEL_LAST'] = last_ckt_value

            # get all the bnp total records
            bnp_df1 = adm_labs_df[adm_labs_df['ITEM'] == 'proBrain Natriuretic Peptide, NT']
            bnp_df2 = adm_labs_df[adm_labs_df['ITEM'] == 'PROBRAIN NATRIURETIC PEPTIDE_NT']
            bnp_df = pd.concat([bnp_df1, bnp_df2])
            bnp_values = [try_float(x) for x in list(bnp_df['OBS_VALUE'].values)]
            bnp_dates = list(bnp_df['OBS_DTM'].values)
            bnp_dates_values = list(zip(bnp_dates, bnp_values))
            bnp_dates_values.sort(key=lambda x: time.strptime(x[0].split()[0], '%m/%d/%Y'))
            if len(bnp_values) > 0:
                target_df.at[iter, 'BNP_LEVEL_AVG'] = stat.mean(bnp_values)
                target_df.at[iter, 'BNP_LEVEL_MIN'] = min(bnp_values)
                target_df.at[iter, 'BNP_LEVEL_MAX'] = max(bnp_values)
                _, first_bnp_value = bnp_dates_values[0]
                target_df.at[iter, 'BNP_LEVEL_FIRST'] = first_bnp_value
                _, last_bnp_value = bnp_dates_values[-1]
                target_df.at[iter, 'BNP_LEVEL_LAST'] = last_bnp_value

    return target_df

def PRESENTATION_DISEASE(target_df, diagnoses_df, visits_df, med_orders_df, procedures_df):
    '''
    Presentation and disease variables
    :param target_df:
    :param diagnoses_df:
    :return:
    '''
    # initializing columns
    target_df['TRANSFER_PATIENT_FLAG'] = [0] * len(target_df)
    target_df['CHEST_PAIN_FLAG'] = [0] * len(target_df)
    target_df['CARDIAC_ARREST_FLAG'] = [0] * len(target_df)
    target_df['REVASCULARIZATION_FLAG'] = [0] * len(target_df)
    target_df['VESSELS_1_FLAG'] = [0] * len(target_df)
    target_df['VESSELS_2_FLAG'] = [0] * len(target_df)
    target_df['VESSELS_3_FLAG'] = [0] * len(target_df)
    target_df['VESSELS_4_FLAG'] = [0] * len(target_df)
    target_df['VESSELS_COUNT'] = [0] * len(target_df)
    target_df['CLOPIDOGREL_FLAG'] = [0] * len(target_df)
    target_df['AMI_LOCATION'] = ['NA'] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_visits_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_visits_df[all_visits_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

            # compute TRANSFER_PATIENT_FLAG
            target_df.at[iter, 'TRANSFER_PATIENT_FLAG'] = 1 if get_one_value_by_foreign_key(visits_df, 'VISIT_NO',
                    earliest_visit_no, 'PAT_CLASS') == 'INPATIENT' else 0

            # compute CHEST_PAIN_FLAG
            target_df.at[iter, 'CHEST_PAIN_FLAG'] = 1 if set(CHEST_PAIN_CODE) & \
                    set(get_values_by_foreign_key(diagnoses_df, 'VISIT_NO', earliest_visit_no, 'CODE')) else 0

            # compute CARDIAC_ARREST_FLAG
            target_df.at[iter, 'CARDIAC_ARREST_FLAG'] = 1 if set(CARDIAC_ARREST_CODE) & \
                    set(get_values_by_foreign_key(diagnoses_df, 'VISIT_NO', earliest_visit_no, 'CODE')) else 0

            # compute the REVASCULARIZATION_FLAG
            r_flag = 0
            pat_procedure = procedures_df[procedures_df['PAT_ID'] == pat_id]
            REVASCULARIZATION_df = pat_procedure[pat_procedure['CODE'].isin(REVASCULARIZATION_CODE)]
            for r_iter, r_row in REVASCULARIZATION_df.iterrows():
                r_flag = 1
            target_df.at[iter, 'REVASCULARIZATION_FLAG'] = r_flag

            # compute VESSELS_1_FLAG ...
            vessel_codes = list(set(REVASCULARIZATION_df['CODE'].values))
            vessel_count = 0
            additional_vessel = 0
            for code in vessel_codes:
                if code in ONE_VESSEL_CODE:
                    vessel_count += 1
                if code == ADDITIONAL_VESSEL_CODE:
                    additional_vessel = 1
            if vessel_count == 1 and additional_vessel == 0:
                target_df.at[iter, 'VESSELS_1_FLAG'] = 1
            elif vessel_count == 1 and additional_vessel == 1:
                target_df.at[iter, 'VESSELS_2_FLAG'] = 1
            elif vessel_count == 2 and additional_vessel == 1:
                target_df.at[iter, 'VESSELS_3_FLAG'] = 1
            elif vessel_count == 3 and additional_vessel == 1:
                target_df.at[iter, 'VESSELS_4_FLAG'] = 1
            target_df.at[iter, 'VESSELS_COUNT'] = vessel_count + additional_vessel

            # get CLOPIDOGREL_FLAG
            target_df.at[iter, 'CLOPIDOGREL_FLAG'] = 1 if set(CLOPIDOGREL_NAMES) & \
                    set(get_values_by_foreign_key(med_orders_df, 'VISIT_NO', earliest_visit_no, 'ITEM')) else 0

            # get AMI_LOCATION
            ami_visits = set(AMI_LOCATION.keys()).intersection(
                set(get_values_by_foreign_key(diagnoses_df, 'VISIT_NO', earliest_visit_no, 'CODE')))
            if len(ami_visits) > 0:
                target_df.at[iter, 'AMI_LOCATION'] = AMI_LOCATION[list(ami_visits)[0]]

    return target_df

def ADMINISTRATIVE_DATA(target_df, diagnoses_df, visits_df):
    '''
    get administrative data table
    :param target_df:
    :param diagnose_df:
    :return:
    '''

    # initialize table columns with default values
    target_df['INDEX_LOS'] = [0] * len(target_df)
    target_df['ED_VISIT_PRIOR_180_DAYS_COUNT'] = [0] * len(target_df)
    target_df['ADMISSION_PRIOR_30_DAYS_COUNT'] = [0] * len(target_df)
    target_df['ED_VISIT_PRIOR_30_DAYS_COUNT'] = [0] * len(target_df)
    target_df['ED_VISIT_PRIOR_30_DAYS_TIME_IN_ED'] = [0.0] * len(target_df)
    target_df['ED_VISIT_PRIOR_30_DAYS_MINUTES_IN_ED'] = [0.0] * len(target_df)
    target_df['ED_TO_IP_VISIT_PRIOR_30_DAYS_COUNT'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        all_visits_df = visits_df[visits_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates,all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

            # copy the LOS value
            try:
                target_df.at[iter, 'INDEX_LOS'] = row['LOS']
            except:
                # compute LOS
                earliest_dsch_date = get_one_value_by_foreign_key(visits_df, 'VISIT_NO', earliest_visit_no, 'DSCH_DATE')
                target_df.at[iter, 'INDEX_LOS'] = get_date_diff(earliest_dsch_date, earliest_adm_date) + 1

            # compute ED_Visit_Prior_180_Days_Count and ED_Visit_Prior_30_Days_Count, ED_Visit_Prior_30_Days_Time_In_ED
            # ED_to_IP_Visit_Prior_30_Days_Count
            ED_visit_180 = 0
            ED_visit_30 = 0
            ED_visit_30_time = 0
            ED_to_IP_Visit_Prior_30_Days_Count = 0
            ED_dates = get_values_by_foreign_key(all_visits_df, 'PAT_CLASS', 'EMERGENCY', 'DSCH_DATE')
            ED_time = get_values_by_foreign_key(all_visits_df, 'PAT_CLASS', 'EMERGENCY', 'CLINICAL_LOS')
            for i, date in enumerate(ED_dates):
                time_diff = get_date_diff(earliest_adm_date, date)
                if 0 <= time_diff <= 180:
                    ED_visit_180 += 1
                if 0 <= time_diff <= 30:
                    ED_visit_30 += 1
                    ED_visit_30_time += try_float(ED_time[i])
                if 0<= time_diff <= 1:
                    ED_to_IP_Visit_Prior_30_Days_Count += 1

            target_df.at[iter, 'ED_VISIT_PRIOR_180_DAYS_COUNT'] = ED_visit_180
            target_df.at[iter, 'ED_VISIT_PRIOR_30_DAYS_COUNT'] = ED_visit_30
            target_df.at[iter, 'ED_VISIT_PRIOR_30_DAYS_TIME_IN_ED'] = ED_visit_30_time
            target_df.at[iter, 'ED_VISIT_PRIOR_30_DAYS_MINUTES_IN_ED'] = ED_visit_30_time * 24 * 60
            target_df.at[iter, 'ED_TO_IP_VISIT_PRIOR_30_DAYS_COUNT'] = ED_to_IP_Visit_Prior_30_Days_Count

            # compute Admission_Prior_30_Days_Count
            total_admission = 0
            inpatient_dates = get_values_by_foreign_key(all_visits_df, 'PAT_CLASS', 'INPATIENT', 'DSCH_DATE')
            for date in inpatient_dates:
                time_diff = get_date_diff(earliest_adm_date, date)
                if time_diff <= 30 and time_diff >=0 :
                    total_admission += 1
            target_df.at[iter, 'ADMISSION_PRIOR_30_DAYS_COUNT'] = total_admission

    return target_df

def DEMOGRAPHICS_ADDITIONS(target_df):
    '''
    Get demographics and additional information
    :param target_df:
    :return:
    '''

    # initialize table columns with default values
    target_df['AGE_AT_ADMIT'] = ['NA'] * len(target_df)
    target_df['INDEX_ADMISSION_FLAG'] = [1] * len(target_df)
    target_df['DISCHARGE_LOCATION'] = [0] * len(target_df)
    target_df['TRANSFER_AT_DISCHARGE_FLAG'] = [0] * len(target_df)
    target_df['REHAB_FLAG'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        try:
            target_df.at[iter, 'AGE_AT_ADMIT'] = get_date_diff(row['ADMIT_DATE'], row['DOB'])/365 + 1
        except:
            pass

    return target_df

def PATIENT_HISTORY(target_df, diagnoses_df, visits_df):
    '''
    get patient history features
    :return:
    '''
    # initialize table columns with default values
    target_df['HISTORY_CHEST_PAIN_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_AMI_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_CABG_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_PCI_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_PVD_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_ANGINA_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_UNSTABLE_ANGINA_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_HYPERTENTION_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_DEPRESSION_FLAG'] = [0] * len(target_df)
    target_df['FAMILY_DEPRESSION_FLAG'] = [0] * len(target_df)
    target_df['MAJOR_DEPRESSION_COUNT'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates,all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

            # compute History_Chest_Pain_Flag
            chest_pain_dates = []
            for name in CHEST_PAIN_NAMES:
                try: # if chest_pain_date is not empty
                    chest_pain_dates += get_values_by_foreign_key(diagnoses_df, 'CODE_DESC', name, 'ADM_DATE')
                except:
                    chest_pain_dates = get_values_by_foreign_key(diagnoses_df, 'CODE_DESC', name, 'ADM_DATE')

            for date in chest_pain_dates:
                if get_date_diff(earliest_adm_date, date) > 0:
                    target_df.at[iter, 'HISTORY_CHEST_PAIN_FLAG'] = 1

            # compute Family_Depression_Flag
            family_depression_dates = []
            for code in FAMILY_DEPRESSION_CODE:
                try:  # if family_depression_dates is not empty
                    family_depression_dates += get_values_by_foreign_key(diagnoses_df, 'CODE', code, 'ADM_DATE')
                except:
                    family_depression_dates = get_values_by_foreign_key(diagnoses_df, 'CODE', code, 'ADM_DATE')

            for date in family_depression_dates:
                if get_date_diff(earliest_adm_date, date) > 0:
                    target_df.at[iter, 'FAMILY_DEPRESSION_FLAG'] = 1

    return target_df

def IN_HOSPITAL_OUTCOMES(target_df, diagnoses_df, procedures_df):
    '''
    get in-hospital outcomes features
    :return:
    '''
    # initialize table columns with default values
    target_df['ECHOCARDIOGRAPHY_FLAG'] = [0] * len(target_df)
    target_df['IN_HOSPITAL_HF_FLAG'] = [0] * len(target_df)
    target_df['IN_HOSPITAL_ISCHEMIA_FLAG'] = [0] * len(target_df)
    target_df['CARDIAC_PROCEDURE_FLAG'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates,all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

            # get ECHOCARDIOGRAPHY_FLAG
            procedure_codes = get_values_by_foreign_key(procedures_df, 'VISIT_NO', earliest_visit_no,'CODE_DESC')
            target_df.at[iter, 'ECHOCARDIOGRAPHY_FLAG'] = 1 if set(procedure_codes) & set(ECHOCARDIOGRAPHY_CODE) else 0

    return target_df

def COMORBIDITIES(target_df):
    '''
    get comorbidities features
    :param target_df:
    :return:
    '''
    # initialize table columns with default values
    target_df['AGE_80_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_ARRHYTHMIA_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_ANEMIA_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_HYPERTENSION_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_COPD_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_CKD_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_STROKE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_TOBACCO_USE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_DEPRESSION_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_HYPERCHOLESTEROLEMIA_FLAG'] = [0] * len(target_df)

    target_df['COMORBID_CAD_FLAG'] = [0] * len(target_df)
    target_df['PRIOR_REVASCULARIZATION_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_DIABETES_CC_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_DIABETES_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_CHF_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_MI_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_PERIPHERAL_VASCULAR_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_CEREBROVASCULAR_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_DEMENTIA_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_CHRONIC_PULMONARY_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_RHEUMATOLOGIC_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_PEPTIC_ULCER_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_MILD_LIVER_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_HEMIPLEGIA_OR_PARAPLEGIA_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_RENAL_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_MODERATE_OR_SEVERE_LIVER_DISEASE_FLAG'] = [0] * len(target_df)
    target_df['COMORBID_AIDS_FLAG'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[
            all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]


    target_df['COMORBID_DIABETES_CC_FLAG_SCORE'] = target_df['COMORBID_DIABETES_CC_FLAG'] * 2
    target_df['COMORBID_DIABETES_FLAG_SCORE'] = target_df['COMORBID_DIABETES_FLAG']
    target_df['COMORBID_CHF_FLAG_SCORE'] = target_df['COMORBID_CHF_FLAG']
    target_df['COMORBID_MI_FLAG_SCORE'] = target_df['COMORBID_MI_FLAG']
    target_df['COMORBID_PERIPHERAL_VASCULAR_DISEASE_FLAG_SCORE'] = target_df['COMORBID_PERIPHERAL_VASCULAR_DISEASE_FLAG']
    target_df['COMORBID_CEREBROVASCULAR_DISEASE_FLAG_SCORE'] = target_df['COMORBID_CEREBROVASCULAR_DISEASE_FLAG']
    target_df['COMORBID_DEMENTIA_FLAG_SCORE'] = target_df['COMORBID_DEMENTIA_FLAG']
    target_df['COMORBID_CHRONIC_PULMONARY_DISEASE_FLAG_SCORE'] = target_df['COMORBID_CHRONIC_PULMONARY_FLAG']
    target_df['COMORBID_RHEUMATOLOGIC_DISEASE_FLAG_SCORE'] = target_df['COMORBID_RHEUMATOLOGIC_DISEASE_FLAG']
    target_df['COMORBID_PEPTIC_ULCER_DISEASE_FLAG_SCORE'] = target_df['COMORBID_PEPTIC_ULCER_DISEASE_FLAG']
    target_df['COMORBID_MILD_LIVER_DISEASE_FLAG_SCORE'] = target_df['COMORBID_MILD_LIVER_DISEASE_FLAG']
    target_df['COMORBID_HEMIPLEGIA_OR_PARAPLEGIA_FLAG_SCORE'] = target_df['COMORBID_HEMIPLEGIA_OR_PARAPLEGIA_FLAG'] * 2
    target_df['COMORBID_RENAL_DISEASE_FLAG_SCORE'] = target_df['COMORBID_RENAL_DISEASE_FLAG'] * 2
    target_df['COMORBID_MODERATE_OR_SEVERE_LIVER_DISEASE_FLAG_SCORE'] = target_df['COMORBID_ MODERATE_OR_SEVERE_LIVER _DISEASE_FLAG'] * 3
    target_df['COMORBID_AIDS_FLAG_SCORE'] = target_df['COMORBID_AIDS_FLAG'] * 6

    target_df['CHARLSON_DEYO_SCORE'] = target_df['COMORBID_DIABETES_CC_FLAG_SCORE'] + \
                                       target_df['COMORBID_DIABETES_FLAG_SCORE'] + \
                                       target_df['COMORBID_CHF_FLAG_SCORE'] + \
                                       target_df['COMORBID_MI_FLAG_SCORE'] + \
                                       target_df['COMORBID_PERIPHERAL_VASCULAR_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_CEREBROVASCULAR_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_DEMENTIA_FLAG_SCORE'] + \
                                       target_df['COMORBID_CHRONIC_PULMONARY_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_RHEUMATOLOGIC_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_PEPTIC_ULCER_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_MILD_LIVER_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_HEMIPLEGIA_OR_PARAPLEGIA_FLAG_SCORE'] + \
                                       target_df['COMORBID_RENAL_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_MODERATE_OR_SEVERE_LIVER_DISEASE_FLAG_SCORE'] + \
                                       target_df['COMORBID_AIDS_FLAG_SCORE']

    return target_df

def LACE_SCORE(target_df):
    '''
    get lace score features
    :param target_df:
    :return:
    '''

    target_df['LACE_ACUITY_SCORE'] = [0] * len(target_df)
    target_df['LACE_LOS_SCORE'] = [0] * len(target_df)
    target_df['LACE_CHARLSON_SCORE'] = [0] * len(target_df)
    target_df['LACE_ED_SCORE'] = [0] * len(target_df)
    target_df['LACE_SCORE'] = [0] * len(target_df)


    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[
            all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

        target_df.at[iter, 'LACE_ACUITY_SCORE'] = target_df.at[iter, 'NONELECTIVE_ADIMSSION_FLAG'] *3

        # compute Lace LOS SCORE
        if 0 < target_df['LOS'] <= 1:
            target_df.at[iter, 'LACE_LOS_SCORE'] = 1
        elif 1 < target_df['LOS'] <= 2:
            target_df.at[iter, 'LACE_LOS_SCORE'] = 2
        elif 2 < target_df['LOS'] <= 3:
            target_df.at[iter, 'LACE_LOS_SCORE'] = 3
        elif 3 < target_df['LOS'] <= 6:
            target_df.at[iter, 'LACE_LOS_SCORE'] = 4
        elif 6 < target_df['LOS'] <= 13:
            target_df.at[iter, 'LACE_LOS_SCORE'] = 5
        elif 13 < target_df['LOS']:
            target_df.at[iter, 'LACE_LOS_SCORE'] = 7

        # compute LACE_CHARLSON_SCORE
        target_df.at[iter, 'LACE_CHARLSON_SCORE'] = target_df.at[iter, 'CHARLSON_DEYO_SCORE']
        if target_df.at[iter, 'LACE_CHARLSON_SCORE'] >= 4:
            target_df.at[iter, 'LACE_CHARLSON_SCORE'] = 5

        target_df.at[iter, 'LACE_ED_SCORE'] = target_df.at[iter, 'ED_VISIT_PRIOR_180_DAYS_COUNT']
        if target_df.at[iter, 'LACE_ED_SCORE'] >= 4:
            target_df.at[iter, 'LACE_ED_SCORE'] = 4

        target_df.at[iter, 'LACE_SCORE'] = target_df.at[iter, 'LACE_ACUITY_SCORE'] + target_df.at[iter, 'LACE_LOS_SCORE'] \
                                        + target_df.at[iter, 'LACE_CHARLSON_SCORE'] + target_df.at[iter, 'LACE_ED_SCORE']

    return target_df

def ENRICHD_SCORE(target_df, diagnoses_df):
    '''
    get ENRICHD score features
    :param target_df:
    :return:
    '''

    target_df['KILLIP_CLASS'] = ['NA'] * len(target_df)
    target_df['LVEF_FLAG'] = [0] * len(target_df)
    target_df['POST_MI_CABG_FLAG'] = [0] * len(target_df)
    target_df['CHF_FLAG'] = [0] * len(target_df)
    target_df['HISTORY_STROKE_FLAG'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[
            all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

            diagnoses_code_desc = get_values_by_foreign_key(diagnoses_df, 'VISIT_NO', earliest_visit_no, 'CODE_DESC')
            if set(diagnoses_code_desc) & set(CHF_CODE):
                target_df.at[iter, 'KILLIP_CLASS'] = 'I'


    return target_df

def GRACE_SCORE(target_df):
    '''
    get grace score features
    :param target_df:
    :return:
    '''

    target_df['IN_HOSPITAL_PCI_FLAG'] = ['NA'] * len(target_df)
    target_df['SYSTOLIC_BP_AVG'] = [0] * len(target_df)
    target_df['HEART_RATE_AVG'] = [0] * len(target_df)
    target_df['ST_SEGMENT_AVG'] = [0] * len(target_df)
    target_df['TROPONIN_AVG'] = [0] * len(target_df)
    target_df['CARDIAC_MARKER_ELEVATION_FLAG'] = [0] * len(target_df)
    target_df['GRACE_SCORE_AGE'] = [0] * len(target_df)
    target_df['GRACE_SCORE_HEART_RATE'] = [0] * len(target_df)
    target_df['GRACE_SCORE_SYSTOLIC_BP'] = [0] * len(target_df)
    target_df['GRACE_SCORE_CREATININE_LEVEL_FIRST'] = [0] * len(target_df)
    target_df['GRACE_SCORE_KILLIP_CLASS'] = [0] * len(target_df)
    target_df['GRACE_SCORE_CARDIAC_MARKER_ELEVATION'] = [0] * len(target_df)
    target_df['GRACE_SCORE_CARDIAC_ARREST'] = [0] * len(target_df)
    target_df['GRACE_SCORE_STEMI'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_diagnoses_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_diagnoses_df[
            all_diagnoses_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_nod
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]

        # compute GRACE_SCORE_AGE
        age =  target_df[iter, 'AGE_AT_ADMIT']
        if 0 < age <= 30:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 0
        elif 30 < age <= 39:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 8
        elif 39 < age <= 49:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 25
        elif 49 < age <= 59:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 41
        elif 59 < age <= 69:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 58
        elif 69 < age <= 79:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 75
        elif 79 < age <= 89:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 91
        elif 89 < age:
            target_df.at[iter, 'GRACE_SCORE_AGE'] = 100

        # compute GRACE_SCORE_HEART_RATE
        heart_rate = target_df.at[iter,'HEART_RATE_AVG']
        if heart_rate < 50:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 0
        elif 50 <= heart_rate < 70:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 3
        elif 70 <= heart_rate < 90:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 9
        elif 90 <= heart_rate < 110:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 15
        elif 110 <= heart_rate < 150:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 24
        elif 150 <= heart_rate < 200:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 38
        elif heart_rate >= 200:
            target_df.at[iter, 'GRACE_SCORE_HEART_RATE'] = 46

        # compute GRACE_SCORE_SYSTOLIC_BP
        systolic_bp_avg = target_df.at[iter, 'SYSTOLIC_BP_AVG']
        if systolic_bp_avg < 80:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 0
        elif 80 <= systolic_bp_avg < 100:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 53
        elif 100 <= systolic_bp_avg < 120:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 43
        elif 120 <= systolic_bp_avg < 140:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 34
        elif 14 <= systolic_bp_avg < 160:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 24
        elif 160 <= systolic_bp_avg < 200:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 10
        elif 200 <= systolic_bp_avg:
            target_df.at[iter, 'GRACE_SCORE_SYSTOLIC_BP'] = 0

        # compute GRACE_SCORE_CREATININE_LEVEL_FIRST
        c_level =  target_df.at[iter, 'CREATININE_LEVEL_FIRST']
        if 0 < c_level < 0.4:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 1
        elif 0.4 <= c_level < 0.8:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 4
        elif 0.8 <= c_level < 1.2:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 7
        elif 1.2 <= c_level < 1.6:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 10
        elif 1.6 <= c_level < 2.0:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 13
        elif 2.0 <= c_level < 4.0:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 21
        elif 4.0 <= c_level:
            target_df.at[iter, 'GRACE_SCORE_CREATININE_LEVEL_FIRST'] = 28

        # compute GRACE_SCORE_KILLIP_CLASS
        k_class =  target_df.at[iter, 'KILLIP_CLASS']
        if k_class == 'I':
            target_df.at[iter, 'GRACE_SCORE_KILLIP_CLASS'] = 0
        elif k_class == 'II':
            target_df.at[iter, 'GRACE_SCORE_KILLIP_CLASS'] = 20
        elif k_class == 'III':
            target_df.at[iter, 'GRACE_SCORE_KILLIP_CLASS'] = 39
        elif k_class == 'IV':
            target_df.at[iter, 'GRACE_SCORE_KILLIP_CLASS'] = 59

        # compute GRACE_SCORE_CARDIAC_MARKER_ELEVATION, GRACE_SCORE_CARDIAC_ARREST, and GRACE_SCORE_STEMI
        target_df.at[iter,'GRACE_SCORE_CARDIAC_MARKER_ELEVATION'] = target_df.at[iter, 'CARDIAC_MARKER_ELEVATION_FLAG'] * 14
        target_df.at[iter,'GRACE_SCORE_CARDIAC_ARREST'] = target_df.at[iter, 'CARDIAC_ARREST_FLAG'] * 39
        target_df.at[iter,'GRACE_SCORE_STEMI'] = target_df.at[iter, 'STEMI_FLAG'] * 28

    # compute total
    target_df['GRACE_SCORE'] = target_df['GRACE_SCORE_AGE'] + target_df['GRACE_SCORE_HEART_RATE'] + \
                                target_df['GRACE_SCORE_SYSTOLIC_BP'] + target_df['GRACE_SCORE_CREATININE_LEVEL_FIRST'] + \
                                target_df['GRACE_SCORE_KILLIP_CLASS'] + target_df['GRACE_SCORE_CARDIAC_MARKER_ELEVATION'] + \
                                target_df['GRACE_SCORE_CARDIAC_ARREST'] + target_df['GRACE_SCORE_STEMI']

    return target_df

def POLYNOMIAL_TERMS(target_df):
    '''
    get polynomial terms features
    :param target_df:
    :return:
    '''

    target_df['QUAD_CREATINE_MAX'] = target_df['CREATINE_LEVEL_MAX'] ** 2
    target_df['CUBIC_CREATINE_MAX'] = target_df['CREATINE_LEVEL_MAX'] ** 3

    target_df['QUAD_HEMOGLOBIN_MAX'] = target_df['HEMOGLOBIN_LEVEL_MAX'] ** 2
    target_df['CUBIC_HEMOGLOBIN_MAX'] = target_df['HEMOGLOBIN_LEVEL_MAX'] ** 3

    target_df['QUAD_LOS_NEW'] = target_df['LOS_NEW'] ** 2
    target_df['CUBIC_LOS_NEW'] = target_df['LOS_NEW'] ** 3

    target_df['QUAD_HOSPITAL_SCORE'] = target_df['HOSPITAL_SCORE'] ** 2
    target_df['CUBIC_HOSPITAL_SCORE'] = target_df['HOSPITAL_SCORE'] ** 3

    target_df['QUAD_GRACE_SCORE'] = target_df['GRACE_SCORE'] ** 2
    target_df['CUBIC_GRACE_SCORE'] = target_df['GRACE_SCORE'] ** 3

    target_df['QUAD_LACE_SCORE'] = target_df['LACE_SCORE'] ** 2
    target_df['CUBIC_LACE_SCORE'] = target_df['LACE_SCORE'] ** 3

    return target_df

def INTERACTION_TERMS(target_df):
    '''
    get interaction terms featurs
    :param target_df:
    :return:
    '''
    target_df['i_AKI_BNP_CAT'] = target_df['AKI_FLAG'] * target_df['BNP_FIRST_CAT']
    target_df['i_AKI_CARDIAC_A'] = target_df['AKI_FLAG'] * target_df['CARDIAC_ARREST_FLAG']
    target_df['i_AKI_Sodium'] = target_df['AKI_FLAG'] * target_df['SODIUM_LEVEL_AVG_136_FLAG']
    target_df['i_AKI_CK_MAX'] = target_df['AKI_FLAG'] * target_df['CK_LEVEL_MAX']
    target_df['i_AKI_CKD'] = target_df['AKI_FLAG'] * target_df['COMORBID_CKD_FLAG']
    target_df['i_AKI_DEMENTIA'] = target_df['AKI_FLAG'] * target_df['COMORBID_DEMENTIA_FLAG']
    target_df['i_AKI_STROKE'] = target_df['AKI_FLAG'] * target_df['COMORBID_STROKE_FLAG']
    target_df['i_AKI_CUB_GRACE'] = target_df['AKI_FLAG'] * target_df['CUBIC_GRACE_SCORE']
    target_df['i_AKI_CUB_HEMOG'] = target_df['AKI_FLAG'] * target_df['CUBIC_HEMOG_MAX']
    target_df['i_AKI_MED_ACE'] = target_df['AKI_FLAG'] * target_df['DISCH_MED_ACE_ARB_FLAG']
    target_df['i_AKI_MED_ANTIDEP'] = target_df['AKI_FLAG'] * target_df['DISCH_MED_ANTIDEP_FLAG']
    target_df['i_AKI_PRIOR_ED_COUNT'] = target_df['AKI_FLAG'] * target_df['ED_VISIT_PRIOR_30_DAYS_COUNT']
    target_df['i_AKI_PVD'] = target_df['AKI_FLAG'] * target_df['HISTORY_PVD_FLAG']
    target_df['i_AKI_HOSPITAL'] = target_df['AKI_FLAG'] * target_df['HOSPITAL_SCORE']
    target_df['I_AKI_INHOSP_ISCHEMI'] = target_df['AKI_FLAG'] * target_df['IN_HOSPITAL_ISCHEMIA_FLAG']
    target_df['i_AKI_LACE'] = target_df['AKI_FLAG'] * target_df['LACE_SCORE']
    target_df['i_AKI_NONELECTIVE'] = target_df['AKI_FLAG'] * target_df['NONELECTIVE_ADMISSION_FLAG']
    target_df['i_AKI_ONCOLOGY'] = target_df['AKI_FLAG'] * target_df['ONCOLOGY_FLAG']
    target_df['i_AKI_DIS_METAB_90D'] = target_df['AKI_FLAG'] * target_df['PRIOR_DIS_MAGN_METAB_90D']
    target_df['i_AKI_PRIOR_YR_COUNT'] = target_df['AKI_FLAG'] * target_df['PRIOR_YEAR_ADMISSIONs_COUNT']
    target_df['i_AKI_QUAD_GRACE'] = target_df['AKI_FLAG'] * target_df['QUAD_GRACE_SCORE']
    target_df['i_AKI_REVASC'] = target_df['AKI_FLAG'] * target_df['REVASCULARIZATION_FLAG']
    target_df['i_AKI_TRANSFER_PT'] = target_df['AKI_FLAG'] * target_df['TRANSFER_PATIENT_FLAG']
    target_df['i_AKI_VESSE_COUNT'] = target_df['AKI_FLAG'] * target_df['VESSELS_COUNT']

    target_df['i_BNP_CARDIAC_A'] = target_df['BNP_FIRST_CAT'] * target_df['CARDIAC_ARREST_FLAG']
    target_df['i_BNP_SODIUM'] = target_df['BNP_FIRST_CAT'] * target_df['SODIUM_LEVEL_AVG_136_FLAG']
    target_df['i_BNP_CK_MAX'] = target_df['BNP_FIRST_CAT'] * target_df['CK_LEVEL_MAX']
    target_df['i_BNP_CKD'] = target_df['BNP_FIRST_CAT'] * target_df['COMORBID_CKD_FLAG']
    target_df['i_BNP_DEMENTIA'] = target_df['BNP_FIRST_CAT'] * target_df['COMORBID_DEMENTIA_FLAG']
    target_df['i_BNP_STROKE'] = target_df['BNP_FIRST_CAT'] * target_df['COMORBID_STROKE_FLAG']
    target_df['i_BNP_CUB_GRACE'] = target_df['BNP_FIRST_CAT'] * target_df['CUBIC_GRACE_SCORE']
    target_df['i_BNP_CUB_HEMOG'] = target_df['BNP_FIRST_CAT'] * target_df['CUBIC_HEMOG_MAX']
    target_df['i_BNP_MED_ACE'] = target_df['BNP_FIRST_CAT'] * target_df['DISCH_MED_ACE_ARB_FLAG']
    target_df['i_BNP_MED_ANTIDEP'] = target_df['BNP_FIRST_CAT'] * target_df['DISCH_MED_ANTIDEP_FLAG']
    target_df['i_BNP_PRIOR_ED_COUNT'] = target_df['BNP_FIRST_CAT'] * target_df['ED_VISIT_PRIOR_30_DAYS_COUNT']
    target_df['i_BNP_PVD'] = target_df['BNP_FIRST_CAT'] * target_df['HISTORY_PVD_FLAG']
    target_df['i_BNP_HOSPITAL'] = target_df['BNP_FIRST_CAT'] * target_df['HOSPITAL_SCORE']
    target_df['i_BNP_INHOSP_ISCHEMIA'] = target_df['BNP_FIRST_CAT'] * target_df['IN_HOSPITAL_ISCHEMIA_FLAG']
    target_df['i_BNP_LACE'] = target_df['BNP_FIRST_CAT'] * target_df['LACE_SCORE']
    target_df['i_BNP_NONELECTIVE'] = target_df['BNP_FIRST_CAT'] * target_df['NONELECTIVE_ADMISSION_FLAG']
    target_df['i_BNP_ONCOLOGY'] = target_df['BNP_FIRST_CAT'] * target_df['ONCOLOGY_FLAG']
    target_df['i_BNP_DIS_METAB_90D'] = target_df['BNP_FIRST_CAT'] * target_df['PRIOR_DIS_MAGN_METAB_90D']
    target_df['i_BNP_PRIOR_YR_COUNT'] = target_df['BNP_FIRST_CAT'] * target_df['PRIOR_YEAR_ADMISSIONS_COUNT']
    target_df['i_BNP_QUAD_GRACE'] = target_df['BNP_FIRST_CAT'] * target_df['QUAD_GRACE_SCORE']
    target_df['i_BNP_REVASCULARIZATION'] = target_df['BNP_FIRST_CAT'] * target_df['REVASCULARIZATION_FLAG']
    target_df['i_BNP_TRANSFER_PT'] = target_df['BNP_FIRST_CAT'] * target_df['TRANSFER_PATIENT_FLAG']
    target_df['i_BNP_VESSEL_COUNT'] = target_df['BNP_FIRST_CAT'] * target_df['VESSELS_COUNT']
    target_df['i_BNP_AKI_DUR'] = target_df['BNP_FIRST_CAT'] * target_df['AKI_DURATION']

    target_df['i_CARDIAC_A_Sodium'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['SODIUM_LEVEL_AVG_136_FLAG']
    target_df['i_CARDIAC_A_CK_MAX'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['CK_LEVEL_MAX']
    target_df['i_CARDIAC_A_CKD'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['COMORBID_CKD_FLAG']
    target_df['i_CARDIAC_A_DEMENTIA'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['COMORBID_DEMENTIA_FLAG']
    target_df['i_CARDIAC_A_STROKE'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['COMORBID_STROKE_FLAG']
    target_df['i_CARDIAC_A_CUB_GRACE'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['CUBIC_GRACE_SCORE']
    target_df['i_CARDIAC_A_CUB_HEMOG'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['CUBIC_HEMOG_MAX']
    target_df['i_CARDIAC_A_MED_ACE'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['DISCH_MED_ACE_ARB_FLAG']
    target_df['i_CARDIAC_A_MED_ANTIDEP'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['DISCH_MED_ANTIDEP_FLAG']
    target_df['i_CARDIAC_A_PRIOR_ED_COUNT'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['ED_VISIT_PRIOR_30_DAYS_COUNT']
    target_df['i_CARDIAC_A_PVD'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['HISTORY_PVD_FLAG']
    target_df['i_CARDIAC_A_HOSPITAL'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['HOSPITAL_SCORE']
    target_df['i_CARDIAC_A_INHOSP_ISCHEMIA'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['IN_HOSPITAL_ISCHEMIA_FLAG']
    target_df['i_CARDIAC_A_LACE'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['LACE_SCORE']
    target_df['i_CARDIAC_A_NONELECTIVE'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['NONELECTIVE_ADMISSION_FLAG']
    target_df['i_CARDIAC_A_ONCOLOGY'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['ONCOLOGY_FLAG']
    target_df['i_CARDIAC_A_DIS_METAB_90D'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['PRIOR_DIS_MAGN_METAB_90D']
    target_df['i_CARDIAC_A_PRIOR_YR_COUNT'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['PRIOR_YEAR_ADMISSIONS_COUNT']
    target_df['i_CARDIAC_A_QUAD_GRACE'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['QUAD_GRACE_SCORE']
    target_df['i_CARDIAC_A_REVASCULARIZATION'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['REVASCULARIZATION_FLAG']
    target_df['i_CARDIAC_A_TRANSFER_PT'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['TRANSFER_PATIENT_FLAG']
    target_df['i_CARDIAC_A_VESSEL_C'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['VESSELS_COUNT']
    target_df['i_CARDIAC_A_AKI_DUR'] = target_df['CARDIAC_ARREST_FLAG'] * target_df['AKI_DURATION']
    target_df['i_SODIUM_CK_MAX'] = target_df['SODIUM_LEVEL_AVG_136_FLAG'] * target_df['CK_LEVEL_MAX']

    return target_df

def DATA_MANAGEMENT(target_df, visits_df):
    '''
    get Variables created for data management/outcome
    :param target_df:
    :return:
    '''

    target_df['LOS_NEW'] = target_df['LOS']
    target_df['NEW_LOS5_FLAG'] = target_df['LOS5_FLAG']
    target_df['PREVIOUS_YR'] = 1 if target_df['PRIOR_YEAR_ADMISSIONS_COUNT'] > 0 else 0
    target_df['PREVIOUS_30_DAY'] = [0] * len(target_df)
    target_df['FLG_30D'] = [0] * len(target_df)

    for iter, row in target_df.iterrows():
        pat_id = row['PERSON_ID']
        # get the patients all visit rows
        all_visits_df = diagnoses_df[diagnoses_df['PAT_ID'] == pat_id]
        ACUTE_MYOCARDIAL_INFARCTION_df = all_visits_df[all_visits_df['CODE'].isin(ACUTE_MYOCARDIAL_INFARCTION_CODE)]
        all_adm_dates = list(ACUTE_MYOCARDIAL_INFARCTION_df['ADM_DATE'].values)
        all_visit_no = list(ACUTE_MYOCARDIAL_INFARCTION_df['VISIT_NO'].values)

        # get the earlist adm and visit_no
        if len(all_adm_dates) > 0:
            date_pairs = list(zip(all_adm_dates, all_visit_no))
            date_pairs.sort(key=lambda x: time.strptime(x[0], '%m/%d/%Y'))
            earliest_adm_date, earliest_visit_no = date_pairs[0]
            target_df.at[iter, 'VISIT_NO'] = earliest_visit_no

            # compute prior 30 day
            prior_30_day_admission = 0
            all_admissions_df = visits_df[visits_df['PAT_ID'] == pat_id]
            for adm_iter, adm_row in all_admissions_df.iterrows():
                if 0 < get_date_diff(earliest_adm_date, adm_row['DSCH_DATE']) <= 30:
                    prior_30_day_admission = 1
            target_df.at[iter, 'PREVIOUS_30_DAY'] = prior_30_day_admission


            # compute flg 30d
            post_30_day_admission = 0
            for adm_iter, adm_row in all_admissions_df.iterrows():
                if 0 < get_date_diff(adm_row['ADM_DATE'], earliest_adm_date) <= 30:
                    post_30_day_admission = 1
            target_df.at[iter, 'FLG_30D'] = post_30_day_admission

    return target_df

def POST_VANDERNILT(target_df):
    '''
    more variables
    :param target_df:
    :return:
    '''
    target_df['CREATININE_LEVEL_DIFF'] = target_df['CREATININE_LEVEL_MAX'] - target_df['CREATININE_LEVEL_MIN']
    target_df['HEMOGLOBIN_LEVEL_DIFF'] = target_df['HEMOGLOBIN_LEVEL_MAX'] - target_df['HEMOGLOBIN_LEVEL_MIN']
    target_df['BNP_LEVEL_DIFF'] = target_df['BNP_LEVEL_MAX'] - target_df['BNP_LEVEL_MIN']

    for iter, row in target_df.iterrows():
        if target_df.at[iter,'CREATININE_LEVEL_DIFF'] >= np.percentile(target_df['CREATININE_LEVEL_DIFF'].values, 75):
            target_df.at[iter, 'CREATININE_75DIFF_FLAG'] = 1
        else:
            target_df.at[iter, 'CREATININE_75DIFF_FLAG'] = 2

        if target_df.at[iter,'HEMOGLOBIN_LEVEL_DIFF'] >= np.percentile(target_df['HEMOGLOBIN_LEVEL_DIFF'].values, 75):
            target_df.at[iter, 'HEMOGLOBIN_75DIFF_FLAG'] = 1
        else:
            target_df.at[iter, 'HEMOGLOBIN_75DIFF_FLAG'] = 2

        if target_df.at[iter,'BNP_LEVEL_DIFF'] >= np.percentile(target_df['BNP_LEVEL_DIFF'].values, 75):
            target_df.at[iter, 'BNP_LEVEL_75DIFF_FLAG'] = 1
        else:
            target_df.at[iter, 'BNP_LEVEL_75DIFF_FLAG'] = 2

    return target_df


if __name__ == '__main__':
    # reading files into panda dataframes
    diagnoses_df = pd.read_csv(DIAGNOSES_PATH, dtype=object, sep='|', quoting=3, na_filter=False)
    labs_df = pd.read_csv(LABS_PATH, dtype=object, sep='|', quoting=3, na_filter=False)
    med_admin_df = pd.read_csv(MED_ADMIN_PATH, dtype=object, sep='|', quoting=3, na_filter=False)
    med_orders_df = pd.read_csv(MED_ORDERS_PATH, dtype=object, sep='|', quoting=3, na_filter=False)
    procedures_df = pd.read_csv(PROCEDURES_PATH, dtype=object, sep='|', quoting=3, na_filter=False)
    visits_w_prov_type_df = pd.read_csv(VISITS_W_PROV_TYPE_PATH, dtype=object, sep='|', quoting=3, na_filter=False)
    demographics_df = pd.read_csv(DEMOGRAPHICS_PATH, dtype=object, sep='|', quoting=3, na_filter=False)

    # choose first columns for testing purpose. Saving some time
    takehead = 0
    if takehead > 0:
        diagnoses_df = diagnoses_df.head(takehead)
        labs_df = labs_df.head(takehead)
        med_admin_df = med_admin_df.head(takehead)
        med_orders_df = med_orders_df.head(takehead)
        procedures_df = procedures_df.head(takehead)
        visits_w_prov_type_df = visits_w_prov_type_df.head(takehead)
        demographics_df = demographics_df.head(takehead)

    # remove all quotations for data unity purpose
    diagnoses_df = diagnoses_df.applymap(remove_quotes)
    labs_df = labs_df.applymap(remove_quotes)
    med_admin_df = med_admin_df.applymap(remove_quotes)
    med_orders_df = med_orders_df.applymap(remove_quotes)
    procedures_df = procedures_df.applymap(remove_quotes)
    demographics_df = demographics_df.applymap(remove_quotes)

    # create the target dataframe for each unique patient as primary key
    patid_diagnoses = list(diagnoses_df['PAT_ID'].values)
    patid_labs = list(labs_df['PAT_ID'].values)
    patid_med_admin = list(med_admin_df['PAT_ID'].values)
    patid_med_orders = list(med_orders_df['PAT_ID'].values)
    patid_procedure = list(procedures_df['PAT_ID'].values)
    patid_visits_w_prov_type = list(visits_w_prov_type_df['PAT_ID'].values)
    patid_demographics = list(demographics_df['PAT_ID'].values)
    patid_all = patid_diagnoses + patid_labs + patid_med_admin+ patid_med_orders\
                + patid_procedure + patid_visits_w_prov_type + patid_demographics
    unique_pat_id = np.unique(patid_all)
    target_df = pd.DataFrame(unique_pat_id)
    # name the primary key of the target table as 'PERSON_ID'
    target_df.columns = ['PERSON_ID']

    # generate table columns by sections
    target_df = DEMOGRAPHICS(target_df, demographics_df, diagnoses_df)
    target_df = PRIOR_MONTH_DIAGNOSIS(target_df, diagnoses_df)
    target_df = HOSPITAL_SCORE(target_df, procedures_df, diagnoses_df, visits_w_prov_type_df, labs_df)
    target_df = LABORATORIES(target_df, diagnoses_df, labs_df)
    target_df = PRESENTATION_DISEASE(target_df, diagnoses_df, visits_w_prov_type_df, med_orders_df, procedures_df)
    target_df = ADMINISTRATIVE_DATA(target_df, diagnoses_df, visits_w_prov_type_df)
    target_df = DEMOGRAPHICS_ADDITIONS(target_df)
    target_df = PATIENT_HISTORY(target_df, diagnoses_df, visits_w_prov_type_df)
    target_df = IN_HOSPITAL_OUTCOMES(target_df, diagnoses_df, procedures_df)
    target_df = COMORBIDITIES(target_df)
    target_df = LACE_SCORE(target_df)
    target_df = ENRICHD_SCORE(target_df, diagnoses_df)
    target_df = GRACE_SCORE(target_df)
    target_df = POLYNOMIAL_TERMS(target_df)
    target_df = DATA_MANAGEMENT(target_df, visits_w_prov_type_df)
    target_df = POST_VANDERNILT(target_df)
    target_df = INTERACTION_TERMS(target_df)


    # write the whole table to csv file
    target_df.fillna('')
    target_df.to_csv(TARGET_PATH)
