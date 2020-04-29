# Dartmouth-Project

This project process raw data of the Dartmouth project  into tables that can be used by machine learning systems.

The code is not complete as some data columns are still missing, but the code can be easily extended.

How to use:
The script is python 3.6 runable.

To use this code, simply put all required files in the same directory with this script. 

Files needed for run:

IRB_90679_Chapman_AMI_demographics_12132018

IRB_90679_Chapman_AMI_diagnoses_12132018

IRB_90679_Chapman_AMI_labs_12132018

IRB_90679_Chapman_AMI_med_admin_12132018

IRB_90679_Chapman_AMI_med_orders_12132018

IRB_90679_Chapman_AMI_procedures_12132018

IRB_90679_Chapman_AMI_visits_w_prov_type_12132018

Then run the script.

```
$ python3 elements.py
```

The script will create a target.csv file with all the data columns in it.

Code configuration:

The code organize all the features in tables by splitting them to different functions.

Each function handles one table in the documentation 'AMI_MasterFlatFile_Data_Dictionary_03.13.19_IMR'.

This segment the both the features and the code so that it is easy to debug and modify.

For any further questions, please email to zhenduow@cs.utah.edu
