# Dartmouth-Project
This project process raw data of the Dartmouth project  into tables that can be used by machine learning systems
The code is not complete as some data columns are still missing, but the code can be easily extended.

How to use:

To use this code, simply put all required files in the same directory with this script. Then run the script.

$ python3 elements.py

The script will create a target.csv file with all the data columns in it.

Code configuration:

The code organize all the features in tables by splitting them to different functions.
Each function handles one table in the documentation 'AMI_MasterFlatFile_Data_Dictionary_03.13.19_IMR'.
This segment the both the features and the code so that it is easy to debug and modify.

For any further questions, please email to zhenduow@cs.utah.edu
