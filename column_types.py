import pandas as pd
import numpy as np
import sys
import os


path = input("Paste path to the training data and press enter (without the '' marks), e.g. /content/drive/MyDrive/Kaggle/kaggle_data/train.csv: ").strip()
data = pd.read_csv(path)

# Create an EDA folder if it doesn't exist
def create_eda_folder():
    current_dir = os.getcwd()

    eda_folder_path = os.path.join(current_dir, 'EDA')
    if not os.path.exists(eda_folder_path):
        os.makedirs(eda_folder_path)

# identifying columns which have different values and so are useless for prediction:
def identifier_cols_fctn(df):
  colnames_return=[]
  for col in df.columns.tolist():
    if len(set(df[col])) == len(df[col]):
      colnames_return.append(col)
    else: pass
    return colnames_return


# identifying columns containing missing values
def missing_val_cols_fctn(df):
  colnames_return=[]
  for col in data.columns.tolist():
    if sum(data[col].isnull()) != 0:
      colnames_return.append(col)
    else: pass
    return colnames_return


def eda_vals_all_cols():
    identifier_cols = identifier_cols_fctn(data)
    #numerical columns
    numerical_cols=list(data.select_dtypes(include='number'))
    # boolean columns
    bool_cols=list(data.select_dtypes(include='bool'))
    # object columns
    object_cols=list(data.select_dtypes(include='object'))
    # category columns
    category_cols=list(data.select_dtypes(include='category'))
    # datetime columns
    datetime_cols=list(data.select_dtypes(include='datetime'))
    #return columns containing missing values
    missing_val_cols=missing_val_cols_fctn(data)
    # return all other column names which were not returned as part of the earlier investigation
    other = list(set(data.columns.tolist()).difference(set(identifier_cols+numerical_cols+bool_cols+object_cols+category_cols+datetime_cols+missing_val_cols)))
    eda_vals=[str(identifier_cols), str(numerical_cols), str(bool_cols), str(object_cols), 
              str(category_cols), str(datetime_cols), str(missing_val_cols), str(other)]

    return eda_vals

def main():
    eda_keys=['identifier', 'numerical', 'boolean', 'object', 'category', 'datetime', 'missing', 'other']
    eda_vals=eda_vals_all_cols()
    eda_col_types = dict(zip(eda_keys, eda_vals))
    return pd.DataFrame.from_dict(eda_col_types, orient='index').rename(columns={0:'column_names'})

col_types_eda_df = main()
create_eda_folder()

eda_folder_path = os.path.join(os.getcwd(), 'EDA')
file_path = os.path.join(eda_folder_path, 'col_types.csv')
with open(file_path, 'w', encoding = 'utf-8-sig') as f:
  col_types_eda_df.to_csv(f)

with pd.option_context('display.max_colwidth', None):
  col_types_eda_df

#del eda_keys, eda_vals, other, missing_val_cols, datetime_cols, category_cols, object_cols, bool_cols, numerical_cols, identifier_cols
# Note: Advantage of wrappping everything in functions: do not create variables that are then stored in memory!!!
# Hence also worth creating functions/classes

print(col_types_eda_df)
print('Dataframe Name:col_types.csv. This dataframe is also saved in the EDA folder')
