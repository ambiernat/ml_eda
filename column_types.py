import pandas as pd
import numpy as np
import sys


mode = input("Paste path to the training data and press enter (without the '' marks), e.g. /content/drive/MyDrive/Kaggle/kaggle_data/train.csv: ").strip()
data = pd.read_csv(mode)

# identifying columns which have different values and so are useless for prediction:
def identifier_cols_fctn(df):
  colnames_return=[]
  for col in df.columns.tolist():
    if len(set(df[col])) == len(df[col]):
      colnames_return.append(col)
    else: pass
    return colnames_return

identifier_cols = identifier_cols_fctn(data)

numerical_cols=list(data.select_dtypes(include='number'))

bool_cols=list(data.select_dtypes(include='bool'))

object_cols=list(data.select_dtypes(include='object'))

category_cols=list(data.select_dtypes(include='category'))

datetime_cols=list(data.select_dtypes(include='datetime'))

# identifying columns containing missing values
def missing_val_cols_fctn(df):
  colnames_return=[]
  for col in data.columns.tolist():
    if sum(data[col].isnull()) != 0:
      colnames_return.append(col)
    else: pass
    return colnames_return

missing_val_cols=missing_val_cols_fctn(data)

other = list(set(data.columns.tolist()).difference(set(identifier_cols+numerical_cols+bool_cols+object_cols+category_cols+datetime_cols+missing_val_cols)))

eda_vals=[str(identifier_cols), str(numerical_cols), str(bool_cols), str(object_cols), str(category_cols), str(datetime_cols), str(missing_val_cols), str(other)]

eda_keys=['identifier', 'numerical', 'boolean', 'object', 'category', 'datetime', 'missing', 'other']

eda_col_types = dict(zip(eda_keys, eda_vals))

col_types_eda_df = pd.DataFrame.from_dict(eda_col_types, orient='index').rename(columns={0:'column_names'})

with pd.option_context('display.max_colwidth', None):
  col_types_eda_df

del eda_col_types, eda_keys, eda_vals, other, missing_val_cols, datetime_cols, category_cols, object_cols, bool_cols, numerical_cols, identifier_cols

print(col_types_eda_df)
print('Dataframe Name:col_types_eda_df')
