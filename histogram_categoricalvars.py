cs# -*- coding: utf-8 -*-
"""histogram_categoricalVars.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U0FF9ra3FC11hivnJUZTpAXKnuiLFExH
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
#

path_train = input("Paste path to the training data and press enter (without the '' marks), e.g. /content/drive/MyDrive/Kaggle/kaggle_data/train.csv: ").strip()
path_column_types = input("Paste path to the table with column types and press enter (without the '' marks), e.g. /content/drive/MyDrive/Kaggle/kaggle_data/test.csv: ").strip()
data_train = pd.read_csv(path_train)
column_types_table_input = pd.read_csv(path_column_types, index_col=0)

# Create an EDA folder if it doesn't exist
def create_eda_folder():
    current_dir = os.getcwd()

    eda_folder_path = os.path.join(current_dir, 'EDA')
    if not os.path.exists(eda_folder_path):
        os.makedirs(eda_folder_path)

def categorical_plot(train, column_types_table):
      # create a dictionary to make a variable name of the plots & plot subplots as a consequence
    object_cols = ast.literal_eval(column_types_table.loc['object'].iloc[0])
    plot_keys = train[object_cols].columns.tolist()
    addText = "_ax"
    plot_vals = [str(i)+addText for i in plot_keys]
    plot_dict=dict(zip(plot_keys, plot_vals))
    
    fig, axs = plt.subplots(nrows=train[object_cols].shape[1], ncols=1, squeeze=False) #squeeze=False is crucial here for creating these for... subscritable plots with variable name!!!!
    
    fig.set_size_inches(w=9, h=15)
    
      # Adjust spacing
    plt.subplots_adjust(#left=0.1, right=0.9,
                        #top=0.9, bottom=0.5,
                        wspace=0.4, hspace=0.9)
    
    
    for c in train[object_cols].columns:
        col=[c]
        index=plot_keys.index(c)
        data_sorted = pd.DataFrame(train[c].sort_values(ascending=True))
        col_counts = train[col[0]].value_counts()
        plot_dict[c] = sns.histplot(data=data_sorted, x=col[0], ax=axs[index][0])
        plot_dict[c].set_title("Counts of "+col[0])
        plot_dict[c].bar_label(plot_dict[c].containers[0], label_type='edge') # displays value above each bar in a histogram
        plot_dict[c].set_xlabel("Levels", fontsize=10)
        plot_dict[c].set_ylabel("Count", fontsize=10)
        plot_dict[c].set_ylim(top=col_counts.max()+5000) #note! for ylim adjustment to have visible effect on subplots, need to make it in multiples of ticks or adjust ticks also
        plot_dict[c].set_xticks(train[col[0]].value_counts().index.tolist())
        plot_dict[c].set_xticklabels(col_counts.index.tolist(),rotation=20) # rotation of the x-labels

plot_to_save = categorical_plot(train=data_train, column_types_table=column_types_table_input)
create_eda_folder()
file_name = 'histogram_categorical.png'
eda_folder_path = os.path.join(os.getcwd(), 'EDA')

file_path = os.path.join(eda_folder_path, file_name)
#with open(file_path, 'w', encoding = 'utf-8-sig') as f:
plt.savefig(file_path)

#plot_to_save
text_to_display = 'File name '+file_name[:-4] +'. '+'This is also saved in the EDA folder'
print(text_to_display)
