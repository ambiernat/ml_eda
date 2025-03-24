import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
#

path_train = input("Paste path to the training data and press enter (without the '' marks), e.g. /content/drive/MyDrive/Kaggle/kaggle_data/train.csv: ").strip()
target_name = input("Enter the name of the target column and press enter (without the '' marks), e.g. loan_status").strip()
variable_name = input("Enter the name of the required numerical column and press enter (without the '' marks), e.g. loan_amount").strip()

data_train = pd.read_csv(path_train)

# Create an EDA folder if it doesn't exist
def create_eda_folder():
    current_dir = os.getcwd()

    eda_folder_path = os.path.join(current_dir, 'EDA')
    if not os.path.exists(eda_folder_path):
        os.makedirs(eda_folder_path)

def numerical1var_variousGraphs(train, var_name, target):
  fig, axs = plt.subplots(nrows=1, ncols=5, squeeze=False)#, gridspec_kw={'width_ratios': [1, 10, 10, 10, 10]}) # adjusts the size of subplots within the plot
  fig.set_size_inches(w=15, h=3)

  #fig.canvas.mpl_connect('close_event', on_close)
  text_plot=axs[0,0].text(0.2, 0.5, var_name, dict(size=15)) # graph with text which is variable name

  hist=sns.histplot(data=train, x=var_name, ax=axs[0][1])
  hist.title.set_text('Univariate - histogram')
  uni_box=sns.boxplot(data=train[var_name], ax=axs[0][2])
  uni_box.title.set_text('Univariate - box plot')
  box=sns.boxplot(x=target, y=var_name, data=train, hue=target, ax=axs[0][3])
  box.title.set_text('Box Plot')
  violin=sns.violinplot(x=target, y=var_name,  data=train, hue=target, ax=axs[0][4])
  violin.title.set_text('Violin Plot')

plot_to_save = numerical1var_variousGraphs(train=data_train, var_name=variable_name, target=target_name)
create_eda_folder()
file_name = 'singleNumerical_variousGraphs.png'
eda_folder_path = os.path.join(os.getcwd(), 'EDA')

file_path = os.path.join(eda_folder_path, file_name)
#with open(file_path, 'w', encoding = 'utf-8-sig') as f:
plt.savefig(file_path)

#plot_to_save
text_to_display = 'File name: '+file_name[:-4] +'. '+'This is also saved in the EDA folder'
print(text_to_display)
