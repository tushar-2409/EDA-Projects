#Importing Necessary Modules
try:
    # To access system-specific parameters and functions.
    import sys, os  

    # For Numerical computing and mathematical functions.
    import numpy as np       

    # For Data manipulation and analysis with DataFrames.
    import pandas as pd
    
    # Data visualization tools.
    import seaborn as sns
    import matplotlib.pyplot as plt

    # For Tabular representation
    from tabulate import tabulate
    
    # To Ignore warning messages.
    import warnings
    warnings.filterwarnings("ignore")
    
    # Scientific and statistical functions.
    from scipy import stats
    
    # For Data preprocessing: label encoding, standardization, and feature scaling.
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

    # Evaluation metrics for machine learning models.
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
    
    # Machine learning algorithms and model evaluation tools.
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split

except Exception as err:
    print(err)

class PreProcessing:
    def __init__(self,path_name,file_name):
        self.path_name=path_name
        self.file_name=file_name

    def reading_file(self):
        try:
            return pd.read_excel(self.path_name+"/"+self.file_name)
        except:
            raise FileExistsError
        
    def splitting_data(self,data,test_size,random_state):
        return train_test_split(data,test_size=test_size, random_state=random_state)
    
    def seperating_dtypes(self,data):
        cont=[]
        cat=[]
        for col in data.columns:
            if data[col].nunique()<10:
                cat.append(col)
            else:
                cont.append(col)
        return cont,cat
    
    def replacing_cat_val(self, data):
        num_replacements = int(input("Enter the number of replacements you want to make: "))
        for _ in range(num_replacements):
            column_name = input("Enter the column name in which you want to replace values: ")
            replace_value = input(f"Enter the value to be replaced in {column_name}: ")
            new_value = input(f"Enter the new value for {column_name}: ")
            data[column_name] = data[column_name].replace(replace_value, new_value)


    def fill_null_values(self,data,cat,cont,cat_impute='mode',cont_impute='median'):
    # To fill null values for categorical columns using mode.
        if cat_impute == 'mode':
            for col in data[cat]:
                if data[col].isnull().any():
                    data[col].fillna(data[col].mode()[0], inplace=True)

    # To Fill null values for continuous columns using median or mean
        if cont_impute == 'median':
            for colu in data[cont]:
                if data[colu].isnull().any():
                    data[colu].fillna(data[colu].median(), inplace=True)
        else:
            for colu in data[cont]:
                if data[colu].isnull().any():
                    data[colu].fillna(data[colu].mean(), inplace=True)

        
    def detect_outliers(self, data,cont_var_list, method="z_score"):  # Function to detect columns with outliers.
        if method=="z_score":
            outliers=[]
            threshold=3
            for col in cont_var_list:
                mean=data[col].mean()
                std=data[col].std()
                z_score=(data[col]-mean)/std
                for z in z_score:
                    if abs(z)>threshold:
                        outliers.append(col)
                        break
            return outliers
        
        else:  # IQR method to check the columns with outliers.
            outlier =[]
            for col in cont_var_list:
                Q1=np.percentile(data[col],25)
                Q3=np.percentile(data[col],75)
                IQR = Q3-Q1
                lwr_bound = Q1-(1.5*IQR)
                upr_bound = Q3+(1.5*IQR)
                if any((data[col] > upr_bound) | (data[col] < lwr_bound)):
                    outlier.append(col)
                else:
                    pass
            return outlier
    
    def treat_outliers_iqr(self,data,outliers_list):
        for outlier in outliers_list:
            q1 = np.percentile(data[outlier], 25)
            q3 = np.percentile(data[outlier], 75)
            # print(q1, q3)
            IQR = q3-q1
            lwr_bound = q1-(1.5*IQR)
            upr_bound = q3+(1.5*IQR)
            data[outlier] = np.where(data[outlier] > upr_bound, upr_bound, data[outlier])
            data[outlier] = np.where(data[outlier] < lwr_bound, lwr_bound, data[outlier])
        return data
    
    def label_encode(self,data):
        fitted_encoder = {}
        columns=data.select_dtypes(object)
        for col in columns:
            le = LabelEncoder()
            fitted_encoder.update({col: le.fit(data[col])})
        return fitted_encoder
    
    def encoder_transform(self,data, fitted_encoders):
        for col, encoder in fitted_encoders.items():
            data[col] = encoder.transform(data[col])
        return data
    
    def perform_hypothesis(self,data,cat,cont,target_col,confidence=0.95,tail=2):
        if tail ==2:
                pass
        else:
            tail = 1

        null = "has no significant effect on Churn"
        alternate= "has significant effect on Churn"
        for col in cat:
            if col!=target_col:
                print(f"Null= {col}",null)
                print(f"Alternate= {col}",alternate) 
                contingency_table = pd.crosstab(data[col], data[target_col])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                print('Test : Chi-square Test , p-value :', p_val)
                if p_val < (1 - confidence)/tail:
                    print("Reject null","\n")
                else:
                    print("Accept null","\n")

        for col in cont:
            if col!=target_col:
                print(f"Null= {col}",null)
                print(f"Alternate= {col}",alternate)   
                r, p = stats.pointbiserialr(data[col], data[target_col])
                print('Test : Point Biserial Test , p-value :', p)
                if p < (1 - confidence)/tail:
                    print("Reject null","\n")
                else:
                    print("Accept null","\n")
        

class Visualization:
    def __init__(self):
        pass

    def Uni_analysis(self,data,category=True):
        fig = plt.figure(figsize=(int(input("Enter Width of figure: ")), int(input("Enter Height of figure: "))))
        gs = fig.add_gridspec(int(input("Enter no. of rows: ")), int(input("Enter no. of Columns: "))) 
        #No. of Rows and columns
        
        if category:
            for i in range(int(input("Enter Range to make subplots :"))):
                x3_column = input(f"Enter column name for countplot subplot {i + 1}: ")
                ax = fig.add_subplot(gs[0, i])  # First row and columns for histograms
                sns.countplot(data=data, x=x3_column, ax=ax)
                ax.set_title(f"Countplot of {x3_column}")

            for i in range(int(input("Enter Range to make subplots :"))):
                x4_column = input(f"Enter column name for countplot subplot {i + 1}: ")
                ax = fig.add_subplot(gs[1, i])  # Second row and columns for boxplots
                sns.countplot(data=data, x=x4_column, ax=ax)
                ax.set_title(f"Countplot of {x4_column}")

        else:
            for i in range(int(input("Enter Range to make subplots :"))):
                x_column = input(f"Enter column name for histogram subplot {i + 1}: ")
                ax = fig.add_subplot(gs[0, i])  # First row and columns for histograms
                sns.histplot(data=data, x=x_column, ax=ax)
                ax.set_title(f"Histogram of {x_column}")

            for i in range(int(input("Enter Range to make subplots :"))):
                x2_column = input(f"Enter column name for boxplot subplot {i + 1}: ")
                ax = fig.add_subplot(gs[1, i])  # Second row and columns for boxplots
                sns.boxplot(data=data, x=x2_column, ax=ax)
                ax.set_title(f"Boxplot of {x2_column}")
            
        plt.tight_layout()  # Adjusts the spacing between subplots
        plt.show()


    def Bi_analysis(self,data):
        plt.figure(figsize=(int(input("Enter Width of figure: ")), int(input("Enter Height of figure: "))))

        row_num=int(input("Enter no. of rows: "))
        col_num=int(input("Enter no. of columns: "))

        for i in range(int(input("Enter starting range for subplots: ")),int(input("Enter ending range for subplots: "))):
            plt.subplot(row_num,col_num,i)
            x_column =input(f"Enter column name for x axis for subplot {i}: ")
            y_column=input(f"Enter column name for y axis for subplot {i}: ")
            plot_type = input("Enter the plot type (barplot/scatterplot/boxplot): ")

            if plot_type == "barplot":
                sns.barplot(data=data, x=x_column, y=y_column)
            elif plot_type == "scatterplot":
                sns.scatterplot(data=data, x=x_column, y=y_column)
            elif plot_type == "boxplot":
                sns.boxplot(data=data, x=x_column, y=y_column)
            else:
                print("Invalid plot type. Please choose 'barplot', 'scatterplot', or 'boxplot'.")
            plt.title(f"Relationship between {x_column} and {y_column}")

    plt.tight_layout()
    plt.show()

    def Mul_analysis(self, data):
        plt.figure(figsize=(int(input("Enter Width of figure: ")), int(input("Enter Height of figure: "))))

        row_num=int(input("Enter no. of rows: "))
        col_num=int(input("Enter no. of columns: "))

        for i in range(int(input("Enter starting range for subplots: ")),int(input("Enter ending range for subplots: "))):
            plt.subplot(row_num,col_num,i)
            x_column =input(f"Enter column name for x axis for subplot {i}: ")
            y_column=input(f"Enter column name for y axis for subplot {i}: ")
            Hue=input(f"Enter column name for hue for subplot {i}: ")
            plot_type = input("Enter the plot type (barplot/scatterplot/boxplot): ")

            if plot_type == "barplot":
                sns.barplot(data=data, x=x_column, y=y_column,hue=Hue)
            elif plot_type == "scatterplot":
                sns.scatterplot(data=data, x=x_column, y=y_column,hue=Hue)
            elif plot_type == "boxplot":
                sns.boxplot(data=data, x=x_column, y=y_column,hue=Hue)
            else:
                print("Invalid plot type. Please choose 'barplot', 'scatterplot', or 'boxplot'.")
            plt.title(f"{x_column} VS {y_column} VS {Hue}")

    plt.tight_layout()
    plt.show()
    

            

                

































