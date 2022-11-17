#Big Data and Machine Learning
#109826005
#

import pandas as pd # to store tabular data
import numpy as np # to do some math
import matplotlib.pyplot as plt # a popular data visualization tool
import seaborn as sns # another popular data visualization tool
        
class Option():
    def __init__(self, questionnumber):
        self.questionnumber = questionnumber
    def check(self,checknumber):
        if int(checknumber) == -1:
            return -1 #Exit
        elif int(checknumber) >=1 and int(checknumber) <=5:
            return int(checknumber)
        else: 
            return 0 #ERROR Input!

class  ShortAnswerQuestions():
            def __init__(self, title,answer):
                self.title = title
                self.answer = answer




while 1:
    option=Option(input('請輸入題號 (1~5, 輸入-1離開程式)：'))
    
    if option.check(option.questionnumber) == -1: #Exit
        print("Thanks! ")
        break
    elif option.check(option.questionnumber) == 0: #ERROR Input!
        print("ERROR Input!")
        continue
    elif option.check(option.questionnumber) == 1:
        print("Question1...")
        
        #Q1
        
        temp_list = [ ('drop missing-valued rows', 0, 0 ),('Impute values with 0 ', 0, 0 ),('Impute values with mean of column', 0, 0 ),('Impute values with median of column', 0, 0 ),('Z-score normalization with median imputing', 0, 0 ),('Min-max normalization with mean imputing', 0, 0 ),('Row-normalization with mean imputing', 0, 0 )]
        #Create a DataFrame object
        df = pd.DataFrame(temp_list, columns = ['Pipeline description' , '# rows model learned from', 'Cross-validated accuracy'])
        
        
        
        # import packages we need for exploratory data analysis (EDA)    
        plt.style.use('fivethirtyeight') # a popular data visualization theme
        # load in our dataset using pandas
        pima_column_names = ['times_pregnant', 'plasma_glucose_concentration','diastolic_blood_pressure', 'triceps_thickness', 'serum_insulin', 'bmi', 'pedigree_function','age', 'onset_diabetes']
        pima = pd.read_csv('pima.data', names=pima_column_names)
        pima['onset_diabetes'].value_counts(normalize=True) 
        
        #missing data
        pima['serum_insulin'] = pima['serum_insulin'].map(lambda x:x if x != 0 else None)# manually replace all 0's with a None value
        # A little faster now for all columns
        columns = ['serum_insulin', 'bmi', 'plasma_glucose_concentration', 'diastolic_blood_pressure','triceps_thickness']
        for col in columns:
            pima[col].replace([0], [None], inplace=True)
          
            
        pima_dropped = pima.dropna()
        
        # now lets do some machine learning
        # note we are using the dataset with the dropped rows
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        
        x_dropped = pima_dropped.drop('onset_diabetes', axis=1)
        # create our feature matrix by removing the response variable
        #print("learning from {} rows".format(x_dropped.shape[0]))
        y_dropped = pima_dropped['onset_diabetes'] #response series
        
        # our grid search variables and instances
        # KNN parameters to try
        knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        
        knn = KNeighborsClassifier() # instantiate a KNN model
        grid = GridSearchCV(knn, knn_params)
        grid.fit(x_dropped, y_dropped)
        #print(len(pima_dropped) )
        df.loc[0,'# rows model learned from'] = len(pima_dropped)
        df.loc[0,'Cross-validated accuracy'] = grid.best_score_
        #print(grid.best_score_, grid.best_params_)
        # but we are learning from way fewer rows…
        
        pima.isnull().sum() # let's fill in the plasma column
        empty_plasma_index = pima[pima['plasma_glucose_concentration'].isnull()].index
        pima.loc[empty_plasma_index]['plasma_glucose_concentration']
        
        pima['plasma_glucose_concentration'].fillna(pima['plasma_glucose_concentration'].mean(), inplace=True)
        # fill the column's missing values with the mean of the rest of the column
        pima.isnull().sum() # the column should now have 0 missing values
        
        pima.loc[empty_plasma_index]['plasma_glucose_concentration']
        
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        pima_imputed = imputer.fit_transform(pima)
        type(pima_imputed) # comes out as an array
        
        pima_imputed = pd.DataFrame(pima_imputed, columns=pima_column_names)
        # turn our numpy array back into a pandas DataFrame object
        #pima_imputed.head()
        # notice for example the triceps_thickness missing values were replaced with 29.15342
        
        pima_imputed.loc[empty_plasma_index]['plasma_glucose_concentration']
        # same values as we obtained with fillna
        
        pima_imputed.isnull().sum() # no missing values
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        x_imputed = pima_imputed.drop('onset_diabetes', axis=1)
        y_imputed = pima_imputed ['onset_diabetes']
        knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        knn = KNeighborsClassifier()
        grid = GridSearchCV(knn, knn_params)
        grid.fit(x_imputed, y_imputed)
        #print(grid.best_score_, grid.best_params_)
        df.loc[1,'# rows model learned from'] = len(pima_imputed)
        df.loc[1,'Cross-validated accuracy'] = grid.best_score_
        
        pima_zero = pima.fillna(0) # impute values with 0
        X_zero = pima_zero.drop('onset_diabetes', axis=1)
        y_zero = pima_zero['onset_diabetes']
        knn_params = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        grid = GridSearchCV(knn, knn_params)
        grid.fit(X_zero, y_zero)
        #print("learning from {} rows".format(X_zero.shape[0]))
        #print(grid.best_score_, grid.best_params_)
        # if the values stayed at 0, our accuracy goes down
        df.loc[2,'# rows model learned from'] = len(pima_zero) 
        df.loc[2,'Cross-validated accuracy'] = grid.best_score_
        
        from sklearn.model_selection import train_test_split
        X = pima[['serum_insulin']].copy()
        y = pima['onset_diabetes'].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
        X.isnull().sum()
        
        entire_data_set_mean = X.mean() # take the entire datasets mean
        X = X.fillna(entire_data_set_mean) # and use it to fill in the missing spots
        #print(entire_data_set_mean)
        # Take the split using a random state so that we can examine the same split.
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
        
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn.score(X_test, y_test)
        
        from sklearn.model_selection import train_test_split
        X = pima[['serum_insulin']].copy()
        y = pima['onset_diabetes'].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=99)
        
        training_mean = X_train.mean()
        X_train = X_train.fillna(training_mean)
        X_test = X_test.fillna(training_mean)
        #print(training_mean)
        
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        #print(knn.score(X_test, y_test))
        
        
        
        from sklearn.pipeline import Pipeline
        knn_params = {'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        knn = KNeighborsClassifier()
        mean_impute = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('classify', knn)])
        
        X = pima.drop('onset_diabetes', axis=1)
        y = pima['onset_diabetes']
        grid = GridSearchCV(mean_impute, knn_params)
        grid.fit(X, y)
        #print(grid.best_score_, grid.best_params_)
        
        
        
        
        
        
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        # we will want to fill in missing values to see all 9 columns
        pima_imputed_mean = pd.DataFrame(imputer.fit_transform(pima),columns=pima_column_names)
        #pima_imputed_mean.hist(figsize=(15, 15))
        
        #pima_imputed_mean.describe()
        #pima_imputed_mean.hist(figsize=(15, 15),sharex=True)
        
        
        
        # built in z-score normalizer
        #ax = pima_imputed_mean ['plasma_glucose_concentration'].hist()
        #ax.set_title('Distribution of plasma_glucose_concentration')
        df.loc[3,'# rows model learned from'] = len(pima_imputed_mean) 
        df.loc[3,'Cross-validated accuracy'] = grid.best_score_
        
        
        
        from sklearn.preprocessing import StandardScaler
        #glucose_z_score_standardized = scaler.fit_transform(pima_imputed_mean[['plasma_glucose_concentration']])
        # note we use the double bracket notation [[ ]] because the transformer requires a dataframe
        #ax = pd.Series(glucose_z_score_standardized.reshape(-1,)).hist()
        #ax.set_title('Distribution of plasma_glucose_concentration after Z Score Scaling')
        
        scale = StandardScaler()
        # instantiate a z-scaler object
        pima_imputed_mean_scaled = pd.DataFrame(scale.fit_transform(pima_imputed_mean),columns=pima_column_names)
        #pima_imputed_mean_scaled.hist(figsize=(15, 15), sharex=True)
        # now all share the same "space"
        
        
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        knn = KNeighborsClassifier()
        knn_params = {'imputer__strategy':['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        mean_impute_standardize = Pipeline([('imputer', SimpleImputer()), ('standardize', StandardScaler()),
        ('classify', knn)])
        X = pima.drop('onset_diabetes', axis=1)
        y = pima['onset_diabetes']
        grid = GridSearchCV(mean_impute_standardize, knn_params)
        grid.fit(X, y)
        
        df.loc[4,'# rows model learned from'] = len(pima_imputed_mean_scaled) 
        df.loc[4,'Cross-validated accuracy'] = grid.best_score_
        ##########################3
        #The min-max scaling method
        #
        from sklearn.preprocessing import MinMaxScaler
        min_max = MinMaxScaler()
        pima_min_maxed = pd.DataFrame(min_max.fit_transform(pima_imputed_mean),columns=pima_column_names)
        pima_min_maxed.describe()
        
        knn_params = {'imputer__strategy': ['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        mean_impute_standardize = Pipeline([('imputer', SimpleImputer()), ('standardize', MinMaxScaler()),('classify', knn)])
        X = pima.drop('onset_diabetes', axis=1)
        y = pima['onset_diabetes']
        grid = GridSearchCV(mean_impute_standardize, knn_params)
        grid.fit(X, y)
        df.loc[5,'# rows model learned from'] = len(pima_min_maxed) 
        df.loc[5,'Cross-validated accuracy'] = grid.best_score_
        
        ##################################
        #The row normalization method :
        #
        from sklearn.preprocessing import Normalizer # our row normalizer
        normalize = Normalizer()
        pima_normalized = pd.DataFrame(normalize.fit_transform(pima_imputed_mean),
        columns=pima_column_names)
        np.sqrt((pima_normalized**2).sum(axis=1)).mean()
        # average vector length of row normalized imputed matrix
        
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        knn = KNeighborsClassifier()
        knn_params = {'imputer__strategy': ['mean', 'median'], 'classify__n_neighbors':[1, 2, 3, 4, 5, 6, 7]}
        mean_impute_normalize = Pipeline([('imputer', SimpleImputer()), ('normalize', Normalizer()), ('classify',knn)])
        X = pima.drop('onset_diabetes', axis=1)
        y = pima['onset_diabetes']
        grid = GridSearchCV(mean_impute_normalize, knn_params)
        grid.fit(X, y)
        #print(grid.best_score_, grid.best_params_)
        df.loc[6,'# rows model learned from'] = len(pima_normalized) 
        df.loc[6,'Cross-validated accuracy'] = grid.best_score_
        
        print(df)
        df.to_csv('PipelineDescription.csv')
        print("Question 1 finished! Check the PipelineDescription.csv file.")
        
        
        #Q1 END
        
    elif option.check(option.questionnumber) == 2:   
        print("Question2...")
        
        #Q2
        # import packages we need for exploratory data analysis (EDA)
        
        climate = pd.read_csv('GlobalLandTemperaturesByCity.csv')
        climate.dropna(axis=0, inplace=True)   # remove missing values 
            
        # Convert the dt column to datetime and extract the year
        climate['dt'] = pd.to_datetime(climate['dt'])
        climate['year'] = climate['dt'].map(lambda value: value.year)
        
        # A subset the data to just the TW
        climate_sub_tw = climate.loc[climate['Country']== 'Taiwan']
        climate_sub_tw = climate_sub_tw.loc[climate_sub_tw['year']>= 1970]# recent fifty years
        
        
        # A subset the data to just the JP
        climate_sub_jp = climate.loc[climate['Country'] == 'Japan']
        climate_sub_jp = climate_sub_jp.loc[climate_sub_jp['year']>= 1970]# recent fifty years
            
        climate_sub_tw.groupby('year').mean()['AverageTemperature'].rolling(10).mean().plot(label="Taiwan")
        climate_sub_jp.groupby('year').mean()['AverageTemperature'].rolling(10).mean().plot(label="Japan")
        plt.legend()
        plt.ylabel("Avg.Temperature") 
        plt.savefig('Q2.png')
        plt.show()
        
        print("Question 2 finished! Check the Q2.png file.")
        #Q2 END
        
    elif option.check(option.questionnumber) == 3:   
        print("Question3...")
        
        #Q3
        salary_ranges = pd.read_csv('Salary_Ranges_by_Job_Classification.csv')
        salary_ranges.isnull().sum()
        
        salary_ranges = salary_ranges[['Biweekly Low Rate', 'Grade']]
        # need to clean our Biweekly High columns to remove the dollarsign in order to visualize
        salary_ranges['Biweekly Low Rate'] = salary_ranges['Biweekly Low Rate'].map(lambda value: value.replace('$',''))
        
        # Convert the Biweeky columns to float
        salary_ranges['Biweekly Low Rate'] = salary_ranges['Biweekly Low Rate'].astype(float)
        # Convert the Grade columns to str
        salary_ranges['Grade'] = salary_ranges['Grade'].astype(str)
        salary_ranges['Grade'].value_counts().sort_values(ascending=False).tail(20).plot(kind='bar')
        
        plt.savefig('Q3.png')
        plt.show()



        #Q3 END
        print("Question 3 finished! Check the Q3.png file.")
        
    elif option.check(option.questionnumber) == 4:   
        print("Question4:\n")
        
        #Q4
        
        
        #Z-score   
        zscore= ShortAnswerQuestions("Z-score 標準化(zero-mean normalization):","Z-score標準化也叫標準差標準化，經過處理的資料符合標準正態分佈，即均值為0，標準差為1,\n經標準差標準化後，資料將符合標準常態分佈(Standard Normal Distribution),\nZ分數標準化適用於分佈大致對稱的資料，因為在非常不對稱的分佈中，標準差的意義並不明確，\n此時若標準化資料，可能會對結果做出錯誤的解讀，另外，當我們未知資料的最大值與最小值，\n或存在超出觀察範圍的離群值時，可透過Z分數標準化來降低離群值對整個模型的影響。")
        print("-",zscore.title)
        print(zscore.answer)
        print("\n")
        
        #min-max scaling
        mmscaling=ShortAnswerQuestions("最小值最大值正規化(Min-Max scaling):","最小值最大值正規化的用意，是將資料等比例縮放到 [0, 1] 區間中，\n此種方法有一點需我們特別注意，即若原始資料有新的數據加入，有可能導致最小值及最大值的改變。")
        print("-",mmscaling.title)
        print(mmscaling.answer)
        print("\n")
        
        #row normalization
        rnormalization=ShortAnswerQuestions("row normalization:","最後一個正規化方法是按行(row)而不是逐個列(欄位)進行。此標準化技術將確保每行數據都具有一樣的範數(Norm)，\n實際上可以看做是在計算長度或距離，代表著每行的向量長度都一樣(正規化每行的向量長度皆為1)，\n而不是計算每列，均值，最小值，最大值等的統計數據。\n另外，因為用來計算距離與長度，在處理文本數據或聚類算法時效果特別顯著。Row normalization可能是此三種正規化方法在資料上表現差。\n但是這並非代表總是表現最差，而是有不同使用情境。")
        print("-",rnormalization.title)
        print(rnormalization.answer)
        print("\n")
        
        conclusion=ShortAnswerQuestions("總結:","在使用正規化方法時應該多做驗證比較，找到適合自己資料集的正規化方法，才能有效優化機器學習的流程。")
        print("-",conclusion.title)
        print(conclusion.answer)
        #Q4 END
        
    elif option.check(option.questionnumber) == 5:   
        print("Question5:\n")
        #Q5
        
        #quant
        quant=ShortAnswerQuestions("quant:","填充分類變數（基於Imputer的自定義填充器，用mean值填充）")
        print("-",quant.title)
        print(quant.answer)
        #category
        category=ShortAnswerQuestions("category:","填充分類變數（基於Imputer的自定義填充器，用眾數填充）")
        print("-",category.title)
        print(category.answer)
        print("\n")
        
        #imputer
        imputer=ShortAnswerQuestions("imputer:","Utilize the imputer to fill in missing values.")
        print("-",imputer.title)
        print(imputer.answer)
        #dummify
        dummify=ShortAnswerQuestions("dummify:","Dummify our categorical columns.")
        print("-",dummify.title)
        print(dummify.answer)
        #encode
        encode=ShortAnswerQuestions("encode:","Encode the ordinal_column.")
        print("-",encode.title)
        print(encode.answer)
        #cut
        cut=ShortAnswerQuestions("encode:","Bucket the quantitative_column.")
        print("-",cut.title)
        print(cut.answer)
        
        #Q5 END



