
# coding: utf-8

# # Predict Major League Baseball (MLB) game outcomes - win/lose
# ## By Wes Harbert
# ### MSDS 692 Data Science Practicum

# In[1]:


import os 

import glob
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

# ann analysis packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras import backend as K

K.set_image_dim_ordering( 'tf' )


# ## Import data

# In[2]:


#data source is retrosheet.org:  http://www.retrosheet.org/gamelogs/index.html. I am using data from all MLB games for
#for the 1960 - 2017 seasons. Each season was downladed as a '.txt' file with each data record separated by a comma.

#local path
path1 = 'baseball/data/seasons/'
path2 = 'baseball/data/headers.txt'

#import all 
files = glob.glob(path1 + "/*.TXT")
seasons = []
for file_ in files:
    df = pd.read_csv(file_, sep=',', index_col = False, encoding = 'latin1')
    seasons.append(df)

#data field descriptions downloaded as separate file. I am saving the headers for later use.    
headers = pd.read_csv(path2, sep='\n', header = None)



# In[3]:


#number of seasons in data set
len(seasons)


# In[4]:


#number of records and features for each season
[season.shape for season in seasons]


# In[5]:


#note, all seasons have the same number of features - which is good!


# In[6]:


headers.head()


# In[7]:


#header shape matches number of features in data
headers.shape


# ## Data field descriptions 
# (for reference)
(from http://www.retrosheet.org/gamelogs/glfields.txt)

Field(s)  Meaning
    1     Date in the form "yyyymmdd"
    2     Number of game:
             "0" -- a single game
             "1" -- the first game of a double (or triple) header
                    including seperate admission doubleheaders
             "2" -- the second game of a double (or triple) header
                    including seperate admission doubleheaders
             "3" -- the third game of a triple-header
             "A" -- the first game of a double-header involving 3 teams
             "B" -- the second game of a double-header involving 3 teams
    3     Day of week  ("Sun","Mon","Tue","Wed","Thu","Fri","Sat")
  4-5     Visiting team and league
    6     Visiting team game number
          For this and the home team game number, ties are counted as
          games and suspended games are counted from the starting
          rather than the ending date.
  7-8     Home team and league
    9     Home team game number
10-11     Visiting and home team score (unquoted)
   12     Length of game in outs (unquoted).  A full 9-inning game would
          have a 54 in this field.  If the home team won without batting
          in the bottom of the ninth, this field would contain a 51.
   13     Day/night indicator ("D" or "N")
   14     Completion information.  If the game was completed at a
          later date (either due to a suspension or an upheld protest)
          this field will include:
             "yyyymmdd,park,vs,hs,len" Where
          yyyymmdd -- the date the game was completed
          park -- the park ID where the game was completed
          vs -- the visitor score at the time of interruption
          hs -- the home score at the time of interruption
          len -- the length of the game in outs at time of interruption
          All the rest of the information in the record refers to the
          entire game.
   15     Forfeit information:
             "V" -- the game was forfeited to the visiting team
             "H" -- the game was forfeited to the home team
             "T" -- the game was ruled a no-decision
   16     Protest information:
             "P" -- the game was protested by an unidentified team
             "V" -- a disallowed protest was made by the visiting team
             "H" -- a disallowed protest was made by the home team
             "X" -- an upheld protest was made by the visiting team
             "Y" -- an upheld protest was made by the home team
          Note: two of these last four codes can appear in the field
          (if both teams protested the game).
   17     Park ID
   18     Attendance (unquoted)
   19     Time of game in minutes (unquoted)
20-21     Visiting and home line scores.  For example:
             "010000(10)0x"
          Would indicate a game where the home team scored a run in
          the second inning, ten in the seventh and didn't bat in the
          bottom of the ninth.
22-38     Visiting team offensive statistics (unquoted) (in order):
             at-bats
             hits
             doubles
             triples
             homeruns
             RBI
             sacrifice hits.  This may include sacrifice flies for years
                prior to 1954 when sacrifice flies were allowed.
             sacrifice flies (since 1954)
             hit-by-pitch
             walks
             intentional walks
             strikeouts
             stolen bases
             caught stealing
             grounded into double plays
             awarded first on catcher's interference
             left on base
39-43     Visiting team pitching statistics (unquoted)(in order):
             pitchers used ( 1 means it was a complete game )
             individual earned runs
             team earned runs
             wild pitches
             balks
44-49     Visiting team defensive statistics (unquoted) (in order):
             putouts.  Note: prior to 1931, this may not equal 3 times
                the number of innings pitched.  Prior to that, no
                putout was awarded when a runner was declared out for
                being hit by a batted ball.
             assists
             errors
             passed balls
             double plays
             triple plays
50-66     Home team offensive statistics
67-71     Home team pitching statistics
72-77     Home team defensive statistics
78-79     Home plate umpire ID and name
80-81     1B umpire ID and name
82-83     2B umpire ID and name
84-85     3B umpire ID and name
86-87     LF umpire ID and name
88-89     RF umpire ID and name
          If any umpire positions were not filled for a particular game
          the fields will be "","(none)".
90-91     Visiting team manager ID and name
92-93     Home team manager ID and name
94-95     Winning pitcher ID and name
96-97     Losing pitcher ID and name
98-99     Saving pitcher ID and name--"","(none)" if none awarded
100-101   Game Winning RBI batter ID and name--"","(none)" if none
          awarded
102-103   Visiting starting pitcher ID and name
104-105   Home starting pitcher ID and name
106-132   Visiting starting players ID, name and defensive position,
          listed in the order (1-9) they appeared in the batting order.
133-159   Home starting players ID, name and defensive position
          listed in the order (1-9) they appeared in the batting order.
  160     Additional information.  This is a grab-bag of informational
          items that might not warrant a field on their own.  The field 
          is alpha-numeric. Some items are represented by tokens such as:
             "HTBF" -- home team batted first.
             Note: if "HTBF" is specified it would be possible to see
             something like "01002000x" in the visitor's line score.
          Changes in umpire positions during a game will also appear in 
          this field.  These will be in the form:
             umpchange,inning,umpPosition,umpid with the latter three
             repeated for each umpire.
          These changes occur with umpire injuries, late arrival of 
          umpires or changes from completion of suspended games. Details
          of suspended games are in field 14.
  161     Acquisition information:
             "Y" -- we have the complete game
             "N" -- we don't have any portion of the game
             "D" -- the game was derived from box score and game story
             "P" -- we have some portion of the game.  We may be missing
                    innings at the beginning, middle and end of the game.
 
Missing fields will be NULL.

# ## Format Data

# In[8]:


#name columns per field descriptions
for season in seasons:
    season.columns = headers.iloc[:,0]


# ### Subset features

# In[9]:


#save team lables - will be removed from the core feature set, but needed later to sort by team and compute statistics
home_vis_labels_list = []
for season in seasons:
    home_vis_labels_list.append(season.loc[:,['visiting_team','home_team']])


# In[10]:


home_vis_labels_list[0].head()


# In[11]:


#save data for building target labels (game scores)
target_label_base_list = []
for season in seasons:
    target_label_base_list.append(season.iloc[:,9:11])


# In[12]:


target_label_base_list[0].head()


# In[13]:


#innitial feature slection - remove features that are not measures of performance suitable for prediction. 
#selection based on personal judgement and some preliminary model interations. 
seasons_subset = []
for season in seasons:
    season = season.iloc[:,3:77]
    season = season.drop(['outs_in_game_54_standard','attendance','duration_in_minutes','visiting_league',
                          'home_league','day_night','completion_info','forfeit_info','protest_info', 'park_ID',
                          'vis_score_by_inning', 'home_score_by_inning','visiting_team','home_team',
                          'visiting_team_score','home_team_score','home_team_game_number','vis_team_game_number'], 
                         axis = 1)
    seasons_subset.append(season)
seasons = seasons_subset    


# In[14]:


seasons[0].shape


# ### Check data for NaN values

# In[15]:


sum([season.isnull().sum() for season in seasons])


# In[16]:


#...NaN detected


# In[17]:


ssn_idx = 0
for season in seasons: 
    print(ssn_idx)
    null_columns=season.columns[season.isnull().any()]
    print(season[season.isnull().any(axis=1)][null_columns].head()) 
    ssn_idx += 1


# In[18]:


#Season 11, record 1131 contains NaN values
seasons[11].iloc[1131,:].head()


# In[19]:


seasons[11].shape


# In[20]:


#drop game with NaN, also drop game from teams label list
seasons[11] = seasons[11].drop(seasons[11].index[1131])
home_vis_labels_list[11] = home_vis_labels_list[11].drop(home_vis_labels_list[11].index[1131])
seasons[11].iloc[1130:1133,:]


# In[21]:


#one less record in season 11 now
seasons[11].shape


# In[22]:


#check for NaN again, to confirm
sum([season.isnull().sum() for season in seasons])


# ## Create target class labels

# In[23]:


target_label_base_list[11].shape


# In[24]:


#before computing target labels, remove record from game eleven that containted NaN values
target_label_base_list[11] = target_label_base_list[11].drop(target_label_base_list[11].index[1131])


# In[25]:


target_label_base_list[11].shape


# In[26]:


target_label_base_list[0].columns


# In[27]:


#check for nullls
sum([labels.isnull().sum() for labels in target_label_base_list])


# ### Binary data labels: visitor win = 0, home win = 1

# In[28]:


#visitor wins = 0
#home wins = 1
target_labels_list = []
for label_base in target_label_base_list:
    label = pd.DataFrame(np.where(label_base["visiting_team_score"] > label_base["home_team_score"], 0,1))
    label.columns = ['winner']
    target_labels_list.append(label)


# In[29]:


[labels.hist() for labels in target_labels_list]    


# These histograms show a distinct pattern, the home team wins more than the visting team

# In[126]:


#percent home wins for all games
all_scores = np.array(pd.concat(target_label_base_list))
home_wins = 0
for score in all_scores:
    if score[1] > score[0]:
        home_wins += 1
round(home_wins/len(all_scores)*100)


# ## Remove features with low variance

# In[30]:


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


# In[31]:


seasons_trimmed = []
for season in seasons:
    season = variance_threshold_selector(season)
    seasons_trimmed.append(season)
seasons = seasons_trimmed


# In[32]:


#check number of features for each season
[season.shape for season in seasons] 


# In[33]:


#check that each season has the same features: total unique features should match individual season feature count
features_all = []
for season in seasons:
    for feat_name in season.columns:
        features_all.append(feat_name)
len(set(features_all))        


# In[34]:


#season feature count varies accross seasons - the following code removes features that are not present in all seasons

#count the frequency of each feature accross all seasons
col_counts = {}
for season in seasons:
    for col in season.columns:
        col_counts[col] = col_counts.get(col,0) + 1    
        
#if feature matches the max feature frequency, add it to keeper list
feat_list = []
for key, value in col_counts.items():
    if value == max(list(col_counts.values())):
        feat_list.append(key)


# In[35]:


#the filtered feature list
feat_list


# In[36]:


#use filtered feature list to select standard features accross all seasons
for i in range(len(seasons)): 
    seasons[i] = seasons[i].loc[:,feat_list]


# In[37]:


[season.shape for season in seasons]


# In[38]:


#check that each season has the same features: total unique features should match individual season feature count
features_all = []
for season in seasons:
    for feat_name in season.columns:
        features_all.append(feat_name)
len(set(features_all)) 


# confirmed

# In[39]:


#select visitor feature names using 'vis_' prefix
vis_cols = [col for col in seasons[0] if col.startswith('vis_')]
del vis_cols[0]


# In[40]:


vis_cols


# In[41]:


#select home feature names using 'home_' prefix
home_cols = [col for col in seasons[0] if col.startswith('home_')]
del home_cols[0]


# In[42]:


home_cols


# In[43]:


print(len(vis_cols))
print(len(home_cols))


# In[44]:


#check if home and visitor feature sets match
vis_suffixlist = []
home_suffixlist = []
[vis_suffixlist.append(col.replace('vis_','')) for col in vis_cols]
[home_suffixlist.append(col.replace('home_','')) for col in home_cols]
set(vis_suffixlist)==set(home_suffixlist)


# ## Transform data distributions

# In[45]:


#transform features so that distributions are normal
seasons = [np.log(season +1) for season in seasons]
#seasons = [np.arcsinh(season) for season in seasons]


# ## More feature selection - now with random forest

# In[46]:


#combine seasons in single dataframe, for use in random forest feature selection below
X_feat = pd.concat(seasons)
Y_feat = pd.concat(target_labels_list)
Y_feat = Y_feat.values.ravel()


# In[47]:


X_feat.shape


# In[48]:


Y_feat.shape


# In[49]:


# Use random forest to compute feature importances
forest = ExtraTreesClassifier(n_estimators=500)
forest.fit(X_feat, Y_feat)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_feat.shape[1]):
    print("%d. %s (%f)" % (f + 1, X_feat.columns[indices[f]], importances[indices[f]]))


# In[50]:


# Plot the feature importances of the forest - this is not my code, but I can't find where I took it from to provide
#a reference
plt.figure(figsize=(18,14))
plt.title("Feature importances")
plt.bar(range(X_feat.shape[1]), importances[indices], color="g", yerr=std[indices], align="center")
plt.xticks(range(X_feat.shape[1]), indices)
plt.xlim([-1, X_feat.shape[1]])
plt.show()


# In[51]:


#select top 9 features
top_feat = seasons[0].iloc[:,indices[0:9]].columns


# In[52]:


X_feat.loc[:,top_feat].hist(figsize=(20,18), bins=10)


# In[53]:


#reduce features in all seasons per features selected by random forest feature importance
seasons = [season.loc[:,top_feat] for season in seasons]


# In[54]:


#combine seasons with reduced reduced feature sets
pd.concat(seasons).shape


# In[55]:


seasons[0].head()


# ## Compute team statistics for use in models

# In[56]:


#redifine these variables based on reduced features (visitor or home team)
vis_cols = [col for col in seasons[0] if col.startswith('vis_')]
home_cols = [col for col in seasons[0] if col.startswith('home_')]


# In[57]:


#function for computing statistics. 
def season_stats(df, groupby_str, col_str, window):
    #rolling mean (or moving average)
    df_mean = df.groupby(groupby_str)[col_str].rolling(window).mean()
    df_mean.fillna(method='bfill', inplace = True)
    df_mean.index = df_mean.index.droplevel()
    df = df.join(df_mean, rsuffix= '_ma')
    
    #expanding mean
    df_mean_x = df.groupby(groupby_str)[col_str].expanding().mean()
    df_mean_x.fillna(method='bfill', inplace = True)
    df_mean_x.index = df_mean_x.index.droplevel()
    df = df.join(df_mean_x, rsuffix= '_ma_x')
    
    #rolling median
    df_median = df.groupby(groupby_str)[col_str].rolling(window).median()
    df_median.fillna(method='bfill', inplace = True)
    df_median.index = df_median.index.droplevel()
    df = df.join(df_median, rsuffix= '_mmed')
    
    #expanding median
    df_median_x = df.groupby(groupby_str)[col_str].expanding().median()
    df_median_x.fillna(method='bfill', inplace = True)
    df_median_x.index = df_median_x.index.droplevel()
    df = df.join(df_median_x, rsuffix= '_mmed_x')
    
    #rolling standard deviation
    df_std = df.groupby(groupby_str)[col_str].rolling(window).std()
    df_std.fillna(method='bfill', inplace = True)
    df_std.index = df_std.index.droplevel()
    df = df.join(df_std, rsuffix= '_mv_sd')
    
    #expanding standard deviation
    df_std_x = df.groupby(groupby_str)[col_str].expanding(window).std()
    df_std_x.fillna(method='bfill', inplace = True)
    df_std_x.index = df_std_x.index.droplevel()
    df = df.join(df_std_x, rsuffix= '_mv_sd_x')
    
    return df


# In[58]:


# append team labels for each game in each season 
for i in range(len(seasons)):
    season = home_vis_labels_list[i].join(seasons[i])
    seasons[i] = season


# In[59]:


#compute home team statistics
for i in range(len(seasons)): 
    for col in home_cols:
        seasons[i] = season_stats(seasons[i],'home_team', col, 5)     


# In[60]:


#compute visiting team statistics
for i in range(len(seasons)): 
    for col in vis_cols:
        seasons[i] = season_stats(seasons[i],'visiting_team', col, 5)   


# In[61]:


#combine all win/lose target labels for all seeasons (single data frame)
Y = pd.concat(target_labels_list)
Y.shape


# In[62]:


#drop features used to compute statistics - will only use statistics for models
feat_drop = list(top_feat)
feat_drop.append('visiting_team')
feat_drop.append('home_team')
feat_drop


# In[63]:


#combine seasons into single dataframe for modeling
X = pd.concat(seasons)
X = X.drop(feat_drop, 
            axis=1)   
X.shape


# In[64]:


X.columns


# ## Random forest to reduce dimensionality...again

# In[65]:


# Use random forest to compute feature importances on new feature set
forest = ExtraTreesClassifier(n_estimators=500)

Y_feat = Y.values.ravel()
forest.fit(X, Y_feat)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))


# In[66]:


# Plot the feature importances of the forest
plt.figure(figsize=(18,14))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[67]:


#select top features
top_feat = X.iloc[:,indices[0:9]].columns


# In[68]:


top_feat


# In[69]:


X.loc[:,top_feat].hist(figsize=(20,18), bins=11)


# In[70]:


#reduce data to only include most important features
X = X.loc[:,top_feat]


# In[71]:


#check that statistics computations did not produce NaN values
X.isnull().sum()


# ## Data ready for use in precdiction efforts

# In[72]:


X.shape


# In[73]:


Y.shape


# ## Predict with: Logistic Regression, Random Forest, AdaBoost, and Ensemble voting

# In[74]:


Y = Y.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#classifiers
num = 500
clf1 = LogisticRegression()
clf2 = RandomForestClassifier(n_estimators=num)
clf3 = AdaBoostClassifier(n_estimators=num)
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ab', clf3)], voting='hard')

#fit models
clf1 = clf1.fit(X_train,y_train)
clf2 = clf2.fit(X_train,y_train)
clf3 = clf3.fit(X_train,y_train)
eclf = eclf.fit(X_train,y_train)

#(code based on snippet from: http://scikit-learn.org/stable/modules/ensemble.html)


# ## Results (Logistic Regression, Random Forest, AdaBoost, Ensemble voting)

# In[75]:


for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'AdaBoost', 'Ensemble']):
     scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


# ## Predict with ANN

# In[83]:


def ANN_model():
    model = Sequential()
    model.add(Dense(20, activation='relu', input_dim=9))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[84]:


model_ANN = ANN_model()
# compile using cross entropy since this problem is a binary classification problem. 
# using adam optimizer 
model_ANN.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['accuracy'])
model_ANN.summary()


# In[85]:


ANN_log = model_ANN.fit(X, Y, validation_split = 0.2, epochs=500, batch_size=100, shuffle = True, verbose=2)


# In[86]:


#define accuracy & validation accuracy plot function
#print(model_record.history.keys())

def plot_model_accuracy(fit_model_obj, title):
    plt.plot(fit_model_obj.history['acc'], 'b')
    plt.plot(fit_model_obj.history['val_acc'], 'g')
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(["train","val"], loc='lower right')
    plt.show()

    #define loss plot function
def plot_model_loss(fit_model_obj, title):
    plt.plot(fit_model_obj.history['loss'],'b')
    plt.plot(fit_model_obj.history['val_loss'], 'g')
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(["train", "val"], loc='upper right')
    plt.show()


# In[87]:


plot_model_accuracy(ANN_log, 'accuracy')


# In[88]:


plot_model_loss(ANN_log, 'loss')


# In[89]:


print("Accuracy: %0.2f [%s]" % (max(ANN_log.history['val_acc']), 'ANN'))

