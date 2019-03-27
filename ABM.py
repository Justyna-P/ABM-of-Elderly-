


import simpy
import random
import pandas as pd
import random
import numpy as np

import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt





#
#global products
#global preferences
#global minmax
#products = pd.read_excel("Products1.xlsx") 

preferencesR = pd.read_excel("PreferencesR.xlsx")  #reservation level
preferencesA = pd.read_excel("PreferencesA.xlsx")  #aspiration levels
minmax = pd.read_excel("Minmax.xlsx")  #global minimum and maximum
prod_sets = ["Products1.xlsx", "Products2.xlsx", "Products3.xlsx"] #different product sets
#preferencesR.iloc[1, 1:]

def shuffle_products(list_to_shuffle):
    pick = random.choice(list_to_shuffle)
    list_to_shuffle.remove(pick)
    return(pick)

    
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
def utility(attributes, preferences_set_a, preferences_set_r, included_list):
    u_total = float(0)
    u_min = float(100)
    for i in range (0, len(attributes)):
        if str(i) in included_list:
            if attributes[i] > preferences_set_a[i]:
                u_i = 1+ PREFERENCES_SET_ALPHA*(attributes[i]-preferences_set_r[i])/(PRODUCTS_SET_MAX[i]-preferences_set_a[i])
            elif attributes[i] >preferences_set_r[i]:
                u_i = (attributes[i]-preferences_set_r[i])/(preferences_set_a[i] -preferences_set_r[i])
            else:
                u_i = float(PREFERENCES_SET_BETA)*(attributes[i]-preferences_set_r[i])/(preferences_set_r[i]-PRODUCTS_SET_MIN[i])
            u_min = min(u_min, u_i )
            u_total = float(u_total) + float(u_i)
    u_total = PREFERENCES_SET_EPSILON*u_total/(1+len(included_list)*PREFERENCES_SET_EPSILON)
    result = u_min + u_total
    return (result)

def get_items_interacted(Client, interactions_df):
    interacted_items = interactions_df.loc[Client]['Product']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, Client, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['Product'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'Product', 
                                                          right_on = 'Product')#[['eventStrength', 'Product', 'Brand', 'Ratings']]


        return recommendations_df



class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, Client, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[Client].sort_values(ascending=False) \
                                    .reset_index().rename(columns={Client: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['Product'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'Product', 
                                                          right_on = 'Product')[['recStrength', 'Product', 'title', 'url', 'lang']]


        return recommendations_df


"""output: summary files"""

"""Parameters for simulation"""

"""simulation"""
#
#
#preferences_picked = tuple([0, 1,2,3,4,5,6,7,8,9,10])
#atr = 4
#preferences_picked =preferences_picked[0:10]

RANDOM_SEED = 42
NEW_CUSTOMERS = 3000  # Total number of agents
INTERVAL_CUSTOMERS = 200.0  # Generate new agents roughly every x seconds

PREFERENCES_SET_ALPHA = 0.2
PREFERENCES_SET_BETA = 0.4
PREFERENCES_SET_EPSILON = 0.3
PRODUCTS_SET_MAX = minmax.iloc[1,:]
PRODUCTS_SET_MIN = minmax.iloc[0,:]

#excels_with_agents = ("AgentsControlGroup", "AgentsP9", "AgentsP8", "AgentsP7", "AgentsP6", "AgentsP5", "AgentsP4", "AgentsP3", "AgentsP2", "AgentsP1")
heuristics_to_run = (0, 0.25, 0.5)
#a=0
        #for a in range(10):
#spreadsheet_name = excels_with_agents[a]+".xlsx"
agents = pd.read_excel("AgentsElderly.xlsx") 

#register = pd.DataFrame(columns=['Client','Product', 'Utility_perceived','Utility_real'])
summary = pd.DataFrame(columns=['ProdSet','attributes', 'heuristics','av_r', 'av_p', 'av_cf', 'av_cb'])
#
#interactions_df = pd.read_csv("results_h0_atr11_prodset2_no_recommendations.csv") 
#interactions_df['eventStrength'] = 1

rec_sys_list = ("random", "Popularity", "CF", "CB")


for prod in range (3):
    products = pd.read_excel(prod_sets[prod])
    
    items_df = products
    items_df = items_df.reset_index()
    items_df = items_df.rename(columns = {'index':'Product'})
    
    ds2 =products.drop(columns=['ID'])
    cosine_similarities = cosine_similarity(ds2, ds2)
    similaritems = {}
    for idx in range(ds2.shape[0]):
        similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
        similaritems[idx] = similar_indices[1:]
    items_df = products
    items_df = items_df.reset_index()
    items_df = items_df.rename(columns = {'index':'Product'})
    
    for atr in range (2,12):
        if prod ==0:
            start = 0
        else:
            start = 1
        #print(str(count))
        for h in range(start, 3):
            HEURISTICS_STRENGHT = heuristics_to_run[h]
            a=0
            
            interactions_df = pd.read_csv('results_h' + str(h) + '_atr' + str(atr) + '_prodset' + str(prod) + '_no_recommendations.csv') 
            interactions_df['eventStrength'] = 1
            interactions_full_df = interactions_df \
                    .groupby(['Client', 'Product'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
            interactions_full_indexed_df = interactions_full_df.set_index('Client')
            item_popularity_df = interactions_full_df.groupby('Product')['eventStrength'].sum().sort_values(ascending=False).reset_index()
            popularity_model = PopularityRecommender(item_popularity_df, items_df)
            
            users_items_pivot_matrix_df = interactions_full_df.pivot(index='Client', 
                                                          columns='Product', 
                                                          values='eventStrength').fillna(0)
            users_items_pivot_matrix_df.head(10)
            users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
            users_ids = list(users_items_pivot_matrix_df.index)
            NUMBER_OF_FACTORS_MF = 15
            U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
            sigma = np.diag(sigma)
            all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
            cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
            cf_recommender_model = CFRecommender(cf_preds_df, items_df)          
            
            #for a in range(10):
    #        spreadsheet_name = excels_with_agents[a]+".xlsx"
    #        agents = pd.read_excel(spreadsheet_name) 
            #name1 = "run_v"+ str(h) + "_" + excels_with_agents[a] +"_A.csv"
            #name2 = "run_v"+ str(h) + "_" + excels_with_agents[a] +"_B.csv"
            register = pd.DataFrame(columns=['Client','Product', 'RecSys', 'Utility_perceived','Utility_real'])
            #plist =  pd.DataFrame(columns=['Client','N', 'Product'])


     
            
            def source(env, number, interval):
                """Source generates customers randomly"""
                for i in range(number):
                    agents_list=agents.iloc[i,5]
                    speed = agents.iloc[i,3]
                    endurance = agents.iloc[i,4]
                    elderly = agents.iloc[i,1]
                    c = customer(env, i, elderly, speed, endurance, agents_list)
                    env.process(c)
                    yield env.timeout(interval)
            
    #id_number =0        
    #5 in preferences_full
                """Customer arrives, looks for products until has endurance and leaves"""
            def customer(env, id_number ,elderly, speed, endurance, preference_list):
                global register
                global atr
                global interactions_df
                #global plist
                arrive = env.now
                i = 0
                #f = -100
                #f_score = -100
                preferences_all = tuple(preference_list.split(","))
                preferences_picked = preferences_all [0:atr]
                # preferences_full = tuple([0,1,2,3,4,5,6,7,8,9,10])
                PRODUCTS_V_LIST = list(range(0, products.shape[0]))
                PREFERENCES_SET_R=preferencesR.iloc[id_number,1:]
                PREFERENCES_SET_A=preferencesA.iloc[id_number,1:]
                set_to_ignore = get_items_interacted(id_number, interactions_full_indexed_df)

                items_purchased = pd.DataFrame(list(set_to_ignore))
                items_purchased =items_purchased.set_index(0)
                
                items_purchased = list(set_to_ignore)

                pref_table = pd.DataFrame()
                for i in range(len(items_purchased)):
                    s_items = pd.DataFrame(similaritems[items_purchased[i]][:50])
                    pref_table = pref_table.append(s_items)
                pref_table = pref_table[0].value_counts()
                pref_table = pref_table.reset_index().drop(columns=[0])


                for r in range(4): # 4 different recommendation techniques
                    for m in range(1): #here we can specify how many products each customer should buy
                        pref_table = pref_table[~pref_table['index'].isin(set_to_ignore)] 
                        endurance_local = endurance
                        f = -100
                        f_score = -100
                        n = 0
                        while endurance_local >0:
                            now = env.now
                            if r ==0:
                                p = shuffle_products(PRODUCTS_V_LIST)
                            elif r ==1:
                                p = popularity_model.recommend_items(0, items_to_ignore=set_to_ignore, topn=15, verbose=False).iloc[n,0]
                            elif r ==2:
                                p = cf_recommender_model.recommend_items(0, items_to_ignore=set_to_ignore, topn=15, verbose=False).iloc[n,0]
                            else:
                                p = pref_table.iloc[n,0]
                            #df4 = pd.DataFrame([[id_number, r, p ]], columns=['Client','N', 'Product'])
                            #plist = plist.append(df4)
                            attributes_test = products.iloc[p, 3:]
                            heuristics = products.iloc[p, 1:3]
                            x = utility(attributes_test,PREFERENCES_SET_A,PREFERENCES_SET_R,preferences_picked)
                            x_real =  utility(attributes_test,PREFERENCES_SET_A,PREFERENCES_SET_R, preferences_all)
                            '''heuristics'''
                            if elderly == 1:
                                if heuristics[1] == 1:
                                    x = x - HEURISTICS_STRENGHT # if both heuristic activated, act only on fear (bad review firs)
                                elif heuristics[0] == 1:
                                    x = x + HEURISTICS_STRENGHT
                            endurance_local = endurance_local - speed
                            n=n+1
                            if x > f_score:
                                f_score= x
                                f = p
                                f_real = x_real
                            yield env.timeout(speed)
                        set_to_ignore.add(f)
                        df2 = pd.DataFrame([[id_number, f, rec_sys_list[r] ,f_score, f_real ]], columns=['Client','Product', 'RecSys', 'Utility_perceived','Utility_real'])
                        register = register.append(df2)
                    
            
            
            # Setup and start the simulation
            random.seed(RANDOM_SEED)
            env = simpy.Environment()
            # Start processes and run
            counter = simpy.Resource(env, capacity=200)
            env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS))
            env.run()
            #plist.to_csv('plist_h' + str(h) + '_atr' + str(atr) + '_prodset' + str(prod) + '_no_recommendations.csv')
            print ('results_h' + str(h) + '_atr' + str(atr) + '_prodset' + str(prod))
            register.to_csv('results_h' + str(h) + '_atr' + str(atr) + '_prodset' + str(prod) + '_after_recommendations_G.csv')
            av_r = register.loc[register['RecSys'] == "random"]["Utility_real"].mean()
            av_p = register.loc[register['RecSys'] == "Popularity"]["Utility_real"].mean()
            av_cf = register.loc[register['RecSys'] == "CF"]["Utility_real"].mean()
            av_cb = register.loc[register['RecSys'] == "CB"]["Utility_real"].mean()
            df3 = pd.DataFrame([[prod, atr, h, av_r, av_p, av_cf, av_cb ]], columns=['ProdSet','attributes', 'heuristics','av_r', 'av_p', 'av_cf', 'av_cb'])
            summary = summary.append(df3)

#        age = agents[['Elderly']]
#        results = pd.concat([register.reset_index(), age], axis=1)
#        summary = pd.pivot_table(results, values='Client', columns = ['Elderly'], index=['Product'], aggfunc=pd.Series.nunique)
summary.to_csv('summary_1prod.csv')



#("random", "Popularity", "CF")
#av_r = register.loc[register['RecSys'] == "random"]["Utility_real"].mean()
#av_p = register.loc[register['RecSys'] == "Popularity"]["Utility_real"].mean()
#av_c = register.loc[register['RecSys'] == "CF"]["Utility_real"].mean()


#        
#        
#        '''run with results from the system'''
#        
#        register2 = pd.DataFrame(columns=['Time','Client','Action','Product', 'Result', 'Favourite_so_far', 'Endurance'])
#        
#        #clear()
#        def source(env, number, interval):
#            """Source generates customers randomly"""
#            for i in range(number):
#                agents_list=agents.iloc[i,5]
#                speed = agents.iloc[i,3]
#                endurance = agents.iloc[i,4]
#                elderly = agents.iloc[i,1]
#                #agents_list = agents_list.split(",")
#                c = customer(env, 'Customer%04d' % i,elderly, speed, endurance, agents_list)
#                env.process(c)
#                yield env.timeout(interval)
#        
#        products_list=[]
#        
#        """Customer arrives, looks for products until has endurance and leaves"""
#        def customer(env, name,elderly, speed, endurance, preference_list):
#            global register2
#            global products_list
#            arrive = env.now
#            i = 0
#            f = -100
#            f_score = -100
#            preferences_picked = tuple(preference_list.split(","))
#            while endurance >0:
#                now = env.now
#                if elderly ==1:
#                    products_list = summary.sort_values([1], ascending=False)
#                    products_list=products_list[1]#.index()
#                if elderly ==0:
#                    products_list = summary.sort_values([0], ascending=False)
#                    products_list=products_list[0]#.index()    
#                products_list = products_list.index.get_values()
#                p = products_list[i] 
#                attributes_test = products.iloc[p, 3:]
#                heuristics = products.iloc[p, 1:3]
#                x = utility(attributes_test,PREFERENCES_SET_A,PREFERENCES_SET_R,preferences_picked  )
#                '''heuristics'''
#                if elderly == 1:
#                    if heuristics[0] == 1:
#                        x = x + HEURISTICS_STRENGHT
#                    if heuristics[1] == 1:
#                        x = x - HEURISTICS_STRENGHT 
#                endurance = endurance - speed
#                if x > f_score:
#                    f_score= x
#                    f = p
#                i = i+1
#                yield env.timeout(speed)
#            df2 = pd.DataFrame([[env.now, name,"leaving the shop with a product", f, f_score ]], columns=['Time','Client','Action','Product','Result'])
#            register2 = register2.append(df2)
#                
#        
#        
#        # Setup and start the simulation
#        print('Recommender System')
#        random.seed(RANDOM_SEED)
#        env = simpy.Environment()
#        
#        # Start processes and run
#        counter = simpy.Resource(env, capacity=200)
#        env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS))
#        env.run()
#        
#        
#        age = agents[['Elderly']]
#        results = pd.concat([register2.reset_index(), age], axis=1)
#        summary2 = pd.pivot_table(results, values='Client', columns = ['Elderly'], index=['Product'], aggfunc=pd.Series.nunique)
#        summary2.to_csv(name2)


#register2.to_csv('run2_results17.02_after_recommendations.csv')

