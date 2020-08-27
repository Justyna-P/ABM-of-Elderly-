
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






preferences = pd.read_excel("Preferences_new.xlsx")  #weights
prod_set = pd.read_excel("Products_pralki.xlsx")  #weights

Agent_ID = preferences.iloc[1,:][0]
Agent_pref = preferences.iloc[1,:][1:]
Prod1 = prod_set.iloc[1,:]
Prod2 = prod_set.iloc[3,:]
included = [0,1,2,3,4,5,6]
included_l = [0,3,4,6]

if 1 in included:
    print("yes")


def shuffle_products(list_to_shuffle):
    pick = random.choice(list_to_shuffle)
    list_to_shuffle.remove(pick)
    return(pick)


def smooth_user_preference(x):
    return math.log(1+x, 2)

def utility_weighted(item_attributes, preferences_weights, included_list):
    utility = float(0)
    for i in range (0, len(item_attributes)):
        #print(i)
        if i in included_list:
            #print(str(1) + "yes")
            utility +=  item_attributes[i] * preferences_weights[i]
    return (utility)

''' testowanie
utility_weighted(Prod1, Agent_pref, included)
utility_weighted(Prod2, Agent_pref, included)

utility_weighted(Prod1, Agent_pref, included_l)
utility_weighted(Prod2, Agent_pref, included_l)
'''


def compare_weighted(item_attributes1, item_attributes2, preferences_weights, included_list):
    U1 = utility_weighted(item_attributes1, preferences_weights, included_list)
    U2 = utility_weighted(item_attributes2, preferences_weights, included_list)
    #print(U1)
    #print(U2)
    if U1>U2:
        return(1)
    else:
        return(2)
        
 '''    testowanie   
compare_weighted(Prod1, Prod2, Agent_pref, included)        
compare_weighted(Prod1, Prod2, Agent_pref, included_l)        
'''
    
def compare_tally(item_attributes1, item_attributes2, preferences_weights, included_list):
    U1 = float(0)
    U2 = float(0)
    for i in range (0, len(item_attributes1)):
        if i in included_list:
            if preferences_weights[i]*item_attributes1[i]>preferences_weights[i]*item_attributes2[i]:
                U1 +=1
            elif preferences_weights[i]*item_attributes1[i]<preferences_weights[i]*item_attributes2[i]:
                U2 +=1
    print(U1)
    print(U2)
    if U1<U2:
        return(2)
    else:
        return(1)
        
 '''    testowanie   
compare_tally(Prod1, Prod2, Agent_pref, included)        
compare_tally(Prod1, Prod2, Agent_pref, included_l)        
'''        
        
        
def compare_ttb(item_attributes1, item_attributes2, preferences_weights, included_list):
    U1 = float(0)
    U2 = float(0)
    for i in range (0, len(item_attributes)):
        if str(i) in included_list:
            if preferences_weights[i]*item_attributes1[i]>preferences_weights[i]*item_attributes2[i]:
                utility1 +=1
            else if preferences_weights[i]*item_attributes1[i]<preferences_weights[i]*item_attributes2[i]:
                utility2 +=1
    if U1>U2:
        return(1)
    else:
        return(2)


def filter_out_below_reference(product_list, preferences_set_r, included_list):
    product_list2 = product_list[product_list['price_scaled']>preferences_set_r[0]]
    if len(list(product_list2.index.values))> 16:
        product_list = product_list2
    #print("prices larger than " + str(preferences_set_r[0]))
    #print()
    for i in range (1, 11):
        if str(i) in included_list:
            product_list2 = product_list[product_list['a'+str(i)]>preferences_set_r[i]]
            #print("a " + str (i) + " larger than " + str(preferences_set_r[i]))
            #print(len(list(product_list2.index.values)))
            if len(list(product_list2.index.values))> 16:
                product_list = product_list2
            #print(len(list(product_list2.index.values)))
    return(list(product_list.index.values))

#listcheck2 = filter_out_below_reference(prodscheck, PREFERENCES_SET_R,preferences_picked)




def get_items_interacted(Client, interactions_df):
    interacted_items = interactions_df.loc[Client]['Product']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


"""output: summary files"""

"""Parameters for simulation"""

"""simulation"""
#
#
#preferences_picked = tuple([0, 1,2,3,4,5,6,7,8,9,10])
#atr = 4
#preferences_picked =preferences_picked[0:10]

RANDOM_SEED = random.randrange(1000,2000,10)
NEW_CUSTOMERS = 3000  # Total number of agents
INTERVAL_CUSTOMERS = 200.0  # Generate new agents roughly every x seconds

PREFERENCES_SET_ALPHA = 0.2
PREFERENCES_SET_BETA = 0.4
PREFERENCES_SET_EPSILON = 0.3
PRODUCTS_SET_MAX = minmax.iloc[1,:]
PRODUCTS_SET_MIN = minmax.iloc[0,:]

#excels_with_agents = ("AgentsControlGroup", "AgentsP9", "AgentsP8", "AgentsP7", "AgentsP6", "AgentsP5", "AgentsP4", "AgentsP3", "AgentsP2", "AgentsP1")
heuristics_to_run = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2,2)
#a=0
        #for a in range(10):
#spreadsheet_name = excels_with_agents[a]+".xlsx"
agents = pd.read_excel("AgentsElderly.xlsx")

#register = pd.DataFrame(columns=['Client','Product', 'Utility_perceived','Utility_real'])
summary = pd.DataFrame(columns=['ProdSet','attributes', 'heuristics','av_r'])
#
#interactions_df = pd.read_csv("results_h0_atr11_prodset2_no_recommendations.csv")
#interactions_df['eventStrength'] = 1

rec_sys_list = ("random", "Popularity", "CF", "CB")


for simulation_n in range(50):
    RANDOM_SEED = random.randrange(1,200,1)
    for prod in range (3):
        products = pd.read_excel(prod_sets[prod])
        for atr in range (2,12):
            if prod ==0:
                start = 0
            else:
                start = 1
            #print(str(count))
            for h in range(start, 14):
                HEURISTICS_STRENGHT = heuristics_to_run[h]
                a=0

                #register = pd.DataFrame(columns=['Client','Product', 'RecSys', 'Utility_perceived','Utility_real'])
                register = pd.DataFrame(columns=['Client','Product', 'RecSys', 'Utility_perceived','Utility_real', 'list_count'])


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
                    #PRODUCTS_V_LIST = list(range(0, products.shape[0]))
                    PREFERENCES_SET_R=preferencesR.iloc[id_number,1:]
                    #PRODUCTS_MAIN_LIST = filter_out_below_reference(products, PREFERENCES_SET_R, preferences_picked)
    #                print (PREFERENCES_SET_R)
    #                print (preferences_picked)
    #                print(PRODUCTS_X_LIST)
                    #print(PRODUCTS_V_LIST)
                    #PREFERENCES_SET_R=preferencesR.iloc[id_number,1:]
                    #PREFERENCES_SET_R=preferencesR.iloc[1,1:]
                    PREFERENCES_SET_A=preferencesA.iloc[id_number,1:]
                    IGNORE_LIST =[]
                    for r in range(1): # 4 different recommendation techniques
                        for m in range(10): #here we can specify how many products each customer should buy
                            #pref_table = pref_table[~pref_table['index'].isin(set_to_ignore)]
                            endurance_local = endurance
                            f = -100
                            f_score = -100
                            n = 0
                            PRODUCTS_CLONE_LIST = filter_out_below_reference(products, PREFERENCES_SET_R, preferences_picked)
                            PRODUCTS_CLONE_LIST = [x for x in PRODUCTS_CLONE_LIST if x not in IGNORE_LIST]
                            while endurance_local >0:
                                now = env.now
                                if r ==0:
                                    p = shuffle_products(PRODUCTS_CLONE_LIST)
    #                            elif r ==1:
    #                                p = popularity_model.recommend_items(0, items_to_ignore=set_to_ignore, topn=15, verbose=False).iloc[n,0]
    #                            elif r ==2:
    #                                p = cf_recommender_model.recommend_items(0, items_to_ignore=set_to_ignore, topn=15, verbose=False).iloc[n,0]
    #                            else:
    #                                p = pref_table.iloc[n,0]
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
    #                        try:
    #                            PRODUCTS_MAIN_LIST.remove(f)
    #                            print("wyszło")
    #                        except:
    #                            print("nie wyszło")
                            IGNORE_LIST.append(f)
                            df2 = pd.DataFrame([[id_number, f, rec_sys_list[r] ,f_score, f_real, len(PRODUCTS_CLONE_LIST) ]], columns=['Client','Product', 'RecSys', 'Utility_perceived','Utility_real', 'list_count'])
                            register = register.append(df2)

                #register.to_csv('diagnose2.csv')

                random.seed(RANDOM_SEED)
                env = simpy.Environment()
                counter = simpy.Resource(env, capacity=200)
                env.process(source(env, NEW_CUSTOMERS, INTERVAL_CUSTOMERS))
                env.run()
                print ('results_h' + str(h) + '_atr' + str(atr) + '_prodset' + str(prod))
                register.to_csv('results_p' + str(h) + '_atr' + str(atr) + '_prodset' + str(prod) + '_no_rec_TG_SEED_nr'+ str(RANDOM_SEED) +'.csv')
                av_r = register.loc[register['RecSys'] == "random"]["Utility_real"].mean()
                df3 = pd.DataFrame([[prod, atr, h, av_r ]], columns=['ProdSet','attributes', 'heuristics','av_r'])
                summary = summary.append(df3)
    summary.to_csv('summary_prod_filter'+ '_TG_SEED_nr'+ str(RANDOM_SEED)+ '.csv')
