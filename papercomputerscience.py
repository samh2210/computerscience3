# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:23:19 2022

@author: My HP
"""
# use 63% of observations for bootstrapping, bootstrap 5,10 or 20 times with that 63%
# other 37% is the test data and is used after we optimized hyperparameters alpha beta gamma and delta

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools
import collections
import math
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from array import array
from more_itertools import locate
from sklearn.model_selection import GridSearchCV
from math import floor
from collections import Counter
from sklearn.model_selection import RepeatedStratifiedKFold
from Levenshtein import distance as lev
from typing import Iterable, Any
from itertools import product

with open(r"C:\Users\My HP\Documents\TVs-all-merged.json") as f:
   data = json.load(f)
test = dict(data)
keys = data.keys()
k_list = list(data.keys())
z = []
for i in range(len(data)):
        z.append(data[k_list[i]][0]['shop'])
        if len(data[k_list[i]]) >=2:
            z.append(data[k_list[i]][1]['shop'])
            if len(data[k_list[i]]) >=3:
                z.append(data[k_list[i]][2]['shop'])
                if len(data[k_list[i]]) >=4:
                    z.append(data[k_list[i]][3]['shop'])
m = []
for i in range(len(data)):
        m.append(data[k_list[i]][0]['title'])
        if len(data[k_list[i]]) >=2:
            m.append(data[k_list[i]][1]['title'])
            if len(data[k_list[i]]) >=3:
                m.append(data[k_list[i]][2]['title'])
                if len(data[k_list[i]]) >=4:
                    m.append(data[k_list[i]][3]['title'])               
o = []
for i in range(len(data)):
        o.append(data[k_list[i]][0]['modelID'])
        if len(data[k_list[i]]) >=2:
            o.append(data[k_list[i]][1]['modelID'])
            if len(data[k_list[i]]) >=3:
                o.append(data[k_list[i]][2]['modelID'])
                if len(data[k_list[i]]) >=4:
                    o.append(data[k_list[i]][3]['modelID'])            
y = []
for i in range(len(data)):
        y.append(data[k_list[i]][0]['featuresMap'])
        if len(data[k_list[i]]) >=2:
            y.append(data[k_list[i]][1]['featuresMap'])
            if len(data[k_list[i]]) >=3:
                y.append(data[k_list[i]][2]['featuresMap'])
                if len(data[k_list[i]]) >=4:
                    y.append(data[k_list[i]][3]['featuresMap'])      

    
    
#print(m)
l = np.array(m)
#l = np.delete(l, (950), axis=0)
l_new = np.reshape(l, (len(m),1))
l_2 = np.char.replace(l_new, '"', '-Inch')
l_3 = np.char.replace(l_2,' Hz', 'Hz')
l_4 = np.char.replace(l_3, ' In.', '-Inch')
l_5 = np.char.replace(l_4,'HZ', 'Hz')
l_6 = np.char.replace(l_5, 'in.', '-Inch')
l_7 = np.char.replace(l_6,'hz', 'Hz')
l_8 = np.char.replace(l_7,'\x99', '')
l_9 = np.char.replace(l_8, '-', '')
l_10 = np.char.replace(l_9, '&', '')
l_11 = np.char.replace(l_10, '/', '')
l_110 = l_11.tolist()
l_82 = l_8.tolist()
   #initialize output dictionary & unique value count
dc   = {}
    #get sample size
b_size = 1000
    #get list of row indexes
idx = [i for i in range(len(m))]
   #loop through the required number of bootstraps
for b in range(5):
    sidx   = np.random.choice(idx,replace=True,size=b_size)
    b_samp = l_11[sidx,:] #this was l_7
    dc['boot_'+str(b)] = {'boot':b_samp}
    oidx   = list(set(idx) - set(sidx))
    o_samp = np.array([])
    if oidx:
        o_samp = l_11[oidx,:]
ooidx = np.array(oidx)  

loc0 = []
loc1 = []
loc2 = []
loc3 = []
bootsamp = list(dc.values())

    
b_samp_0 = list(bootsamp[0].values())
b_samp_00 = b_samp_0[0] #bootstrap number 0
for i in range(len(b_samp_00)):
    loc0.append(l_110.index(b_samp_00[i]))  
loc00 = np.array(loc0) #place matrix of bootstrap number 0 fill in for place in hsmalgorithm when running with b_samp_00

b_samp_1 = list(bootsamp[1].values())
b_samp_11 = b_samp_1[0] #bootstrap number 1
for i in range(len(b_samp_11)):
    loc1.append(l_110.index(b_samp_11[i]))
loc11 = np.array(loc1) #place matrix of bootstrap number 1 fill in for place in hsmalgorithm when running with b_samp_11

b_samp_2 = list(bootsamp[2].values())
b_samp_22 = b_samp_2[0] #bootstrap number 2
for i in range(len(b_samp_22)):
    loc2.append(l_110.index(b_samp_22[i]))
loc22 = np.array(loc2) #place matrix of bootstrap number 2 fill in for place in hsmalgorithm when running with b_samp_22

b_samp_3 = list(bootsamp[3].values())
b_samp_33 = b_samp_3[0] #bootstrap number 3
for i in range(len(b_samp_33)):
    loc3.append(l_110.index(b_samp_33[i]))
loc33 = np.array(loc3) #place matrix of bootstrap number 3 fill in for place in hsmalgorithm when running with b_samp_33
#b_samp number 4 here and sidx as place matrix for this one





oidx0 = list(set(idx) - set(loc00))
oidx00 = np.array(oidx0) #place matrix for test_sample 0
o_samp0 = l_11[oidx0,:] #test_sample number 0

oidx1 = list(set(idx) - set(loc11))
oidx11 = np.array(oidx1) #place matrix for test_sample 1
o_samp1 = l_11[oidx1,:] #test_sample number 1

oidx2 = list(set(idx) - set(loc22))
o_samp2 = l_11[oidx2,:] #test_sample number 2
oidx22 = np.array(oidx2) #place matrix for test_sample 2

oidx3 = list(set(idx) - set(loc33))
oidx33 = np.array(oidx3) #place matrix for test_sample 3
o_samp3 = l_11[oidx3,:]  #test_sample number 3
#o_samp is number 4 and ooidx is test_sample number 4


#FOR HSMALGORITHM FILL IN THE BOOTSAMPLE IN COMBINATION WITH THE RIGHT PLACE MATRIX, SO B_SAMP WITH SIDX, B_SAMP_33 WITH LOC33
#O_SAMP2 WITH OIDX22



def minhash (sample):
    newList = list()
    for j in range(len(sample)):
        def findall(i,pattern=r'[\w]?[\w\w\w\w]?[-]?[\w\w\w\w\w]?\d[\w]?[\d]?[.]?[-]?[\w\w\w]?\d[-]?\w+'):
            return re.findall(pattern,i)
        newList.append([findall(i) for i in sample[j]])
    flat_ls = [item for sublist in newList for item in sublist]
    flat_ls_2 = [item for sublist in flat_ls for item in sublist] #all model words for each product extracted here
    deduplicated_list = list()
    for item in flat_ls_2:
        if item not in deduplicated_list:
            deduplicated_list.append(item)
    res = []
    for j in range(len(sample)):
        for i in range(len(deduplicated_list)):
            res.append([any(deduplicated_list[i] in t for t in sample[j])])

    for k in range(len(res)):
        if res[k] == [True]:
            res[k] = 1
        else:
            res[k] = 0
    res_2 = np.array(res)
    u = np.reshape(res_2, (len(sample),len(deduplicated_list)))
    u_2 = u.T #matrix with the matching model words per product if this is the case it is 1, 0 otherwise

    permut = []
    for k in range(600):
        permut.append(np.random.choice(range(1,(len(deduplicated_list)+1)), size = len(deduplicated_list), replace = False))
    permut_2 = np.ravel(permut)
    permut_new = np.reshape(permut_2, (len(deduplicated_list)*600,1))
    permut_3 = np.reshape(permut_new, (600,len(deduplicated_list)))
    permut_4 = permut_3.T #permutation matrix

    uu = []
    for k in range(600):
        for i in range(1,len(deduplicated_list)+1):
            uu.append(np.where(permut_4[:,k] == i)[0])
    uu_new = np.reshape(uu, (600,len(deduplicated_list)))
    uu_new_2 = uu_new.T
    ii = []
    i = 0
    for k in range(600):
        for j in range(len(sample)):
            for i in range(len(deduplicated_list)):
                if u_2[uu_new_2[i,k],j] == 1:
                    ii.append(i+1)
                    break
    ii_new = np.reshape(ii, (600,len(sample))) #signature matrix

    b=60 # here the bands and rows are determined to decide for the pairs that we fill in for the HSMalgorithm
    r=10
    n, d = ii_new.shape
    assert(n==b*r)
    hashbuckets = collections.defaultdict(set)
    bands = np.array_split(ii_new, b, axis=0)
    for i,band in enumerate(bands):
        for j in range(d):
            band_id = tuple(list(band[:,j])+[str(i)])
            hashbuckets[band_id].add(j)
    c_pairs = set()
    for bucket in hashbuckets.values():
        if len(bucket) > 1:
            for pair in itertools.combinations(bucket, 2):
                c_pairs.add(pair)

    com_pairs = list(c_pairs)[:]
    return com_pairs

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
WORD = re.compile(r"\w+")
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)
        
def jaro_distance(s1, s2) :
 
    # If the strings are equal
    if (s1 == s2) :
        return 1.0;
 
    # Length of two strings
    len1 = len(s1);
    len2 = len(s2);
 
    if (len1 == 0 or len2 == 0) :
        return 0.0;
 
    # Maximum distance upto which matching
    # is allowed
    max_dist = (max(len(s1), len(s2)) // 2 ) - 1;
 
    # Count of matches
    match = 0;
 
    # Hash for matches
    hash_s1 = [0] * len(s1) ;
    hash_s2 = [0] * len(s2) ;
 
    # Traverse through the first string
    for i in range(len1) :
 
        # Check if there is any matches
        for j in range( max(0, i - max_dist),
                    min(len2, i + max_dist + 1)) :
             
            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0) :
                hash_s1[i] = 1;
                hash_s2[j] = 1;
                match += 1;
                break;
         
    # If there is no match
    if (match == 0) :
        return 0.0;
 
    # Number of transpositions
    t = 0;
 
    point = 0;
 
    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1) :
        if (hash_s1[i]) :
 
            # Find the next matched character
            # in second string
            while (hash_s2[point] == 0) :
                point += 1;
 
            if (s1[i] != s2[point]) :
                point += 1;
                t += 1;
            else :
                point += 1;
                 
        t /= 2;
 
    # Return the Jaro Similarity
    return ((match / len1 + match / len2 +
            (match - t) / match ) / 3.0);

def jaro_Winkler(s1, s2) :
 
    jaro_dist = jaro_distance(s1, s2);
 
    # If the jaro Similarity is above a threshold
    if (jaro_dist > 0.7) :
 
        # Find the length of common prefix
        prefix = 0;
 
        for i in range(min(len(s1), len(s2))) :
         
            # If the characters match
            if (s1[i] == s2[i]) :
                prefix += 1;
 
            # Else break
            else :
                break;
 
        # Maximum of 4 characters are allowed in prefix
        prefix = min(4, prefix);
 
        # Calculate jaro winkler Similarity
        jaro_dist += 0.1 * prefix * (1 - jaro_dist);
 
    return jaro_dist;
# totalduplicates = sum(duplicatestotal)

#HSM algorithm
def HSMalgorithm (sample, alpha, beta, gamma, delta, place):
    com_pairs = minhash(sample) #first function as mentioned in readme filled in so that if you run second function first function does not have to run first
    titlepairs_element1 = []
    titlepairs_element2 = []
    position1 = [] # these positions that keep coming back are the original position of the dataset the ith element of the sample is in
    position2 = []

    for i in range(len(com_pairs)):
        pairs_element1 = com_pairs[i][0]
        titlepairs_element1.append(sample[pairs_element1])
        pairs_element2 = com_pairs[i][1]
        titlepairs_element2.append(sample[pairs_element2])
    for i in range(len(com_pairs)):
        position1.append(l_110.index(titlepairs_element1[i]))
        position2.append(l_110.index(titlepairs_element2[i]))
    sameshop = []
    for i in range(len(com_pairs)):
        sameshop.append(z[position1[i]] == z[position2[i]]) #if from the same shop pairs are thrown out as potential duplicates
        indices = locate(sameshop, lambda x: x == False)
        indices2 = list(indices)
        newcom_pairs = []
        for i in range(len(indices2)):
            newcom_pairs.append(com_pairs[indices2[i]])
    brand1 = []
    for i in range(len(m)):
        brand1.append(m[i].split()[0]) # the same here for brands if different brand no longer candidate for duplicate
    for i in range(len(m)):
        if brand1[i] == 'Newegg.com':
            brand1[i] = y[i]['Brand']
    brandnew = list(map(lambda st: str.replace(st, "Insignia\x99", "Insignia"), brand1))   
    brandnew2 = list(map(lambda st: str.replace(st, "Dynex\x99", "Dynex"), brandnew))   
    brandnew3 = list(map(lambda st: str.replace(st, "JVC TV", "JVC"), brandnew2)) 
    titlepairs_element11 = []
    titlepairs_element22= []
    position11 = []
    position22 = []

    for i in range(len(newcom_pairs)):
        pairs_element11 = newcom_pairs[i][0]
        titlepairs_element11.append(sample[pairs_element11])
        pairs_element22 = newcom_pairs[i][1]
        titlepairs_element22.append(sample[pairs_element22])

    for i in range(len(newcom_pairs)):
        position11.append(l_110.index(titlepairs_element11[i]))
        position22.append(l_110.index(titlepairs_element22[i]))
        difbrands = []
    for i in range(len(newcom_pairs)):
        difbrands.append(brandnew3[position11[i]] == brandnew3[position22[i]])
    indices3 = locate(difbrands, lambda x: x == True)
    indices4 = list(indices3)
    newcom_pairs2 = []
    for i in range(len(indices4)):
        newcom_pairs2.append(newcom_pairs[indices4[i]])

    newcom_pairs3 = []
    for j in range(len(newcom_pairs2)):
        for i in range(len(newcom_pairs2)):
            if newcom_pairs2[i][0] == newcom_pairs2[j][1]:
                if newcom_pairs2[i][1] == newcom_pairs2[j][0]:
                    newcom_pairs3.append(1)
                else:
                    newcom_pairs3.append(0)
            else:
                newcom_pairs3.append(0)
    newcom_pairs4 = list(locate(newcom_pairs3, lambda x: x == 1))
    newcom_pairs5 = []
    newcom_pairs6 = []
    for i in range(len(newcom_pairs4)):
        newcom_pairs5.append(math.floor(newcom_pairs4[i]/len(newcom_pairs2)))
        newcom_pairs6.append(newcom_pairs4[i] - len(newcom_pairs2)*newcom_pairs5[i])
    newcom_pairs7 = []
    for i in range(len(newcom_pairs4)):  
        newcom_pairs7.append(max(newcom_pairs5[i],newcom_pairs6[i]))
    kkk = [*set(newcom_pairs7)]
    kkk.sort(reverse=True)
    for i in range(len(kkk)):
        del newcom_pairs2[kkk[i]]
    
    titlepairs_element100 = []
    titlepairs_element200 = []
    position100 = []
    position200 = []

    for i in range(len(newcom_pairs2)):
        pairs_element100 = newcom_pairs2[i][0]
        titlepairs_element100.append(sample[pairs_element100])
        pairs_element200 = newcom_pairs2[i][1]
        titlepairs_element200.append(sample[pairs_element200])
    for i in range(len(newcom_pairs2)):
        position100.append(l_110.index(titlepairs_element100[i]))
        position200.append(l_110.index(titlepairs_element200[i]))


    cosine = []
    m_10 = [sub.replace('/', '') for sub in m]
    m_11 = [sub.replace('-', '') for sub in m_10]


    for i in range(len(position100)): # we start here with the TMWM method
        text1 = m_11[position100[i]]
        text2 = m_11[position200[i]]
    
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)
    
        cosine.append(get_cosine(vector1, vector2))
    cosine_2 = [n > alpha for n in cosine] # this is the alpha that determines whether the titles match
    cosine_3 =locate(cosine_2, lambda x: x == True)
    cosine_4 = list(cosine_3)
    titelword1 = list()
    for j in range(len(position100)):
        def findall(i,pattern=r'[\w]?[\w\w\w]?[-]?[\w\w\w\w]?\d[.]?[-]?\d[-]?\w+'):
            return re.findall(pattern,i)
        titelword1.append([findall(i) for i in l_110[position100[j]]])
    titelword2 = list()
    for j in range(len(position100)):
        def findall(i,pattern=r'[\w]?[\w\w\w]?[-]?[\w\w\w\w]?\d[.]?[-]?\d[-]?\w+'):
            return re.findall(pattern,i)
        titelword2.append([findall(i) for i in l_110[position200[j]]])

    titelword11 = []
    titelword22 = []
    for i in range(len(titelword1)):
        titelword11.append([*set(titelword1[i][0])])
        titelword22.append([*set(titelword2[i][0])])
    

    lev_dist = []
    zeroes = []
    avglev_dist = []

    for k in range(len(titelword11)):
        for p in range(len(titelword22[k])):
            for j in range(len(titelword11[k])):
                lev_dist.append(lev(titelword11[k][j], titelword22[k][p]))
        zeroes.append([n < 1 for n in lev_dist])
        avglev_dist.append(sum(zeroes[k])/min(len(titelword22[k]), len(titelword11[k])))
        lev_dist.clear()
 
    avg_levdist_we_use =locate(avglev_dist, lambda x: x >= 1)
    avglvsim = list(avg_levdist_we_use)

    finalsim = []
    for i in range(len(cosine)):
        finalsim.append(beta*cosine[i] + (1-beta)*avglev_dist[i])
    pairsfortitle =locate(finalsim, lambda x: x >= 0.70)
    pairsfortitle2 = list(pairsfortitle) #these are the pairs that are coming out of TMWM as duplicates

    values = [] # until next comment here we calculate the keys that match these are listed in ccc
    values2 = []
    avgsim = []
    theta = []
    for r in range(len(position100)):
        values = []
        values2 = []
        for key in y[position100[r]].keys() :
            values.append(y[position100[r]][key])
        keys = list(y[position100[r]].keys())
        for key in y[position200[r]].keys() :
            values2.append(y[position200[r]][key])
        keys2 = list(y[position200[r]].keys())
        keys_sim = []
        for j in range(len(keys)):
            for i in range(len(keys2)):
                keys_sim.append(jaro_Winkler(keys[j],keys2[i]))
        c = [n > gamma for n in keys_sim]
        cc =locate(c, lambda x: x == True)
        ccc = list(cc)
        theta.append(min(len(ccc)/min(len(keys),len(keys2)),1))
    
        valueinkeys = [] # here we compare the values of the matched keys after the matched keys ccc are found
        valueinkeys_2 = []
        for i in range(len(ccc)):
            valueinkeys.append(math.floor(ccc[i]/len(keys2)))
            valueinkeys_2.append(ccc[i] - len(keys2)*valueinkeys[i])
        similarity = []
        for i in range(len(valueinkeys)):
            similarity.append(jaro_Winkler(values[valueinkeys[i]],values2[valueinkeys_2[i]]))
        if len(valueinkeys) == 0:
            avgsim.append(0)
        else:
            avgsim.append(sum(similarity)/len(valueinkeys)) #this is average sim value

    mwperc = [] #part until next comment here calculates the matching percentage in the attribute words that did not have a key match
    for r in range(len(newcom_pairs2)):
        values2 = []
        values = []
        for key in y[position100[r]].keys() :
            values.append(y[position100[r]][key])
        keys = list(y[position100[r]].keys())
        for key in y[position200[r]].keys() :
            values2.append(y[position200[r]][key])
        keys2 = list(y[position200[r]].keys())
        keysfind = []
        goodkeys = []
        valueinkeys3 = []
        valueinkeys4 = []
        for i in range(len(keys)):
            for j in range(len(keys2)):
                keysfind.append(jaro_Winkler(keys[i],keys2[j]))
        goodkeys = [n > 0.8 for n in keysfind]
        goodkeys2 = list(locate(goodkeys, lambda x: x == True))
        for i in range(len(goodkeys2)):
            valueinkeys3.append(math.floor(goodkeys2[i]/len(keys2)))
            valueinkeys4.append(goodkeys2[i] - len(keys2)*valueinkeys3[i])

        valueinkeys3 = [*set(valueinkeys3)]
        valueinkeys4 = [*set(valueinkeys4)]
        newkeys4 = sorted(valueinkeys4)
        newkeys3 = sorted(valueinkeys3)

        for i in range(len(newkeys3)):
            del keys[newkeys3[-1-i]]
        for i in range(len(newkeys4)):
            del keys2[newkeys4[-1-i]]
        
        values5 = []
        for i in range(len(keys)):
            values5.append(y[position100[r]][keys[i]])
        values6 = []
        for i in range(len(keys2)):
            values6.append(y[position200[r]][keys2[i]])
   
        values5 = [*set(values5)]
        values6 = [*set(values6)]
        sim = []
        for i in range(len(values5)):
            for j in range(len(values6)):
                sim.append(jaro_Winkler(values5[i],values6[j]))
        perc = [n > 0.8 for n in sim]
        percwords = list(locate(perc, lambda x: x == True))
        if min(len(values5),len(values6)) == 0:
            mwperc.append(0)
        else:
            mwperc.append(len(percwords)/min(len(values5),len(values6)))
        
    bsim = []
    for i in range(len(newcom_pairs2)):        
        bsim.append(theta[i]*avgsim[i] + (1-theta[i])*mwperc[i])
        bsim2 = [n > delta for n in bsim]
        bsim3 = list(locate(bsim2, lambda x: x == True)) #this is the hsim in the paper


    duplicatestotal = [] #this part calculates all the duplicates in the sample were running the algorithm over
    for j in range(len(m)):
        for i in range(len(m)):
            if i == j:
                duplicatestotal.append(0)
            else:
                if o[i] == o[j]:
                    duplicatestotal.append(1)
                else:
                    duplicatestotal.append(0)
    duplicatestotal3 = list(locate(duplicatestotal, lambda x: x == 1))
    pair_1 = []
    pair2 = []
    for i in range(len(duplicatestotal3)):
        pair_1.append(math.floor(duplicatestotal3[i]/len(m)))
    for i in range(len(duplicatestotal3)):
        pair2.append(duplicatestotal3[i] - len(m)*pair_1[i])
    
    pair_3 = [item for sublist in zip(pair_1, pair2) for item in sublist]
    pair_4 = []
    for i in range(2,len(pair_3)+2,2):
        pair_4.append(pair_3[i-2:i])
    
    pair_5 = []

    for i in range(len(pair_4)):
        if pair_4[i][1] >= pair_4[i][0]:
            pair_5.append(0)
        else:
            pair_5.append(1)
    pair_6 = list(locate(pair_5, lambda x: x == 1))  
    pair_7 = []
    for i in range(len(pair_6)):
        pair_7.append(pair_4[pair_6[i]])
    
    pair_8 = []
    pair_9 = []
    for i in range(len(pair_7)):
        if sum(pair_7[i][0] == place)>=1:
            pair_8.append(1)
        else:
            pair_8.append(0)
    for i in range(len(pair_7)):
        if sum(pair_7[i][1] == place)>=1:
            pair_9.append(1)
        else:
            pair_9.append(0)
    pair_10 = [item for sublist in zip(pair_8, pair_9) for item in sublist]
    pair_11 = []
    for i in range(2,len(pair_10)+2,2):
        pair_11.append(pair_10[i-2:i])
    bsampdups = list(locate(pair_11, lambda x: x == [1,1]))
    bsampduplicates = []
    for i in range(len(bsampdups)):
        bsampduplicates.append(pair_7[bsampdups[i]]) #these are all the duplicates in the sample we run
    

    ultimatepairs = [] #TMWM pairs and KVP matches are combined here
    for i in range(len(pairsfortitle2)):
        ultimatepairs.append(newcom_pairs2[pairsfortitle2[i]])
    for i in range(len(bsim3)):
        ultimatepairs.append(newcom_pairs2[bsim3[i]])
    ultimatepairs = [*set(ultimatepairs)]


    pairs_algo = [] #until white line this part sets highest number as first element in ultimatepairs just like we do for the duplicates in the whole sample
    for i in range(len(ultimatepairs)):
        for j in range(2):
            pairs_algo.append(ultimatepairs[i][j])
    pairs_algo2 = []
    for i in range(2,len(pairs_algo)+2,2):
        pairs_algo2.append(pairs_algo[i-2:i])
    for i in range(len(pairs_algo2)):
        pairs_algo2[i].sort(reverse=True)
    sidx1 = []
    sidx2 = []
    for i in range(len(pairs_algo2)):
        sidx1.append(place[pairs_algo2[i][0]])
    for i in range(len(pairs_algo2)):
        sidx2.append(place[pairs_algo2[i][1]])
    sidx3 = [item for sublist in zip(sidx1, sidx2) for item in sublist]
    sidx4 = []
    for i in range(2,len(sidx3)+2,2):
        sidx4.append(sidx3[i-2:i])   
    for i in range(len(sidx4)):
        sidx4[i].sort(reverse=True) # these are the pairs the algorithm thinks are duplicates
        
    precision = [] #the number of matches in our algorithm compared to all the duplicates in the sample
    for i in range(len(sidx4)):
        for j in range(len(bsampduplicates)):
            if sidx4[i] == bsampduplicates[j]:
                precision.append(1)
            else:
                precision.append(0) 
    print('pair quality:', sum(precision)/len(sidx4))
    print('pair completeness', sum(precision)/len(bsampduplicates))
    return ((sum(precision)/len(sidx4))*(sum(precision)/len(bsampduplicates))*2)/((sum(precision)/len(bsampduplicates)) + (sum(precision)/len(sidx4)))
   



def GridsearchHSM (sample, place): #fill in b_samp and it will calculate the optimal combination with highest f1 score for b_samp then only
    beta = [0.85, 0.9] 
    gamma = [0.8, 0.85, 0.9] 
    delta = [0.4, 0.45, 0.5] 
 
    alloptions = []
    options = list(itertools.product(beta, gamma, delta))
    for i in range(len(options)):
        alloptions.append(HSMalgorithm(sample, 0.9, options[i][0], options[i][1], options[i][2], place))
    choices = list(locate(alloptions, lambda x: x == max(alloptions)))[0]
    choices2 = options[choices]
    return(choices2)

#FILL IN O_SAMP TO GET THE SCORE FOR ALL 5

def f1scorefortest (sample):  #fill in o_samp and the hsm will calculate for the 5 test_samples
    test_samp = []
    test_samp.append(sample)
    test_samp.append(o_samp0)
    test_samp.append(o_samp1)
    test_samp.append(o_samp2)
    test_samp.append(o_samp3)
    test_place = []
    test_place.append(ooidx)
    test_place.append(oidx00)
    test_place.append(oidx11)
    test_place.append(oidx22)
    test_place.append(oidx33)
    score = []
    for i in range(len(test_samp)):
        score.append(HSMalgorithm(test_samp[i], 0.9, 0.89, 0.83, 0.4, test_place[i]))
    return sum(score)/5

