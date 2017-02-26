'''
Copied from Sub1 and Modified for Next Level  
''' 

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
import sklearn.feature_extraction.text as sktf
from difflib import SequenceMatcher as seq_matcher
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from ngram import NGram

from nltk.stem.porter import *
stemmer = PorterStemmer()

print ("Starting Data Loading... ") 
df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('../input/attributes.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../input/product_descriptions.csv', encoding="ISO-8859-1")

print ("Starting Feature Building ... ") 
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
print('df_brand \n', df_brand.head())

df_col = df_attr[df_attr.name.str.contains("Color", na=False)][["product_uid", "value"]].rename(columns={"value": "color"}) 
df_color = df_col.groupby('product_uid', as_index=False).agg(lambda x: ' '.join(x))
print('df_color \n', df_color.head())

df_mat = df_attr[df_attr.name.str.contains("Material", na=False)][["product_uid", "value"]].rename(columns={"value": "material"}) 
df_material = df_mat.groupby('product_uid', as_index=False).agg(lambda x: ' '.join(x))
print('df_material \n', df_material.head())

num_train = df_train.shape[0]

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def ngram_similarity(data,col1,col2):
    cos=[]
    for i in range(len(data.id)):        
        st=data[col1][i]
        title=data[col2][i]   
        n = NGram(title.split(), key=lambda x:x[1])
        for s in st.split():
            n.search(s) 
                
        tfidf = sktf.TfidfVectorizer().fit_transform([st,title])
        c=((tfidf * tfidf.T).A)[0,1]     
        cos.append(c)                                   
    return cos 
    

def dist_cosine(data,col1,col2):   
    cos=[]
    for i in range(len(data.id)):        
        st=data[col1][i]
        title=data[col2][i]        
        tfidf = sktf.TfidfVectorizer().fit_transform([st,title])
        c=((tfidf * tfidf.T).A)[0,1]     
        cos.append(c)                                   
    return cos  

def mean_dist(data,col1,col2):
    mean_edit_s_t=[]
    for i in range(len(data)):
        search=data[col1][i]
        title=data[col2][i]
        max_edit_s_t_arr=[]       
        for s in search.split(): 
                max_edit_s_t=[]                
                for t in title.split():                   
                    a=seq_matcher(None,s,t).ratio()
                    max_edit_s_t.append(a)
                max_edit_s_t_arr.append(max_edit_s_t)         
        l=0   
        for item in max_edit_s_t_arr:
                l=l+max(item)     
        mean_edit_s_t.append(l/len(max_edit_s_t_arr))            
    return mean_edit_s_t 

def str_stem(s): 
    if isinstance(s, basestring):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"
    
def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
df_all = pd.merge(df_all, df_material, how='left', on='product_uid')

print('df_all \n', df_all.head())

df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
df_all['color'] = df_all['color'].astype(str).map(lambda x:str_stem(x))
df_all['material'] = df_all['material'].astype(str).map(lambda x:str_stem(x))

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

print('df_all \n', df_all.head())

df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_color'] = df_all['color'].astype(str).map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_material'] = df_all['material'].astype(str).map(lambda x:len(x.split())).astype(np.int64)

print('df_all \n', df_all.head())

df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))

df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))

df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))

df_all["cosine_s.brand"]=dist_cosine(df_all,"search_term","brand")
df_all["cosine_s.material"]=dist_cosine(df_all,"search_term","material")
df_all["cosine_s.title"]=dist_cosine(df_all,"search_term","product_title")
df_all["mean_s.brand"]=mean_dist(df_all,"search_term","brand")
df_all["mean_s.material"]=mean_dist(df_all,"search_term","material")
df_all["mean_s.title"]=mean_dist(df_all,"search_term","product_title")

print('df_all \n', df_all.head())

df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']
df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']+"\t"+df_all['color'].astype(str)+"\t"+df_all['material'].astype(str)
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_color'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
df_all['word_in_material'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))

df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
df_all['ratio_color'] = df_all['word_in_color']/df_all['len_of_color']
df_all['ratio_material'] = df_all['word_in_material']/df_all['len_of_material']

print('df_all \n', df_all.head())

df_brand = pd.unique(df_all.brand.ravel())
d={}
i = 1000
for s in df_brand:
    d[s]=i
    i+=3

df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))

print('df_all \n', df_all.head())
#df_all.to_excel('../intrim/df_all-1.xls')

df_all = df_all.drop(['search_term','product_title','product_description','product_info','brand', 'color', 'material', 'attr'],axis=1)

print('df_all \n', df_all.head())
df_all.to_csv('../intrim/df_all-2.csv')

print ("Starting Model Training... ") 
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

#print('X_train \n', X_train.head())
#print('y_train \n', y_train.head())

########################## RandomForestRegressor #################################

clf = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
yo_pred = clf.predict(X_train)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_RFR.csv',index=False)

RSME  = fmean_squared_error(y_train, yo_pred)
print ("RandomForestRegressor - RSME = ", RSME) 

########################## SVR #################################

clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
yo_pred = clf.predict(X_train)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_SVR.csv',index=False)

RSME  = fmean_squared_error(y_train, yo_pred)
print ("SVR - RSME = ", RSME) 

########################## LinearRegression #################################

clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
yo_pred = clf.predict(X_train)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_GLR.csv',index=False)

RSME  = fmean_squared_error(y_train, yo_pred)
print ("LinearRegression - RSME = ", RSME) 

########################## GradientBoostingRegressor #################################

clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
yo_pred = clf.predict(X_train)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission_V9_GBR.csv',index=False)

RSME  = fmean_squared_error(y_train, yo_pred)
print ("GradientBoostingRegressor - RSME = ", RSME) 
