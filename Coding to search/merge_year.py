import pandas as pd  
import matplotlib.pyplot as plt
import matplotlib_venn

def make_a_plot(dict_multi):
    counter_var = 0
    minx = 2000000000000
    maxx = 0
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        minx = min(min(X_axis), minx)
        maxx = max(max(X_axis), maxx)
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        X_axis = [X_axis[i] + 1 / (len(dict_multi) + 1) * (counter_var - (len(dict_multi) - 1) / 2) for i in range(len(X_axis))]
        plt.bar(X_axis, list(dict_multi[dict_one_name].values()), 1 / (len(dict_multi) + 1), label = dict_one_name)
        counter_var += 1 
    plt.legend()
    plt.xticks(range(minx, maxx + 1), range(minx, maxx + 1))
    plt.gca().set_xticklabels(range(minx, maxx + 1), rotation = (45), va = 'top', ha = 'right') 
    plt.show()
    plt.close()
 
open_ieee = open("export2023.csv", "r", encoding = "utf-8")
lines_open_ieee = open_ieee.readlines()
write_ieee = ""
for i in lines_open_ieee:
    replaced = i.replace('2018 IEEE International Conference Quality Management, Transport and Information Security," Information Technologies"" (IT&QM&IS)"""', '"2018 IEEE International Conference Quality Management, Transport and Information Security, Information Technologies (IT&QM&IS)"')
    replaced = replaced.replace('2013 IEEE 14th International Symposium on A World of Wireless," Mobile and Multimedia Networks"" (WoWMoM)"""', '"2013 IEEE 14th International Symposium on A World of Wireless, Mobile and Multimedia Networks (WoWMoM)"') 
    replaced = replaced.replace('"Microwave & Telecommunication Technology"', 'Microwave & Telecommunication Technology')
    replaced = replaced.replace('International Radar Conference "Surveillance for a Safer World" (RADAR 2009)', 'International Radar Conference Surveillance for a Safer World (RADAR 2009)')
    write_ieee += replaced

open_ieee.close()
limy = 1
open_ieee2 = open("export2023_2.csv", "w", encoding = "utf-8")
open_ieee2.write(write_ieee)
open_ieee2.close()

df_ieee = pd.read_csv("export2023_2.csv")
df_scopus = pd.read_csv("scopus.csv")
df_scopus2 = pd.read_csv("../SCOPUS_LATEST/scopus.csv") 
df_sd = pd.read_csv("ScienceDirect.csv")
 
data_top = df_sd.head() 

print(df_ieee.columns)
print(df_scopus.columns)
print(df_sd.columns)

print(len(df_ieee["Document Title"]))
print(len(df_scopus["Title"]))
print(len(df_sd["title"]))

name_doi = dict()
doi_name = dict()
name_index = dict()
doi_index = dict()

for i in range(len(df_ieee["Document Title"])):
    tmp_year = int(df_ieee["Publication Year"][i])
    if tmp_year < limy:
        continue 
    tmp_name = str(df_ieee["Document Title"][i]).lower()
    tmp_doi = str(df_ieee["DOI"][i]).lower()
    if tmp_doi not in doi_index:
        doi_index[tmp_doi] = set()
    if tmp_name not in name_index:
        name_index[tmp_name] = set()
    if tmp_doi not in doi_name:
        doi_name[tmp_doi] = set()
    if tmp_name not in name_doi:
        name_doi[tmp_name] = set()
    doi_index[tmp_doi].add((i, "IEEE"))
    name_index[tmp_name].add((i, "IEEE"))
    doi_name[tmp_doi].add(tmp_name)
    name_doi[tmp_name].add(tmp_doi) 
print(len(name_doi), len(doi_name))

seen_scopus = set()
for i in range(len(df_scopus["Title"])):
    tmp_year = int(df_scopus["Year"][i])
    if tmp_year < limy:
        continue 
    tmp_name = str(df_scopus["Title"][i]).lower()
    if tmp_name in seen_scopus:
        continue
    seen_scopus.add(tmp_name)
    tmp_doi = str(df_scopus["DOI"][i]).lower()
    if tmp_doi not in doi_index:
        doi_index[tmp_doi] = set()
    if tmp_name not in name_index:
        name_index[tmp_name] = set()
    if tmp_doi not in doi_name:
        doi_name[tmp_doi] = set()
    if tmp_name not in name_doi:
        name_doi[tmp_name] = set()
    doi_index[tmp_doi].add((i, "scopus"))
    name_index[tmp_name].add((i, "scopus"))
    doi_name[tmp_doi].add(tmp_name)
    name_doi[tmp_name].add(tmp_doi) 
print(len(name_doi), len(doi_name))
for i in range(len(df_scopus2["Title"])):
    tmp_year = int(df_scopus2["Year"][i])
    if tmp_year < limy:
        continue 
    tmp_name = str(df_scopus2["Title"][i]).lower()
    if tmp_name in seen_scopus:
        continue
    else:
        print("new in scopus")
    seen_scopus.add(tmp_name)
    tmp_doi = str(df_scopus2["DOI"][i]).lower()
    if tmp_doi not in doi_index:
        doi_index[tmp_doi] = set()
    if tmp_name not in name_index:
        name_index[tmp_name] = set()
    if tmp_doi not in doi_name:
        doi_name[tmp_doi] = set()
    if tmp_name not in name_doi:
        name_doi[tmp_name] = set()
    doi_index[tmp_doi].add((i, "scopus"))
    name_index[tmp_name].add((i, "scopus"))
    doi_name[tmp_doi].add(tmp_name)
    name_doi[tmp_name].add(tmp_doi) 
print(len(name_doi), len(doi_name))

for i in range(len(df_sd["title"])):
    if int(df_sd["year"][i]) < limy:
        continue
    tmp_name = str(df_sd["title"][i]).lower()
    tmp_doi = str(df_sd["doi"][i]).replace("https://doi.org/", "").lower()
    if tmp_doi not in doi_index:
        doi_index[tmp_doi] = set()
    if tmp_name not in name_index:
        name_index[tmp_name] = set()
    if tmp_doi not in doi_name:
        doi_name[tmp_doi] = set()
    if tmp_name not in name_doi:
        name_doi[tmp_name] = set()
    doi_index[tmp_doi].add((i, "SD"))
    name_index[tmp_name].add((i, "SD"))
    doi_name[tmp_doi].add(tmp_name)
    name_doi[tmp_name].add(tmp_doi)  
print(len(name_doi), len(doi_name))

yes_words_lists = [
    ["maritime", "analysis", "analyze", "analyzing", "anomaly", "anomalies", "anomalous", "detect", "detection", "detecting"]
    ]

yes_words_lists = [
    ["RNN", "recurrent neural network"], 
    ["CNN", "convolutional neural network"], 
    ["HMM", "hidden markov model"], 
    ["behavior", "behaviour"], 
    ["watercraft"],
    ["smuggling", "smuggle"],
    ["piracy", "pirate"],
    ["trafficking", "traffic"],
    ["illegal trafficking", "illegal traffick"],
    ["illegal zone", "illegal entry", "illegal zone entry"],
    ["GPS", "global positioning system"],
    ["GNSS", "global navigation satellite system"],
    ["SAR", "synthetic-aperture radar", "synthetic aperture radar"],
    ["AIS", "automatic identification system"],
    ["trajectory", "trajectories"],
    ["position", "location", "zone", "longitude", "latitude"],
    ["speed"],
    ["SVM", "support vector machine"], 
    ["SVR", "support vector regression"],
    ["GMM", "gaussian mixture model", "gauss", "gaussian"], 
    ["KDE", "kernel density estimation"], 
    ["K-means"], 
    ["DBSCAN"], 
    ["NN", "neural network"],
    ["ANN", "artificial neural network"],
    ["ML", "machine learning"],
    ["UT", "unusual turn", "U turn", "U-turn"],
    ["AT", "anomalous trajectories"],
    ["graph"],
    ["OOS", "on-off switching", "on off switching"],
    ["VRNN", "variational recurrent neural network", "variational", "recurrent neural network"],
    ["autoencoder", "encode", "decode", "encoder", "decoder"],
    ["DL", "deep learning"],
    ["AI ", "artificial intelligence", "AI,", "AI.", "AI)"],
    ["BN", "bayesian network", "bayes", "bayesian"],
    ["LSTM", "long short-term memory"],
    ["maritime", "marine vessell", "ship"],
    ["anomaly", "anomalies", "anomalous"],
    ["analysis", "analyze", "analyzing"],
    ["detect", "detection", "detecting"],
    [" supervised", ",supervised", ". Supervised", "(supervised"],
    ["unsupervised"], 
    ["semi-supervised", "semisupervised"],
    ] 

any_words_list = [
    ["maritime", "marine vessell", "ship"],
    ["anomaly", "anomalies", "anomalous"],
    ["vessell", "vehicle", "traffic", "trajector", "path", "speed", "position", "location", "zone", "longitude", "latitude", "global positioning system", "GPS", "automatic identification system", "AIS"],
    #["analysis", "analyze", "analyzing", "detect", "detection", "detecting"]
]

no_words = ["naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"] 
no_words = ["vision", "radar", "image", "internet", "IoT", "visual", "brain", "medicine", "disease", "biolog", "city", "chemistry", "wi-fi", "covid"]

aggregate_cols = dict()
for paper in name_doi:
    for indexes in name_index[paper]:
        if paper not in aggregate_cols:
            aggregate_cols[paper] = ""
        if indexes[1] == "IEEE":
            for colname in df_ieee.columns:
                aggregate_cols[paper] += '"' + str(df_ieee[colname][indexes[0]]) + '",'
        if indexes[1] == "scopus":
            for colname in df_scopus.columns:
                aggregate_cols[paper] += '"' + str(df_scopus[colname][indexes[0]]) + '",'
        if indexes[1] == "SD":
            for colname in df_sd.columns:
                aggregate_cols[paper] += '"' + str(df_sd[colname][indexes[0]]) + '",'

all_my = dict()
by_keyword = dict()

for yes_words in yes_words_lists:
    new_range = 0
    match_array = dict()
    for x in yes_words:
        match_array[x] = set()
    for paper in aggregate_cols:  
        banned = False
        for any_of in any_words_list:
            found_any = False
            for x in any_of:
                if aggregate_cols[paper].lower().count(x.lower()):
                    found_any = True
                    break
            if not found_any:
                banned = True
                break
        for x in no_words:
            if aggregate_cols[paper].lower().count(x.lower()):
                banned = True
                break
        if banned:
            continue
        for x in yes_words:
            if aggregate_cols[paper].lower().count(x.lower()):
                match_array[x].add(paper)
                if paper not in all_my:
                    all_my[paper] = set()
                all_my[paper].add(x)
    all_words = match_array[yes_words[0]]
    any_words = set()
    for x in yes_words:
        all_words = all_words.intersection(match_array[x])
        any_words = any_words.union(match_array[x])
        print(x, len(match_array[x]))
    if len(yes_words) > 1:
        print(yes_words, len(all_words), (len(any_words)))
    year_dict = dict()
    for paper in any_words: 
        #print(paper)
        min_pub_year = 1000000000
        for indexes in name_index[paper]:
            if indexes[1] == "IEEE":
                if not str(df_ieee["Publication Year"][indexes[0]]).isdigit():
                    print("Error", df_ieee["Publication Year"][indexes[0]], indexes[0])
                else: 
                    min_pub_year = min(min_pub_year, int(df_ieee["Publication Year"][indexes[0]]))
            if indexes[1] == "scopus":
                min_pub_year = min(min_pub_year, int(df_scopus["Year"][indexes[0]]))
            if indexes[1] == "SD":
                min_pub_year = min(min_pub_year, int(df_sd["year"][indexes[0]]))
        if min_pub_year not in year_dict:
                year_dict[min_pub_year] = 0
        year_dict[min_pub_year] += 1
    if sum(list(year_dict.values())) > 0:
        by_keyword[yes_words[0]] = year_dict 
    #for year in year_dict:
        #print(year, year_dict[year])
    #break
    '''
    '''
new_csv = "title,keywords\n"
print(len(all_my))
for x in all_my:
    print(x, all_my[x])
    new_csv += '"' + x + '","'
    first = True
    for k in all_my[x]:
        if k == "":
            continue
        if not first:
            new_csv += ', '
        new_csv += k
        first = False
    new_csv += '"\n'
#print(new_csv)

open_2 = open("mycsv.csv", "w", encoding = "utf-8")
open_2.write(new_csv)
open_2.close()

by_keyword2 = dict()
for x in by_keyword:
    print(x, sum(list(by_keyword[x].values())))
    if sum(list(by_keyword[x].values())) >= 10:
        by_keyword2[x] = by_keyword[x]
#print(by_keyword2)

dict1 = {2017: 4, 2018: 19, 2020: 24, 2021: 25}
dict2 = {2010: 34, 2019: 22, 2020: 23, 2023: 23}
dict3 = {2010: 25, 2019: 2, 2020: 27, 2021: 34}
#make_a_plot({"my1": dict1, "my2": dict2, "my3": dict3})
#print(by_keyword)
#make_a_plot(by_keyword)
#make_a_plot(by_keyword2)

category_keyword = { 
    "RNN": "ML",
    "CNN": "ML", 
    "HMM": "ML", 
    "behavior": "B", 
    "watercraft": "KW",
    "smuggling": "B",
    "piracy": "B",
    "trafficking": "B",
    "illegal trafficking": "B",
    "illegal zone": "B",
    "GPS": "D",
    "GNSS": "D",
    "SAR": "D",
    "AIS": "D",
    "trajectory": "D", 
    "position": "D",
    "speed": "D",
    "SVM": "ML",  
    "SVR": "ML", 
    "GMM": "ML", 
    "KDE": "ML", 
    "K-means": "ML", 
    "DBSCAN": "ML", 
    "NN": "ML", 
    "ANN": "ML",
    "ML": "ML", 
    "UT": "B", 
    "AT": "B", 
    "graph": "ML", 
    "OOS": "B", 
    "VRNN": "ML",   
    "autoencoder": "ML",  
    "DL": "ML", 
    "AI ": "ML",   
    "BN": "ML",  
    "LSTM": "ML", 
    "maritime": "KW",
    "anomaly": "KW",  
    "analysis": "KW",  
    "detect": "KW",  
    "": "KW",
    " supervised": "ML", 
    "unsupervised": "ML", 
    "semi-supervised": "ML", 
}

by_keyword_category = dict()
by_keyword_category2 = dict()

for cat in by_keyword: 
    if category_keyword[cat] not in by_keyword_category:
        by_keyword_category[category_keyword[cat]] = dict()
    by_keyword_category[category_keyword[cat]][cat] = by_keyword[cat]
    if sum(list(by_keyword[cat].values())) >= 10:
        if category_keyword[cat] not in by_keyword_category2:
            by_keyword_category2[category_keyword[cat]] = dict()
        by_keyword_category2[category_keyword[cat]][cat] = by_keyword[cat]
#print(by_keyword_category)
#print(by_keyword_category2)

#for cat in by_keyword_category: 
    #make_a_plot(by_keyword_category[cat])
    
#for cat in by_keyword_category2: 
    #make_a_plot(by_keyword_category2[cat])
    
keyword_names = dict()
for x in all_my:
    for kw in all_my[x]:
        if kw not in keyword_names:
            keyword_names[kw] = set()
        keyword_names[kw].add(x)
        
keywords_category_names = dict()
for yes_words in yes_words_lists:
    first_word = yes_words[0]
    keywords_category_names[first_word] = set()
    for kw in yes_words:
        if kw in keyword_names:
            for paper in keyword_names[kw]:
                keywords_category_names[first_word].add(paper)

keywords_category_names2 = dict()
for yes_words in yes_words_lists:
    first_word = yes_words[0]
    my_category = category_keyword[first_word]
    if my_category not in keywords_category_names2:
        keywords_category_names2[my_category] = set()
    for kw in yes_words: 
        if kw in keyword_names:
            for paper in keyword_names[kw]:
                keywords_category_names2[my_category].add(paper)
     
matplotlib_venn.venn3([keywords_category_names2['B'], keywords_category_names2['KW'], keywords_category_names2['D']], set_labels = ('behaviour', 'keywords', "data"))
plt.show()
#for w1 in by_keyword:
    #for w2 in by_keyword:
        #for w3 in by_keyword:
