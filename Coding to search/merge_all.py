import pandas as pd 

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
    ["recurrent neural network", "RNN"], 
    ["convolutional neural network", "CNN"], 
    ["hidden markov model", "HMM"], 
    ["behavior", "behaviour"], 
    ["watercraft"],
    ["smuggling", "smuggle"],
    ["piracy", "pirate"],
    ["trafficking", "traffic"],
    ["illegal", "trafficking", "traffic"],
    ["illegal", "entry", "zone", "enter"],
    ["global positioning system", "GPS"],
    ["global navigation satellite system", "GNSS"],
    ["synthetic-aperture radar", "synthetic aperture radar", "SAR"],
    ["automatic identification system", "AIS"],
    ["trajectory", "trajectories"],
    ["position"],
    ["speed"],
    ["support vector machine", "SVM"], 
    ["support vector regression", "SVR"],
    ["gaussian mixture model", "GMM", "gauss", "gaussian"], 
    ["kernel density estimation", "KDE"], 
    ["K-means"], 
    ["DBSCAN"], 
    ["neural network", "NN"],
    ["artificial neural network", "ANN"],
    ["machine learning"],
    ["deep learning"],
    ["artificial intelligence", "AI"],
    ["bayesian network", "BN", "bayes", "bayesian"],
    ["LSTM", "long short-term memory"],
    ["maritime", "marine", "sea", "ship"],
    ["anomaly", "anomalies", "anomalous"],
    ["analysis", "analyze", "analyzing", "detect", "detection", "detecting"],
    [""]
    ]

yes_words_lists = [
    ["support vector machine", "SVM"], 
]

any_words_list = [
    ["maritime", "marine", "sea", "ship", "vessell", "vehicle", "traffic", "trajector", "path", "speed", "position"],
    ["anomaly", "anomalies", "anomalous"],
    ["analysis", "analyze", "analyzing", "detect", "detection", "detecting"]
]

no_words = ["vision", "radar", "image", "internet", "IoT", "visual", "brain", "medicine", "disease", "biolog"]
no_words = ["naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"] 

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
        for indexes in name_index[paper]:
            if indexes[1] == "IEEE":
                if not str(df_ieee["Publication Year"][indexes[0]]).isdigit():
                    print("Error", df_ieee["Publication Year"][indexes[0]], indexes[0])
                else:
                    if int(df_ieee["Publication Year"][indexes[0]]) not in year_dict:
                        year_dict[int(df_ieee["Publication Year"][indexes[0]])] = 0
                    year_dict[int(df_ieee["Publication Year"][indexes[0]])] += 1
            if indexes[1] == "scopus":
                if int(df_scopus["Year"][indexes[0]]) not in year_dict:
                    year_dict[int(df_scopus["Year"][indexes[0]])] = 0
                year_dict[int(df_scopus["Year"][indexes[0]])] += 1
            if indexes[1] == "SD":
                if int(df_sd["year"][indexes[0]]) not in year_dict:
                    year_dict[int(df_sd["year"][indexes[0]])] = 0
                year_dict[int(df_sd["year"][indexes[0]])] += 1
    #for year in year_dict:
        #print(year, year_dict[year])
    #break
    '''
    '''
print(len(all_my))
for x in all_my:
    print(x, all_my[x])
'''
for x in name_doi:
    if len(name_doi[x]) > 1:
        print(x, name_doi[x])
        print(x, name_index[x])

for x in doi_name:
    if len(doi_name[x]) > 1:
        print(x, doi_name[x])
        print(x, doi_index[x])

for x in range(len(df_ieee['Unnamed: 30'])):
    if not np.isnan(float(df_ieee['Unnamed: 30'][x])):
        print(x, df_ieee['Unnamed: 30'][x])
'''
