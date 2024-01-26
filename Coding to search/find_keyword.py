import pandas as pd  
import matplotlib.pyplot as plt 
import os

def make_a_plot(dict_multi, name_plot):
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
    plt.savefig(name_plot) 
    plt.show()
    plt.close()
   
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

limy = 1

keywords_all = set()
keywords_all_title = dict()
keywords_all_doi = dict()
title_keywords_all = dict()
doi_keywords_all = dict()

keywords_author = set()
keywords_author_title = dict()
keywords_author_doi = dict()
title_keywords_author = dict()
doi_keywords_author = dict()

keywords_ieee = set()
keywords_ieee_title = dict()
keywords_ieee_doi = dict()
title_keywords_ieee = dict()
doi_keywords_ieee = dict()

keywords_inspec_c = set()
keywords_inspec_c_title = dict()
keywords_inspec_c_doi = dict()
title_keywords_inspec_c = dict()
doi_keywords_inspec_c = dict()
 
keywords_inspec_nc = set()
keywords_inspec_nc_title = dict()
keywords_inspec_nc_doi = dict()
title_keywords_inspec_nc = dict()
doi_keywords_inspec_nc = dict()

keywords_mesh = set()
keywords_mesh_title = dict()
keywords_mesh_doi = dict()
title_keywords_mesh = dict()
doi_keywords_mesh = dict()

doi_year = dict()
title_year = dict()

doi_abstract = dict()
title_abstract = dict()

for i in range(len(df_ieee["Document Title"])):
    tmp_year = int(df_ieee["Publication Year"][i])
    if tmp_year < limy:
        continue 
    tmp_name = str(df_ieee["Document Title"][i]).lower()
    tmp_doi = str(df_ieee["DOI"][i]).lower()
    if tmp_doi not in doi_year:
        doi_year[tmp_doi] = set()
    doi_year[tmp_doi].add(tmp_year)
    tmp_abstract = str(df_ieee["Abstract"][i])
    if tmp_doi not in doi_abstract:
        doi_abstract[tmp_doi] = set()
    doi_abstract[tmp_doi].add(tmp_abstract)
    if tmp_name not in title_abstract:
        title_abstract[tmp_name] = set()
    title_abstract[tmp_name].add(tmp_abstract)
    if tmp_name not in title_year:
        title_year[tmp_name] = set()
    title_year[tmp_name].add(tmp_year) 
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
    tmp_keywords = df_ieee["Author Keywords"][i] 
    if str(tmp_keywords) != "nan":
        for kw_tmp_comma in tmp_keywords.split(","):
            kw_tmp_semicolon = kw_tmp_comma.split(";")
            for kw_tmp in kw_tmp_semicolon:
                kw = kw_tmp  + "" 
                while len(kw) > 0 and kw[0] == " ":
                    kw = kw[1:]
                while len(kw) > 0 and kw[-1] == " ":
                    kw = kw[:-1]
                if len(kw) == 0:
                    continue   
                keywords_author.add(kw)
                if kw not in keywords_author_title:
                    keywords_author_title[kw] = set()
                keywords_author_title[kw].add(tmp_name)
                if kw not in keywords_author_doi:
                    keywords_author_doi[kw] = set()
                keywords_author_doi[kw].add(tmp_doi)
                if tmp_name not in title_keywords_author:
                    title_keywords_author[tmp_name] = set()
                title_keywords_author[tmp_name].add(kw)
                if tmp_doi not in doi_keywords_author:
                    doi_keywords_author[tmp_doi] = set()
                doi_keywords_author[tmp_doi].add(kw)
                kw_lower = kw
                keywords_all.add(kw_lower)
                if kw_lower not in keywords_all_title:
                    keywords_all_title[kw_lower] = set()
                keywords_all_title[kw_lower].add(tmp_name)
                if kw_lower not in keywords_all_doi:
                    keywords_all_doi[kw_lower] = set()
                keywords_all_doi[kw_lower].add(tmp_doi)
                if tmp_name not in title_keywords_all:
                    title_keywords_all[tmp_name] = set()
                title_keywords_all[tmp_name].add(kw_lower)
                if tmp_doi not in doi_keywords_all:
                    doi_keywords_all[tmp_doi] = set()
                doi_keywords_all[tmp_doi].add(kw_lower)
    tmp_keywords = df_ieee["IEEE Terms"][i] 
    if str(tmp_keywords) != "nan":
        for kw_tmp_comma in tmp_keywords.split(","):
            kw_tmp_semicolon = kw_tmp_comma.split(";")
            for kw_tmp in kw_tmp_semicolon:
                kw = kw_tmp  + "" 
                while len(kw) > 0 and kw[0] == " ":
                    kw = kw[1:]
                while len(kw) > 0 and kw[-1] == " ":
                    kw = kw[:-1]
                if len(kw) == 0:
                    continue  
                keywords_ieee.add(kw)
                if kw not in keywords_ieee_title:
                    keywords_ieee_title[kw] = set()
                keywords_ieee_title[kw].add(tmp_name)
                if kw not in keywords_ieee_doi:
                    keywords_ieee_doi[kw] = set()
                keywords_ieee_doi[kw].add(tmp_doi)
                if tmp_name not in title_keywords_ieee:
                    title_keywords_ieee[tmp_name] = set()
                title_keywords_ieee[tmp_name].add(kw)
                if tmp_doi not in doi_keywords_ieee:
                    doi_keywords_ieee[tmp_doi] = set()
                doi_keywords_ieee[tmp_doi].add(kw)
                kw_lower = kw
                keywords_all.add(kw_lower)
                if kw_lower not in keywords_all_title:
                    keywords_all_title[kw_lower] = set()
                keywords_all_title[kw_lower].add(tmp_name)
                if kw_lower not in keywords_all_doi:
                    keywords_all_doi[kw_lower] = set()
                keywords_all_doi[kw_lower].add(tmp_doi)
                if tmp_name not in title_keywords_all:
                    title_keywords_all[tmp_name] = set()
                title_keywords_all[tmp_name].add(kw_lower)
                if tmp_doi not in doi_keywords_all:
                    doi_keywords_all[tmp_doi] = set()
                doi_keywords_all[tmp_doi].add(kw_lower)
    tmp_keywords = df_ieee["INSPEC Controlled Terms"][i]
    if str(tmp_keywords) != "nan":
        for kw_tmp_comma in tmp_keywords.split(","):
            kw_tmp_semicolon = kw_tmp_comma.split(";")
            for kw_tmp in kw_tmp_semicolon:
                kw = kw_tmp  + "" 
                while len(kw) > 0 and kw[0] == " ":
                    kw = kw[1:]
                while len(kw) > 0 and kw[-1] == " ":
                    kw = kw[:-1]
                if len(kw) == 0:
                    continue  
                keywords_inspec_c.add(kw)
                if kw not in keywords_inspec_c_title:
                    keywords_inspec_c_title[kw] = set()
                keywords_inspec_c_title[kw].add(tmp_name)
                if kw not in keywords_inspec_c_doi:
                    keywords_inspec_c_doi[kw] = set()
                keywords_inspec_c_doi[kw].add(tmp_doi)
                if tmp_name not in title_keywords_inspec_c:
                    title_keywords_inspec_c[tmp_name] = set()
                title_keywords_inspec_c[tmp_name].add(kw)
                if tmp_doi not in doi_keywords_inspec_c:
                    doi_keywords_inspec_c[tmp_doi] = set()
                doi_keywords_inspec_c[tmp_doi].add(kw)
                kw_lower = kw
                keywords_all.add(kw_lower)
                if kw_lower not in keywords_all_title:
                    keywords_all_title[kw_lower] = set()
                keywords_all_title[kw_lower].add(tmp_name)
                if kw_lower not in keywords_all_doi:
                    keywords_all_doi[kw_lower] = set()
                keywords_all_doi[kw_lower].add(tmp_doi)
                if tmp_name not in title_keywords_all:
                    title_keywords_all[tmp_name] = set()
                title_keywords_all[tmp_name].add(kw_lower)
                if tmp_doi not in doi_keywords_all:
                    doi_keywords_all[tmp_doi] = set()
                doi_keywords_all[tmp_doi].add(kw_lower)
    tmp_keywords = df_ieee["INSPEC Non-Controlled Terms"][i]
    if str(tmp_keywords) != "nan":
        for kw_tmp_comma in tmp_keywords.split(","):
            kw_tmp_semicolon = kw_tmp_comma.split(";")
            for kw_tmp in kw_tmp_semicolon:
                kw = kw_tmp  + "" 
                while len(kw) > 0 and kw[0] == " ":
                    kw = kw[1:]
                while len(kw) > 0 and kw[-1] == " ":
                    kw = kw[:-1]
                if len(kw) == 0:
                    continue 
                keywords_inspec_nc.add(kw)
                if kw not in keywords_inspec_nc_title:
                    keywords_inspec_nc_title[kw] = set()
                keywords_inspec_nc_title[kw].add(tmp_name)
                if kw not in keywords_inspec_nc_doi:
                    keywords_inspec_nc_doi[kw] = set()
                keywords_inspec_nc_doi[kw].add(tmp_doi)
                if tmp_name not in title_keywords_inspec_nc:
                    title_keywords_inspec_nc[tmp_name] = set()
                title_keywords_inspec_nc[tmp_name].add(kw)
                if tmp_doi not in doi_keywords_inspec_nc:
                    doi_keywords_inspec_nc[tmp_doi] = set()
                doi_keywords_inspec_nc[tmp_doi].add(kw)
                kw_lower = kw
                keywords_all.add(kw_lower)
                if kw_lower not in keywords_all_title:
                    keywords_all_title[kw_lower] = set()
                keywords_all_title[kw_lower].add(tmp_name)
                if kw_lower not in keywords_all_doi:
                    keywords_all_doi[kw_lower] = set()
                keywords_all_doi[kw_lower].add(tmp_doi)
                if tmp_name not in title_keywords_all:
                    title_keywords_all[tmp_name] = set()
                title_keywords_all[tmp_name].add(kw_lower)
                if tmp_doi not in doi_keywords_all:
                    doi_keywords_all[tmp_doi] = set()
                doi_keywords_all[tmp_doi].add(kw_lower)
    tmp_keywords = df_ieee["Mesh_Terms"][i]
    if str(tmp_keywords) != "nan":
        for kw_tmp_comma in tmp_keywords.split(","):
            kw_tmp_semicolon = kw_tmp_comma.split(";")
            for kw_tmp in kw_tmp_semicolon:
                kw = kw_tmp  + "" 
                while len(kw) > 0 and kw[0] == " ":
                    kw = kw[1:]
                while len(kw) > 0 and kw[-1] == " ":
                    kw = kw[:-1]
                if len(kw) == 0:
                    continue
                keywords_mesh.add(kw)
                if kw not in keywords_mesh_title:
                    keywords_mesh_title[kw] = set()
                keywords_mesh_title[kw].add(tmp_name)
                if kw not in keywords_mesh_doi:
                    keywords_mesh_doi[kw] = set()
                keywords_mesh_doi[kw].add(tmp_doi)
                if tmp_name not in title_keywords_mesh:
                    title_keywords_mesh[tmp_name] = set()
                title_keywords_mesh[tmp_name].add(kw)
                if tmp_doi not in doi_keywords_mesh:
                    doi_keywords_mesh[tmp_doi] = set()
                doi_keywords_mesh[tmp_doi].add(kw)
                kw_lower = kw
                keywords_all.add(kw_lower)
                if kw_lower not in keywords_all_title:
                    keywords_all_title[kw_lower] = set()
                keywords_all_title[kw_lower].add(tmp_name)
                if kw_lower not in keywords_all_doi:
                    keywords_all_doi[kw_lower] = set()
                keywords_all_doi[kw_lower].add(tmp_doi)
                if tmp_name not in title_keywords_all:
                    title_keywords_all[tmp_name] = set()
                title_keywords_all[tmp_name].add(kw_lower)
                if tmp_doi not in doi_keywords_all:
                    doi_keywords_all[tmp_doi] = set()
                doi_keywords_all[tmp_doi].add(kw_lower)
print(len(name_doi), len(doi_name))

c = 0
print("Author Keywords")
for x in keywords_author_title:
    if len(keywords_author_title[x]) > 10:
        #print(x, len(keywords_author_title[x]))
        c += 1
print(len(keywords_author_title), c)
c = 0
print("IEEE Terms")
for x in keywords_ieee_title:
    if len(keywords_ieee_title[x]) > 10:
        #print(x, len(keywords_ieee_title[x]))
        c += 1
print(len(keywords_ieee_title), c)
c = 0
print("INSPEC Controlled Terms")
for x in keywords_inspec_c_title:
    if len(keywords_inspec_c_title[x]) > 10:
        #print(x, len(keywords_inspec_c_title[x]))
        c += 1 
print(len(keywords_inspec_c_title), c)
c = 0
print("INSPEC Non-Controlled Terms")
for x in keywords_inspec_nc_title:
    if len(keywords_inspec_nc_title[x]) > 10:
        #print(x, len(keywords_inspec_nc_title[x]))
        c += 1
print(len(keywords_inspec_nc_title), c)
c = 0
print("Mesh_Terms")
for x in keywords_mesh_title:
    if len(keywords_mesh_title[x]) > 10:
        #print(x, len(keywords_mesh_title[x]))
        c += 1
print(len(keywords_mesh_title), c)

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
    if tmp_doi not in doi_year:
        doi_year[tmp_doi] = set()
    doi_year[tmp_doi].add(tmp_year) 
    if tmp_name not in title_year:
        title_year[tmp_name] = set()
    title_year[tmp_name].add(tmp_year)   
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
    seen_scopus.add(tmp_name)
    tmp_doi = str(df_scopus2["DOI"][i]).lower()
    if tmp_doi not in doi_year:
        doi_year[tmp_doi] = set()
    doi_year[tmp_doi].add(tmp_year) 
    if tmp_name not in title_year:
        title_year[tmp_name] = set()
    title_year[tmp_name].add(tmp_year)   
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

keywords_sd = set()
keywords_sd_title = dict()
keywords_sd_doi = dict()
title_keywords_sd = dict()
doi_keywords_sd = dict()

for i in range(len(df_sd["title"])):
    tmp_year = int(df_sd["year"][i])
    if tmp_year < limy:
        continue 
    tmp_name = str(df_sd["title"][i]).lower()
    tmp_doi = str(df_sd["doi"][i]).replace("https://doi.org/", "").lower()
    if tmp_doi not in doi_year:
        doi_year[tmp_doi] = set()
    doi_year[tmp_doi].add(tmp_year)
    tmp_abstract = str(df_sd["abstract"][i])
    if tmp_doi not in doi_abstract:
        doi_abstract[tmp_doi] = set()
    doi_abstract[tmp_doi].add(tmp_abstract)
    if tmp_name not in title_abstract:
        title_abstract[tmp_name] = set()
    title_abstract[tmp_name].add(tmp_abstract)
    if tmp_name not in title_year:
        title_year[tmp_name] = set()
    title_year[tmp_name].add(tmp_year)  
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
    tmp_keywords = df_sd["keywords"][i]
    if str(tmp_keywords) != "nan":
        for kw_tmp_comma in tmp_keywords.split(","):
            kw_tmp_semicolon = kw_tmp_comma.split(";")
            for kw_tmp in kw_tmp_semicolon:
                kw = kw_tmp  + "" 
                while len(kw) > 0 and kw[0] == " ":
                    kw = kw[1:]
                while len(kw) > 0 and kw[-1] == " ":
                    kw = kw[:-1]
                if len(kw) == 0:
                    continue
                keywords_sd.add(kw)
                if kw not in keywords_sd_title:
                    keywords_sd_title[kw] = set()
                keywords_sd_title[kw].add(tmp_name)
                if kw not in keywords_sd_doi:
                    keywords_sd_doi[kw] = set()
                keywords_sd_doi[kw].add(tmp_doi)
                if tmp_name not in title_keywords_sd:
                    title_keywords_sd[tmp_name] = set()
                title_keywords_sd[tmp_name].add(kw)
                if tmp_doi not in doi_keywords_sd:
                    doi_keywords_sd[tmp_doi] = set()
                doi_keywords_sd[tmp_doi].add(kw)
                kw_lower = kw
                keywords_all.add(kw_lower)
                if kw_lower not in keywords_all_title:
                    keywords_all_title[kw_lower] = set()
                keywords_all_title[kw_lower].add(tmp_name)
                if kw_lower not in keywords_all_doi:
                    keywords_all_doi[kw_lower] = set()
                keywords_all_doi[kw_lower].add(tmp_doi)
                if tmp_name not in title_keywords_all:
                    title_keywords_all[tmp_name] = set()
                title_keywords_all[tmp_name].add(kw_lower)
                if tmp_doi not in doi_keywords_all:
                    doi_keywords_all[tmp_doi] = set()
                doi_keywords_all[tmp_doi].add(kw_lower)
                if ";" in kw_lower or "," in kw_lower:
                    print(kw_lower)
print(len(name_doi), len(doi_name))
 
c = 0
print("keywords")
for x in keywords_sd_title:
    if len(keywords_sd_title[x]) > 10:
        #print(x, len(keywords_sd_title[x]))
        c += 1
print(len(keywords_sd_title), c)

c = 0
print("keywords all")
for x in keywords_all_title:
    if len(keywords_all_title[x]) > 100:
        #print(x, len(keywords_all_title[x]))
        c += 1
print(len(keywords_all_title), c)
 
maritime_keywords = set()
maritime_keywords.update({"maritime", "marine", "ship", "vessel"})
for x in keywords_all_title:
    if "maritime" in x.lower() or "marine" in x.lower() or "ship" in x.lower() or "vessel" in x.lower():
        #print(x, keywords_all_title[x]) 
        maritime_keywords.add(x)  
#print(len(maritime_keywords))
#print(maritime_keywords)
 
anomaly_keywords = set()
anomaly_keywords.update({"anomaly", "anomalous", "abnormal", "unusual"}) 
for x in keywords_all_title:
    if "anomaly" in x.lower() or "anomalous" in x.lower() or "abnormal" in x.lower() or "unusual" in x.lower():
        #print(x, keywords_all_title[x]) 
        anomaly_keywords.add(x)  
#print(len(anomaly_keywords))
#print(anomaly_keywords)
 
detect_keywords = set()
detect_keywords.update({"detect"}) 
for x in keywords_all_title:
    if "detect" in x.lower():
        #print(x, keywords_all_title[x]) 
        detect_keywords.add(x)  
#print(len(detect_keywords))
#print(detect_keywords)
 
analyze_keywords = set()
analyze_keywords.update({"analyze", "analysis"}) 
for x in keywords_all_title:
    if "analyze" in x.lower() or "analysis" in x.lower():
        #print(x, keywords_all_title[x]) 
        analyze_keywords.add(x)  
#print(len(analyze_keywords))
#print(analyze_keywords) 

learning_keywords = set()
learning_keywords.update({"machine learning", "ML", "deep learning", "DL"}) 
for x in keywords_all_title: 
    if "learning" in x:
        if ("e-learning" in x.lower() and "machine" not in x.lower() and "ensemble" not in x.lower()) or "elearning" in x.lower() or "distance learning" in x.lower():
            continue
        #print(x, keywords_all_title[x])
        learning_keywords.add(x) 
for x in keywords_all_title:
    if "ML" in x and "HTML" not in x and "XML" not in x:
        #print(x, keywords_all_title[x])
        learning_keywords.add(x) 
for x in keywords_all_title:
    if "DL" in x and "HDL" not in x:
        #print(x, keywords_all_title[x])
        learning_keywords.add(x) 
#print(len(learning_keywords))
#print(learning_keywords)

ml_keywords = set()
ml_keywords.update({"machine learning", "ML"}) 
for x in keywords_all_title:
    if "learning" in x.lower() and "machine" in x.lower():
        #print(x, keywords_all_title[x])
        ml_keywords.add(x)  
for x in keywords_all_title:
    if "ML" in x and "HTML" not in x and "XML" not in x:
        #print(x, keywords_all_title[x])
        ml_keywords.add(x)  
#print(len(ml_keywords))
#print(ml_keywords)

dl_keywords = set() 
dl_keywords.update({"deep learning", "DL"}) 
for x in keywords_all_title:
    if "learning" in x.lower() and "deep" in x.lower():
        #print(x, keywords_all_title[x])
        dl_keywords.add(x)  
for x in keywords_all_title:
    if "DL" in x and "HDL" not in x:
        #print(x, keywords_all_title[x])
        dl_keywords.add(x)  
#print(len(dl_keywords))
#print(dl_keywords)

artificial_keywords = set()
artificial_keywords.update({"artificial intellig", "AI"}) 
for x in keywords_all_title:
    if "artificial" in x.lower() and "line" not in x.lower() and "trade" not in x.lower() and "rabbits" not in x.lower() and "image" not in x.lower() and "immune" not in x.lower() and "artificial intelligent methods" not in x.lower() and "custom artificial intelligence streaming architecture" not in x.lower():
        #print(x, keywords_all_title[x])
        artificial_keywords.add(x) 
#print(len(artificial_keywords))
#print(artificial_keywords)
 
nn_keywords = set()
nn_keywords.update({"neural network", "NN"})
for x in keywords_all_title:
    if "neural" in x.lower() and "network" in x.lower():
        #print(x, keywords_all_title[x])
        nn_keywords.add(x) 
for x in keywords_all_title:
    if "CNN" in x or "RNN" in x: 
        #print(x, keywords_all_title[x])
        nn_keywords.add(x)  
#print(len(nn_keywords))
#print(nn_keywords)
 
cnn_keywords = set()
cnn_keywords.update({"CNN", "convolutional neural network", "convolutional NN"}) 
for x in keywords_all_title:
    if ("neural" in x.lower() and "network" in x.lower() and "convolutional" in x.lower()) or "CNN" in x or ("NN" in x and "convolutional" in x.lower()):
        #print(x, keywords_all_title[x]) 
        cnn_keywords.add(x)  
#print(len(cnn_keywords))
#print(cnn_keywords)

kde_keywords = set()
kde_keywords.update({"kernel density estimation", "KDE"})
for x in keywords_all_title:
    if ("kernel" in x.lower() and "density" in x.lower() and "estimation" in x.lower()) or "KDE" in x :
        #print(x, keywords_all_title[x]) 
        kde_keywords.add(x)  
#print(len(kde_keywords))
print(kde_keywords)

ou_keywords = set()
ou_keywords.update({"ornstein", "uhlenbeck", "OU", "O-U"})
for x in keywords_all_title:
    if "ornstein" in x.lower() or "uhlenbeck" in x.lower() or ("OU" in x and "MOU" not in x and "EIOU" not in x):
        #print(x, keywords_all_title[x]) 
        ou_keywords.add(x)  
#print(len(ou_keywords))
print(ou_keywords)

rnn_keywords = set()
rnn_keywords.update({"RNN", "recurrent neural network", "recurrent NN"}) 
for x in keywords_all_title:
    if ("recurrent" in x.lower() and "network" in x.lower() and "neural" in x.lower()) or "RNN" in x:
        #print(x, keywords_all_title[x]) 
        rnn_keywords.add(x)  
#print(len(rnn_keywords))
#print(rnn_keywords)

vrnn_keywords = set()
vrnn_keywords.update({"VRNN", "variational recurrent neural network", "variational recurrent NN", "variational RNN", "VRNN", "V-RNN"}) 
for x in keywords_all_title:
    if ("variational" in x.lower() and "recurrent" in x.lower() and "network" in x.lower() and "neural" in x.lower()) or "VRNN" in x or "V-RNN" in x or "V-RNN" in x or ("RNN" in x and "variational" in x.lower()) or ("NN" in x and "variational" in x.lower() and "recurrent" in x.lower()):
        #print(x, keywords_all_title[x]) 
        vrnn_keywords.add(x)  
#print(len(vrnn_keywords))
#print(vrnn_keywords)
 
ais_keywords = set()
ais_keywords.update({"AIS", "automatic identification system"}) 
for x in keywords_all_title:
    if (("automatic" in x.lower() and "identification" in x.lower() and "system" in x.lower()) or "AIS" in x) and "CAIS" not in x:
        #print(x, keywords_all_title[x]) 
        ais_keywords.add(x)  
#print(len(ais_keywords))
#print(ais_keywords)

location_keywords = set()
location_keywords.update({"GPS", "automatic identification system", "GPS", "global navigation sattelite system", "global positioning system", "location", "position", "longitude", "latitude"}) 
for x in keywords_all_title:
    if "GPS" in x or ("global" in x.lower() and "positioning" in x.lower() and "system" in x.lower()) or ("global" in x.lower() and "navigation" in x.lower() and "sattelite" in x.lower() and "system" in x.lower()) or "GNSS" in x or "location" in x.lower() or "longitude" in x.lower() or "latitude" in x.lower() or "position" in x.lower():
        #print(x, keywords_all_title[x]) 
        location_keywords.add(x)  
#print(len(location_keywords))
#print(location_keywords)

visual_keywords = set()
visual_keywords.update({"video", "image", "radar", "visual", "camera", "visible", "picture"}) 
for x in keywords_all_title:
    if "video" in x.lower() or "image" in x.lower() or "radar" in x.lower() or "visual" in x.lower() or "camera" in x.lower() or "visible" in x.lower() or "picture" in x.lower():
        #print(x, keywords_all_title[x]) 
        visual_keywords.add(x)  
#print(len(visual_keywords))
#print(visual_keywords)

camera_keywords = set()
camera_keywords.update({"video", "image", "picture", "camera"}) 
for x in keywords_all_title:
    if "video" in x.lower() or "image" in x.lower() or "picture" in x.lower() or "camera" in x.lower():
        #print(x, keywords_all_title[x]) 
        camera_keywords.add(x)  
#print(len(camera_keywords))
#print(camera_keywords)

radar_keywords = set()
radar_keywords.update({"synthetic apperture radar", "SAR", "radar"})
for x in keywords_all_title:
    if "SAR" in x or ("synthetic" in x.lower() and "radar" in x.lower() and "apperture" in x.lower()) or "radar" in x.lower():
        #print(x, keywords_all_title[x]) 
        radar_keywords.add(x)  
#print(len(radar_keywords))
#print(radar_keywords)

sar_keywords = set()
sar_keywords.update({"synthetic apperture radar", "SAR"})
for x in keywords_all_title:
    if "SAR" in x or ("synthetic" in x.lower() and "radar" in x.lower() and "apperture" in x.lower()):
        #print(x, keywords_all_title[x]) 
        sar_keywords.add(x)  
#print(len(sar_keywords))
#print(sar_keywords)

hmm_keywords = set()
hmm_keywords.update({"hidden Markov model", "HMM", "hidden markov model"})
for x in keywords_all_title:
    if "HMM" in x or ("hidden" in x.lower() and "markov" in x.lower() and "model" in x.lower()):
        #print(x, keywords_all_title[x]) 
        hmm_keywords.add(x)  
#print(len(hmm_keywords))
#print(hmm_keywords)

svr_keywords = set()
svr_keywords.update({"support vector regression", "SVR"})
for x in keywords_all_title:
    if "SVR" in x or ("support" in x.lower() and "vector" in x.lower() and "regression" in x.lower()):
        #print(x, keywords_all_title[x]) 
        svr_keywords.add(x)  
#print(len(svr_keywords))
#print(svr_keywords)

svm_keywords = set()
svm_keywords.update({"support vector machine", "SVM"})
for x in keywords_all_title:
    if "SVM" in x or ("support" in x.lower() and "vector" in x.lower() and "machine" in x.lower()):
        #print(x, keywords_all_title[x]) 
        svm_keywords.add(x)  
#print(len(svm_keywords))
#print(svm_keywords)

lstm_keywords = set()
lstm_keywords.update({"long short term memory", "LSTM"})
for x in keywords_all_title:
    if "LSTM" in x or ("long" in x.lower() and "short" in x.lower() and "term" in x.lower() and "memory" in x.lower()):
        #print(x, keywords_all_title[x]) 
        lstm_keywords.add(x)  
#print(len(lstm_keywords))
#print(lstm_keywords)

cluster_keywords = set()
cluster_keywords.update({"cluster"})
for x in keywords_all_title:
    if "cluster" in x.lower():
        #print(x, keywords_all_title[x]) 
        cluster_keywords.add(x)  
#print(len(cluster_keywords))
#print(cluster_keywords)

DBSCAN_keywords = set()
DBSCAN_keywords.update({"DBSCAN"})
for x in keywords_all_title:
    if "DBSCAN" in x:
        #print(x, keywords_all_title[x]) 
        DBSCAN_keywords.add(x)  
#print(len(DBSCAN_keywords))
#print(DBSCAN_keywords)

kmeans_keywords = set()
kmeans_keywords.update({"K-means"})
for x in keywords_all_title:
    if "K-means" in x:
        #print(x, keywords_all_title[x]) 
        kmeans_keywords.add(x)  
#print(len(kmeans_keywords))
#print(kmeans_keywords)

kalman_keywords = set()
kalman_keywords.update({"kalman"})
for x in keywords_all_title:
    if "kalman" in x.lower():
        #print(x, keywords_all_title[x]) 
        kalman_keywords.add(x)  
#print(len(kalman_keywords))
#print(kalman_keywords)

gauss_keywords = set()
gauss_keywords.update({"gauss"})
for x in keywords_all_title:
    if "gauss" in x.lower():
        #print(x, keywords_all_title[x]) 
        gauss_keywords.add(x)  
#print(len(gauss_keywords))
#print(gauss_keywords)

GMM_keywords = set()
GMM_keywords.update({"gaussian mixture model", "GMM"})
for x in keywords_all_title:
    if ("gauss" in x.lower() and "mixture" in x.lower() and "model" in x.lower()) or "GMM" in x:
        #print(x, keywords_all_title[x]) 
        GMM_keywords.add(x)  
#print(len(GMM_keywords))
#print(GMM_keywords)

knn_keywords = set()
knn_keywords.update({"k-nearest neighbor", "k-nearest neighbour", "k nearest neighbor", "k nearest neighbour", "k-NN", "kNN"})
for x in keywords_all_title:
    if "k-nearest neighbour" in x.lower() or "k nearest neighbour" in x.lower() or "k-nearest neighbor" in x.lower() or "k nearest neighbor" in x.lower() or "k-NN" in x or "K-NN" in x or "KNN" in x or "kNN" in x:
        #print(x, keywords_all_title[x]) 
        knn_keywords.add(x)  
#print(len(knn_keywords))
#print(knn_keywords)
  
bayes_keywords = set()
bayes_keywords.update({"bayes"})
for x in keywords_all_title:
    if "bayes" in x.lower():
        #print(x, keywords_all_title[x]) 
        bayes_keywords.add(x)  
#print(len(bayes_keywords))
#print(bayes_keywords)

autoencoder_keywords = set()
autoencoder_keywords.update({"autoencoder", "enocde", "decode"})
for x in keywords_all_title:
    if "autoencoder" in x.lower() or "encode" in x.lower() or "decode" in x.lower():
        #print(x, keywords_all_title[x]) 
        autoencoder_keywords.add(x)  
#print(len(autoencoder_keywords))
#print(autoencoder_keywords)

speed_keywords = set() 
w_speed_keywords = set() 
for x in keywords_all_title:
    if "speed" in x.lower() and "high-speed systems" not in x.lower() and "high data speed" not in x.lower() and "data rate" not in x.lower() and "information service" not in x.lower() and "sound" not in x.lower() and "internet" not in x.lower() and "processor" not in x.lower() and "engine" not in x.lower() and "disk" not in x.lower() and "connect" not in x.lower() and "magnet" not in x.lower() and "wireless" not in x.lower() and "transmission" not in x.lower() and "link" not in x.lower() and "memory" not in x.lower() and "communication" not in x.lower() and "shaft" not in x.lower():
        if "wind" in x.lower():
            #print(x, keywords_all_title[x]) 
            w_speed_keywords.add(x)  
        else:
            #print(x, keywords_all_title[x]) 
            speed_keywords.add(x)  
#print(len(speed_keywords))
#print(speed_keywords)
#print(len(w_speed_keywords))
#print(w_speed_keywords)

heading_keywords = set()
for x in keywords_all_title:
    if "heading" in x.lower():
        #print(x, keywords_all_title[x]) 
        heading_keywords.add(x)  
#print(len(heading_keywords))
#print(heading_keywords)

course_keywords = set() 
for x in keywords_all_title:
    if "course" in x.lower() and "education" not in x.lower() and "doing" not in x.lower() and "cadet" not in x.lower() and "level" not in x.lower() and "renovation" not in x.lower() and "action" not in x.lower() and "engineering" not in x.lower():   
        #print(x, keywords_all_title[x]) 
        course_keywords.add(x)  
#print(len(course_keywords))
#print(course_keywords)

trajectory_keywords = set()
for x in keywords_all_title:
    if "trajector" in x.lower():
        #print(x, keywords_all_title[x]) 
        trajectory_keywords.add(x)  
#print(len(trajectory_keywords))
#print(trajectory_keywords)

behaviour_keywords = set()
for x in keywords_all_title:
    if "trafficking" in x.lower() or "piracy" in x.lower() or "smuggling" in x.lower() or "illegal" in x.lower():
        #print(x, keywords_all_title[x]) 
        behaviour_keywords.add(x)  
#print(len(behaviour_keywords))
#print(behaviour_keywords)

sonar_keywords = set()
for x in keywords_all_title:
    if "sonar" in x.lower():
        #print(x, keywords_all_title[x]) 
        sonar_keywords.add(x)  
#print(len(sonar_keywords))
#print(sonar_keywords)

communication_keywords = set()
for x in keywords_all_title:
    if "communication" in x.lower():
        #print(x, keywords_all_title[x]) 
        communication_keywords.add(x)  
#print(len(communication_keywords))
#print(communication_keywords)

anomaly_keywords = set()
for x in keywords_all_title:
    if "anomal" in x.lower():
        #print(x, keywords_all_title[x]) 
        anomaly_keywords.add(x)  
#print(len(anomaly_keywords))
#print(anomaly_keywords)

data_mining_keywords = set()
for x in keywords_all_title:
    if "data mining" in x.lower():
        #print(x, keywords_all_title[x]) 
        data_mining_keywords.add(x)  
#print(len(data_mining_keywords))
#print(data_mining_keywords)

graph_keywords = set()
for x in keywords_all_title:
    if "graph" in x.lower() and "graphic" not in x.lower() and "graphy" not in x.lower():
        #print(x, keywords_all_title[x]) 
        graph_keywords.add(x)  
#print(len(graph_keywords))
#print(graph_keywords)

GAM_keywords = set()
GAM_keywords.update({"generalized additive model", "generalized Additive model", "generalized Additive Model", "generalized additive Model", "GAM"})
for x in keywords_all_title:
    if ("generalized" in x.lower() and "additive" in x.lower() and "model" in x.lower()) or "Additive" in x:
        #print(x, keywords_all_title[x]) 
        GAM_keywords.add(x)  
#print(len(GAM_keywords))
#print(GAM_keywords)

RF_keywords = set()
RF_keywords.update({"random forest", "random Forest", "RF"})
for x in keywords_all_title:
    if ("random" in x.lower() and "forest" in x.lower()) or "RF" in x:
        #print(x, keywords_all_title[x]) 
        RF_keywords.add(x)  
#print(len(RF_keywords))
#print(RF_keywords)

read_papers = set()
for name in title_keywords_all:
    is_ok = False
    for kw in title_keywords_all[name]:
        if kw in learning_keywords or kw in ml_keywords or kw in dl_keywords or kw in artificial_keywords or kw in nn_keywords or kw in cnn_keywords or kw in rnn_keywords or kw in gauss_keywords or kw in graph_keywords or kw in knn_keywords or kw in bayes_keywords or kw in hmm_keywords or kw in DBSCAN_keywords or kw in kmeans_keywords or kw in svr_keywords or kw in svm_keywords  or kw in knn_keywords or kw in bayes_keywords or kw in hmm_keywords or kw in DBSCAN_keywords or kw in kmeans_keywords or kw in svr_keywords or kw in kalman_keywords:
            is_ok = True
            break
    if not is_ok:
        continue
    '''
    is_ok = False
    for kw in title_keywords_all[name]:
        if kw in ais_keywords or kw in location_keywords or kw in trajectory_keywords:
            is_ok = True
            break
    if not is_ok:
        continue
    '''
    is_ok = False
    for kw in title_keywords_all[name]:
        if kw in anomaly_keywords:
            is_ok = True
            break
    if not is_ok:
        continue
    is_ok = True
    for kw in title_keywords_all[name]:
        if kw in visual_keywords or kw in camera_keywords or kw in radar_keywords or kw in sar_keywords or kw in sonar_keywords:
            is_ok = False
            break
    if is_ok:
        read_papers.add(name)
#for x in read_papers:
    #print(x, title_keywords_all[x])
    #print(x)
    #print(title_keywords_all[x])
print(len(read_papers))
#print(read_papers)

'''
find_new_tags = set()
for name in read_papers:
    for kw in title_keywords_all[name]:
        if kw in learning_keywords or kw in ml_keywords or kw in dl_keywords or kw in artificial_keywords or kw in nn_keywords or kw in cnn_keywords or kw in rnn_keywords or kw in gauss_keywords or kw in graph_keywords or kw in knn_keywords or kw in bayes_keywords or kw in hmm_keywords or kw in DBSCAN_keywords or kw in kmeans_keywords:
            continue
        if kw in ais_keywords or kw in location_keywords or kw in trajectory_keywords:
            continue
        find_new_tags.add(kw)    
print(len(find_new_tags))
print(find_new_tags)
'''

scopus_filter = set()
for i in range(len(df_scopus["Title"])):
    if int(df_scopus["Year"][i]) < limy:
        continue
    tmp_name = str(df_scopus["Title"][i]).lower()
    tmp_doi = str(df_scopus["DOI"][i]).lower()
    if "anomal" in tmp_name and "maritime" in tmp_name and "image" not in tmp_name and tmp_name not in read_papers:
        if "video" not in tmp_name and "image" not in tmp_name and "radar" not in tmp_name and "visual" not in tmp_name and "camera" not in tmp_name and "visible" not in tmp_name and "sonar" not in tmp_name and "radar" not in tmp_name:
            scopus_filter.add(tmp_name)
for i in range(len(df_scopus2["Title"])):
    if int(df_scopus2["Year"][i]) < limy:
        continue
    tmp_name = str(df_scopus2["Title"][i]).lower()
    tmp_doi = str(df_scopus2["DOI"][i]).lower()
    if "anomal" in tmp_name and "maritime" in tmp_name and "image" not in tmp_name and tmp_name not in read_papers:
        if "video" not in tmp_name and "image" not in tmp_name and "radar" not in tmp_name and "visual" not in tmp_name and "camera" not in tmp_name and "visible" not in tmp_name and "sonar" not in tmp_name and "radar" not in tmp_name:
            scopus_filter.add(tmp_name)
print(len(scopus_filter)) 
print(scopus_filter)

ieee_filter = set()
for i in range(len(df_ieee["Document Title"])):
    if int(df_ieee["Publication Year"][i]) < limy:
        continue
    tmp_name = str(df_ieee["Document Title"][i]).lower()
    tmp_doi = str(df_ieee["DOI"][i]).lower()
    tmp_abs = str(df_ieee["Abstract"][i]).lower()
    if (("anomal" in tmp_name and "maritime" in tmp_name) or ("anomal" in tmp_abs and "maritime" in tmp_abs)) and tmp_name not in read_papers:
        if "video" not in tmp_name and "image" not in tmp_name and "radar" not in tmp_name and "visual" not in tmp_name and "camera" not in tmp_name and "visible" not in tmp_name and "sonar" not in tmp_name and "radar" not in tmp_name:
            if "video" not in tmp_abs and "image" not in tmp_abs and "radar" not in tmp_abs and "visual" not in tmp_abs and "camera" not in tmp_abs and "visible" not in tmp_abs and "sonar" not in tmp_abs and "radar" not in tmp_abs:
                ieee_filter.add(tmp_name) 
print(len(ieee_filter))

sd_filter = set()
for i in range(len(df_sd["title"])):
    if int(df_sd["year"][i]) < limy:
        continue
    tmp_name = str(df_sd["title"][i]).lower()
    tmp_doi = str(df_sd["doi"][i]).lower()
    tmp_abs = str(df_sd["abstract"][i]).lower()
    if (("anomal" in tmp_name and "maritime" in tmp_name) or ("anomal" in tmp_abs and "maritime" in tmp_abs)) and tmp_name not in read_papers:
        if "video" not in tmp_name and "image" not in tmp_name and "radar" not in tmp_name and "visual" not in tmp_name and "camera" not in tmp_name and "visible" not in tmp_name and "sonar" not in tmp_name and "radar" not in tmp_name:
            if "video" not in tmp_abs and "image" not in tmp_abs and "radar" not in tmp_abs and "visual" not in tmp_abs and "camera" not in tmp_abs and "visible" not in tmp_abs and "sonar" not in tmp_abs and "radar" not in tmp_abs:
                sd_filter.add(tmp_name) 
print(len(sd_filter))

doi_year_one = dict()
title_year_one = dict()

for doi in doi_year:
    list_year = list(doi_year[doi])
    min_year = min(list_year)
    doi_year_one[doi] = min_year

for name in title_year:
    list_year = list(title_year[name])
    min_year = min(list_year)
    title_year_one[name] = min_year

my_dict_read = dict()
for name in read_papers:
    year_paper = title_year_one[name]
    if year_paper not in my_dict_read:
        my_dict_read[year_paper] = 0
    my_dict_read[year_paper] += 1

my_dict_sd = dict()
for name in sd_filter:
    year_paper = title_year_one[name]
    if year_paper not in my_dict_sd:
        my_dict_sd[year_paper] = 0
    my_dict_sd[year_paper] += 1
 
#make_a_plot({"IEEE scopus": my_dict_read, "SD": my_dict_sd})

save_new_table = '"name","keywords"\n'
for name in read_papers:
    save_new_table += '"' + name + '","'
    for kw in title_keywords_all[name]:
        save_new_table += kw + ","
    save_new_table += '"\n'
#print(save_new_table)

open_save_new_table = open("save_new_table.csv", "w", encoding = "utf-8")
open_save_new_table.write(save_new_table)
open_save_new_table.close()
 
save_sd_table = '"name"\n'
for name in sd_filter:
    save_sd_table += '"' + name + '"\n'
#print(save_sd_table)

open_save_sd_table = open("save_sd_table.csv", "w", encoding = "utf-8")
open_save_sd_table.write(save_sd_table)
open_save_sd_table.close()

save_new_ieee = '"name","keywords"\n'
for name in ieee_filter:
    save_new_ieee += '"' + name + '","'
    if name in title_keywords_all:
        for kw in title_keywords_all[name]:
            save_new_ieee += kw + ","
    save_new_ieee += '"\n'
#print(save_new_ieee)

open_save_new_ieee = open("save_new_ieee.csv", "w", encoding = "utf-8")
open_save_new_ieee.write(save_new_ieee)
open_save_new_ieee.close()

save_new_scopus = '"name","keywords"\n'
for name in scopus_filter:
    save_new_scopus += '"' + name + '","'
    if name in title_keywords_all:
        for kw in title_keywords_all[name]:
            save_new_scopus += kw + ","
    save_new_scopus += '"\n'
#print(save_new_scopus)

open_save_new_scopus = open("save_new_scopus.csv", "w", encoding = "utf-8")
open_save_new_scopus.write(save_new_scopus)
open_save_new_scopus.close()

save_new_all = '"name","keywords"\n'
for name in scopus_filter.union(sd_filter).union(ieee_filter).union(read_papers):
    save_new_all += '"' + name + '","'
    if name in title_keywords_all:
        for kw in title_keywords_all[name]:
            save_new_all += kw + ","
    save_new_all += '"\n'
#print(save_new_all)

open_save_new_all = open("save_new_all.csv", "w", encoding = "utf-8")
open_save_new_all.write(save_new_all)
open_save_new_all.close()

new_papers_by_topic = dict()

def save_keyword_papers(set_of_kw, name_of_file):
    save_new_some_kw = '"name","year","abstract","keywords"\n' 
    new_kw_papers = set() 

    for i in range(len(df_sd["title"])):
        tmp_year = int(df_sd["year"][i])
        if tmp_year < limy:
            continue
        tmp_name = str(df_sd["title"][i])
        name = tmp_name.lower()
        if name in new_kw_papers:
            continue 
        tmp_abs = str(df_sd["abstract"][i])
        for kw in set_of_kw: 
            if kw in tmp_name or kw in tmp_abs or kw.capitalize() in tmp_name or kw.capitalize() in tmp_abs or (name in title_keywords_all and kw in title_keywords_all[name]): 
                new_kw_papers.add(name)

    for i in range(len(df_ieee["Document Title"])):
        tmp_year = int(df_ieee["Publication Year"][i])
        if tmp_year < limy:
            continue
        tmp_name = str(df_ieee["Document Title"][i])
        name = tmp_name.lower()
        if name in new_kw_papers:
            continue 
        tmp_abs = str(df_ieee["Abstract"][i])
        for kw in set_of_kw: 
            if kw in tmp_name or kw in tmp_abs or kw.capitalize() in tmp_name or kw.capitalize() in tmp_abs or (name in title_keywords_all and kw in title_keywords_all[name]):
                new_kw_papers.add(name)

    for i in range(len(df_scopus["Title"])):
        tmp_year = int(df_scopus["Year"][i])
        if tmp_year < limy:
            continue
        tmp_name = str(df_scopus["Title"][i])
        name = tmp_name.lower()
        if name in new_kw_papers:
            continue  
        for kw in set_of_kw: 
            if kw in tmp_name or kw in tmp_abs or kw.capitalize() in tmp_name or kw.capitalize() in tmp_abs or (name in title_keywords_all and kw in title_keywords_all[name]): 
                new_kw_papers.add(name)
    for i in range(len(df_scopus2["Title"])):
        tmp_year = int(df_scopus2["Year"][i])
        if tmp_year < limy:
            continue
        tmp_name = str(df_scopus2["Title"][i])
        name = tmp_name.lower()
        if name in new_kw_papers:
            continue  
        for kw in set_of_kw:
            if kw in tmp_name or kw in tmp_abs or kw.capitalize() in tmp_name or kw.capitalize() in tmp_abs or (name in title_keywords_all and kw in title_keywords_all[name]): 
                new_kw_papers.add(name)

    for name in new_kw_papers:
        save_new_some_kw += '"' + name.replace('"', "'") + '","' + str(title_year_one[name]).replace('"', "'") + '","'
        if name in title_abstract:
            for abstract in title_abstract[name]:
                save_new_some_kw += abstract.replace('"', "'")
        save_new_some_kw += '","'
        if name in title_keywords_all:
            for kw in title_keywords_all[name]:
                save_new_some_kw += kw.replace('"', "'") + ","
        save_new_some_kw += '"\n'
    #print(save_new_all)

    if not os.path.isdir("keywords/"):
        os.makedirs("keywords")

    open_save_new_some_kw = open("keywords/" + name_of_file + ".csv", "w", encoding = "utf-8")
    open_save_new_some_kw.write(save_new_some_kw)
    open_save_new_some_kw.close()
    new_papers_by_topic[name_of_file] = new_kw_papers
    print(name_of_file, len(new_papers_by_topic[name_of_file]))
    return new_kw_papers 
    
save_keyword_papers(maritime_keywords, "maritime_keywords")
save_keyword_papers(anomaly_keywords, "anomaly_keywords")
save_keyword_papers(detect_keywords, "detect_keywords")
save_keyword_papers(analyze_keywords, "analyze_keywords")
save_keyword_papers(learning_keywords, "learning_keywords")
save_keyword_papers(ml_keywords, "ML_keywords")
save_keyword_papers(dl_keywords, "DL_keywords")
save_keyword_papers(artificial_keywords, "artificial_keywords")
save_keyword_papers(nn_keywords, "NN_keywords")
save_keyword_papers(cnn_keywords, "CNN_keywords")
save_keyword_papers(kde_keywords, "KDE_keywords")
save_keyword_papers(ou_keywords, "OU_keywords")
save_keyword_papers(rnn_keywords, "RNN_keywords")
save_keyword_papers(vrnn_keywords, "VRNN_keywords")
save_keyword_papers(ais_keywords, "AIS_keywords")
save_keyword_papers(location_keywords, "location_keywords")
save_keyword_papers(visual_keywords, "visual_keywords")
save_keyword_papers(camera_keywords, "camera_keywords")
save_keyword_papers(radar_keywords, "radar_keywords")
save_keyword_papers(sar_keywords, "SAR_keywords")
save_keyword_papers(hmm_keywords, "HMM_keywords")
save_keyword_papers(svr_keywords, "SVR_keywords")
save_keyword_papers(svm_keywords, "SVM_keywords")
save_keyword_papers(lstm_keywords, "LSTM_keywords")
save_keyword_papers(cluster_keywords, "cluster_keywords")
save_keyword_papers(DBSCAN_keywords, "DBSCAN_keywords")
save_keyword_papers(kmeans_keywords, "kmeans_keywords")
save_keyword_papers(kalman_keywords, "kalman_keywords")
save_keyword_papers(gauss_keywords, "gauss_keywords")
save_keyword_papers(GMM_keywords, "GMM_keywords")
save_keyword_papers(knn_keywords, "KNN_keywords")
save_keyword_papers(bayes_keywords, "bayes_keywords")
save_keyword_papers(autoencoder_keywords, "autoencoder_keywords")
save_keyword_papers(speed_keywords, "speed_keywords")
save_keyword_papers(heading_keywords, "heading_keywords")
save_keyword_papers(course_keywords, "course_keywords")
save_keyword_papers(trajectory_keywords, "trajectory_keywords")
save_keyword_papers(behaviour_keywords, "behaviour_keywords")
save_keyword_papers(sonar_keywords, "sonar_keywords")
save_keyword_papers(communication_keywords, "communication_keywords") 
save_keyword_papers(data_mining_keywords, "data_mining_keywords")
save_keyword_papers(graph_keywords, "graph_keywords")
save_keyword_papers(GAM_keywords, "GAM_keywords")
save_keyword_papers(RF_keywords, "RF_keywords")

def save_anything(new_kw_papers, name_of_file):
    save_new_some_kw = '"name","year","abstract","keywords"\n' 
    for name in new_kw_papers:
        save_new_some_kw += '"' + name.replace('"', "'") + '","' + str(title_year_one[name]).replace('"', "'") + '","'
        if name in title_abstract:
            for abstract in title_abstract[name]:
                save_new_some_kw += abstract.replace('"', "'")
        save_new_some_kw += '","'
        if name in title_keywords_all:
            for kw in title_keywords_all[name]:
                save_new_some_kw += kw.replace('"', "'") + ","
        save_new_some_kw += '"\n'
    #print(save_new_all)

    open_save_new_some_kw = open(name_of_file + ".csv", "w", encoding = "utf-8")
    open_save_new_some_kw.write(save_new_some_kw)
    open_save_new_some_kw.close()
    
detect_or_analyze = new_papers_by_topic["detect_keywords"].union(new_papers_by_topic["analyze_keywords"])
sets_intersect_maritime = dict()
sets_intersect_anomaly = dict()
sets_intersect_detect = dict()
sets_intersect_maritime_anomaly = dict()
sets_intersect_maritime_detect = dict()
sets_intersect_anomaly_detect = dict()
sets_intersect_maritime_anomaly_detect = dict()
for set_name in new_papers_by_topic:
    sets_intersect_maritime[set_name] = new_papers_by_topic[set_name].intersection(new_papers_by_topic["maritime_keywords"])
    sets_intersect_anomaly[set_name] = new_papers_by_topic[set_name].intersection(new_papers_by_topic["anomaly_keywords"])
    sets_intersect_detect[set_name] = new_papers_by_topic[set_name].intersection(detect_or_analyze)
    sets_intersect_maritime_anomaly[set_name] = sets_intersect_maritime[set_name].intersection(sets_intersect_anomaly[set_name])
    sets_intersect_maritime_detect[set_name] = sets_intersect_maritime[set_name].intersection(sets_intersect_detect[set_name])
    sets_intersect_anomaly_detect[set_name] = sets_intersect_anomaly[set_name].intersection(sets_intersect_detect[set_name])
    sets_intersect_maritime_anomaly_detect[set_name] = sets_intersect_maritime_anomaly[set_name].intersection(sets_intersect_detect[set_name])
    save_anything(sets_intersect_maritime[set_name], set_name + '_maritime')
    save_anything(sets_intersect_anomaly[set_name], set_name + '_anomaly')
    save_anything(sets_intersect_detect[set_name], set_name + '_detect')
    save_anything(sets_intersect_maritime_anomaly[set_name], set_name + '_maritime_anomaly')
    save_anything(sets_intersect_maritime_detect[set_name], set_name + '_maritime_detect')
    save_anything(sets_intersect_anomaly_detect[set_name], set_name + '_anomaly_detect') 
    save_anything(sets_intersect_maritime_anomaly_detect[set_name], set_name + '_maritime_anomaly_detect') 
     
def process_multi_sets(sets_topic):
    for depth1 in range(len(sets_topic)):
        name1 = sets_topic[depth1]
        print(name1, "Size", len(new_papers_by_topic[name1]))
        print("Maritime", len(sets_intersect_maritime[name1]))
        print("Anomaly", len(sets_intersect_anomaly[name1]))
        print("Detect", len(sets_intersect_detect[name1]))
        print("Maritime Anomaly", len(sets_intersect_maritime_anomaly[name1]))
        print("Maritime Detect", len(sets_intersect_maritime_detect[name1]))
        print("Anomaly Detect", len(sets_intersect_anomaly_detect[name1]))
        print("Maritime Anomaly Detect", len(sets_intersect_maritime_anomaly_detect[name1]))
        for depth2 in range(depth1 + 1, len(sets_topic)):
            name2 = sets_topic[depth2]
            print(name2, "intersect", len(new_papers_by_topic[name1].intersection(new_papers_by_topic[name2]))) 
            print(name2, "union", len(new_papers_by_topic[name1].union(new_papers_by_topic[name2]))) 
            
            print(name2, "Maritime intersect", len(sets_intersect_maritime[name1].intersection(sets_intersect_maritime[name2])))
            print(name2, "Maritime union", len(sets_intersect_maritime[name1].union(sets_intersect_maritime[name2])))
	 
            print(name2, "Anomaly intersect", len(sets_intersect_anomaly[name1].intersection(sets_intersect_anomaly[name2])))
            print(name2, "Anomaly union", len(sets_intersect_anomaly[name1].union(sets_intersect_anomaly[name2])))
            
            print(name2, "Detect intersect", len(sets_intersect_detect[name1].intersection(sets_intersect_detect[name2])))
            print(name2, "Detect union", len(sets_intersect_detect[name1].union(sets_intersect_detect[name2])))
            
            print(name2, "Maritime Anomaly intersect", len(sets_intersect_maritime_anomaly[name1].intersection(sets_intersect_maritime_anomaly[name2])))
            print(name2, "Maritime Anomaly union", len(sets_intersect_maritime_anomaly[name1].union(sets_intersect_maritime_anomaly[name2])))
	 
            print(name2, "Maritime Detect intersect", len(sets_intersect_maritime_detect[name1].intersection(sets_intersect_maritime_detect[name2]))) 
            print(name2, "Maritime Detect union", len(sets_intersect_maritime_detect[name1].union(sets_intersect_maritime_detect[name2]))) 
            
            print(name2, "Anomaly Detect intersect", len(sets_intersect_anomaly_detect[name1].intersection(sets_intersect_anomaly_detect[name2])))
            print(name2, "Anomaly Detect union", len(sets_intersect_anomaly_detect[name1].union(sets_intersect_anomaly_detect[name2])))
	 
            print(name2, "Maritime Anomaly Detect intersect", len(sets_intersect_maritime_anomaly_detect[name1].intersection(sets_intersect_maritime_anomaly_detect[name2])))
            print(name2, "Maritime Anomaly Detect union", len(sets_intersect_maritime_anomaly_detect[name1].union(sets_intersect_maritime_anomaly_detect[name2])))
	 
sets_cluster = ["DBSCAN_keywords", "kmeans_keywords", "GMM_keywords", "KDE_keywords", "OU_keywords"]
#process_multi_sets(sets_cluster)
sets_img = ["AIS_keywords", "radar_keywords", "camera_keywords"]
#process_multi_sets(sets_img)
sets_ML = ["CNN_keywords", "RNN_keywords", "VRNN_keywords", "NN_keywords", "learning_keywords", "artificial_keywords", "lstm_keywords"]
#process_multi_sets(sets_ML)
 
def process_multi_sets_all(set_to_use, limit): 
    list_keys = list(set_to_use.keys())
    for depth1 in range(len(list_keys)):
        name1 = list_keys[depth1]
        for depth2 in range(depth1 + 1, len(list_keys)):
            name2 = list_keys[depth2]
            if len(set_to_use[name1].intersection(set_to_use[name2])) >= len(set_to_use[name1]) * limit:
                print(name1, "Size", len(set_to_use[name1]), name2, "Size", len(set_to_use[name2])) 
                print(name1, name2, "intersect", len(set_to_use[name1].intersection(set_to_use[name2]))) 
                print(name1, name2, "union", len(set_to_use[name1].union(set_to_use[name2]))) 
                
process_multi_sets_all(sets_intersect_maritime_anomaly_detect, 0)

def make_new_dict_subset(names, dict_to_use):
    new_dict_ret = dict()
    for name in names:
        if name in dict_to_use:
            new_dict_ret[name] = dict()
            for paper in dict_to_use[name]:
                year_of_paper = title_year_one[paper]
                if int(year_of_paper) not in new_dict_ret[name]:
                    new_dict_ret[name][int(year_of_paper)] = 0
                new_dict_ret[name][int(year_of_paper)] += 1 
    return new_dict_ret
             
'''
make_a_plot(make_new_dict_subset(sets_cluster, sets_intersect_maritime_anomaly_detect), "cluster_time.png")   
make_a_plot(make_new_dict_subset(sets_img, sets_intersect_maritime_anomaly_detect), "cluster_img.png")  
make_a_plot(make_new_dict_subset(sets_ML, sets_intersect_maritime_anomaly_detect), "cluster_ML.png")    
set_to_compare = maritime_keywords.intersection(anomaly_keywords).intersection(detect_or_analyze)

matplotlib_venn.venn3([ais_keywords.intersection(maritime_keywords), radar_keywords.intersection(maritime_keywords), camera_keywords.intersection(maritime_keywords)], set_labels = ('AIS', 'radar', "Camera"))
plt.savefig("tmp_venn.png")

all_nn = nn_keywords.union(cnn_keywords).union(rnn_keywords).union(vrnn_keywords).union(lstm_keywords).intersection(maritime_keywords)
all_clustering = cluster_keywords.union(DBSCAN_keywords).union(kmeans_keywords).union(GMM_keywords).union(kde_keywords).union(ou_keywords).intersection(maritime_keywords)
all_ml = learning_keywords.union(all_nn).union(all_clustering).intersection(maritime_keywords)
all_ai = artificial_keywords.intersection(maritime_keywords).union(all_ml)
plt.close()
matplotlib_venn.venn3([all_nn, all_clustering, all_ai], set_labels = ('NN', 'Clustering', "AI"))
plt.savefig("tmp_venn2.png")
'''
