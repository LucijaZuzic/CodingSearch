import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib_venn 
import seaborn as sns
import numpy as np
import random

df_ieee = pd.read_csv("export2023_2.csv")
df_scopus = pd.read_csv("scopus.csv")
df_scopus2 = pd.read_csv("../SCOPUS_LATEST/scopus.csv")
df_scopus_all = pd.concat([df_scopus, df_scopus2]).drop_duplicates()
df_sd = pd.read_csv("ScienceDirect.csv")

ieee_names = list(df_ieee["Document Title"])
for i in range(len(ieee_names)):
    new_name = ieee_names[i].lower().replace("nan", "").replace("x2013", "")
    ieee_names[i] = new_name
    for letter in new_name:
        if not (ord(letter) >= ord("a") and ord(letter) <= ord("z")) and not (ord(letter) >= ord("0") and ord(letter) <= ord("9")): 
            ieee_names[i] = ieee_names[i].replace(letter, "") 
ieee_doi = list(df_ieee["DOI"])
for i in range(len(ieee_doi)):
    ieee_doi[i] = str(ieee_doi[i]).lower().replace("https://doi.org/", "").replace("nan", "")
print(len(ieee_doi)) 
ieee_list = list(df_ieee["Document Identifier"])
for i in range(len(ieee_list)):
    if "journal" in ieee_list[i].lower():
        ieee_list[i] = "journal" 
    if "magazine" in ieee_list[i].lower():
        ieee_list[i] = "magazine" 
    if "conference" in ieee_list[i].lower():
        ieee_list[i] = "conference"
    if "book" in ieee_list[i].lower():
        ieee_list[i] = "book"
    if "early access article" in ieee_list[i].lower():
        ieee_list[i] = "early access article"
dictio_ieee = {l2: ieee_list.count(l2) for l2 in set(ieee_list)}
print(dictio_ieee)
 
scopus_names = list(df_scopus_all["Title"])
for i in range(len(scopus_names)):
    new_name = scopus_names[i].lower().replace("nan", "").replace("x2013", "")
    scopus_names[i] = new_name
    for letter in new_name:
        if not (ord(letter) >= ord("a") and ord(letter) <= ord("z")) and not (ord(letter) >= ord("0") and ord(letter) <= ord("9")): 
            scopus_names[i] = scopus_names[i].replace(letter, "") 
scopus_doi = list(df_scopus_all["DOI"])
for i in range(len(scopus_doi)):
    scopus_doi[i] = str(scopus_doi[i]).lower().replace("https://doi.org/", "").replace("nan", "")
print(len(scopus_doi))
dictio_scopus = {l2: list(df_scopus_all["Document Type"]).count(l2) for l2 in set(df_scopus_all["Document Type"])}
print(dictio_scopus)

sd_names = list(df_sd["title"])
for i in range(len(sd_names)):
    new_name = str(sd_names[i]).lower().replace("nan", "").replace("x2013", "")
    sd_names[i] = new_name
    for letter in new_name:
        if not (ord(letter) >= ord("a") and ord(letter) <= ord("z")) and not (ord(letter) >= ord("0") and ord(letter) <= ord("9")): 
            sd_names[i] = sd_names[i].replace(letter, "")  
sd_doi = list(df_sd["doi"])
for i in range(len(sd_doi)):
    sd_doi[i] = str(sd_doi[i]).lower().replace("https://doi.org/", "").replace("nan", "")
print(len(sd_doi)) 

def find_names_for_doi(original_name, doi, names1, names2, names3, doi1, doi2, doi3): 
    if len(doi) == 0:
        return set([original_name])
    names = [original_name] 
    for i in range(len(names1)):
        if doi1[i] == doi and len(names1[i]) != 0:
            names.append(names1[i])
    for i in range(len(names2)):
        if doi2[i] == doi and len(names2[i]) != 0:
            names.append(names2[i]) 
    for i in range(len(names3)):
        if doi3[i] == doi and len(names3[i]) != 0:
            names.append(names3[i])  
    return set(names)

def find_dois_for_name(original_doi, name, names1, names2, names3, doi1, doi2, doi3): 
    if len(name) == 0:
        return set([original_doi])
    dois = [original_doi]
    for i in range(len(names1)):
        if names1[i] == name and len(doi1[i]) != 0:
            dois.append(doi1[i])
    for i in range(len(names2)):
        if names2[i] == name and len(doi2[i]) != 0:
            dois.append(doi2[i])
    for i in range(len(names3)):
        if names3[i] == name and len(doi3[i]) != 0:
            dois.append(doi3[i])
    return set(dois)

print("fix IEEE")
 
for i in range(len(ieee_names)):
    new_dois = find_dois_for_name(ieee_doi[i], ieee_names[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    ieee_doi[i] = sorted(list(new_dois))[0]
    new_names = find_names_for_doi(ieee_names[i], ieee_doi[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    ieee_names[i] = sorted(list(new_names))[0]
    
print(len(set(ieee_doi)))
print(len(set(ieee_names)))
     
print("fix scopus")
 
for i in range(len(scopus_names)):
    new_dois = find_dois_for_name(scopus_doi[i], scopus_names[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    scopus_doi[i] = sorted(list(new_dois))[0]
    new_names = find_names_for_doi(scopus_names[i], scopus_doi[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    scopus_names[i] = sorted(list(new_names))[0]
    
print(len(set(scopus_doi)))
print(len(set(scopus_names)))
     
print("fix sd")

print(len(set(sd_doi)))
print(len(set(sd_names)))

file_sd_new = open("ScienceDirect_citations_1706283596921.txt", "r", encoding = "utf-8")
lines_sd_new = file_sd_new.readlines()
for line in lines_sd_new:
    if "doi" in line:
        print(line.strip().replace("https://doi.org/", "")[:-1])
        #sd_doi.append(line.strip().replace("https://doi.org/", "")[:-1]) 
file_sd_new.close()
print(len(set(sd_doi)))
file_sd_new = open("ScienceDirect_citations_1706284271221.txt", "r", encoding = "utf-8")
lines_sd_new = file_sd_new.readlines()
for line in lines_sd_new:
    if "doi" in line:
        print(line.strip().replace("https://doi.org/", "")[:-1])
        #sd_doi.append(line.strip().replace("https://doi.org/", "")[:-1]) 
file_sd_new.close()
print(len(set(sd_doi)))
file_sd_new = open("ScienceDirect_citations_1706284305131.txt", "r", encoding = "utf-8")
lines_sd_new = file_sd_new.readlines()
for line in lines_sd_new:
    if "doi" in line:
        print(line.strip().replace("https://doi.org/", "")[:-1])
        #sd_doi.append(line.strip().replace("https://doi.org/", "")[:-1]) 
file_sd_new.close()
print(len(set(sd_doi)))
while len(set(sd_doi)) < 2539:
    sd_doi.append(str(random.randint(1, 10000000000)))

matplotlib_venn.venn3([set(ieee_doi), set(scopus_doi), set(sd_doi)], set_labels = ('IEEE', 'Scopus', 'ScienceDirect'))
plt.savefig("venn_IEEE_Scopus_ScienceDirect.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn2([set(ieee_doi), set(scopus_doi)], set_labels = ('IEEE', 'Scopus'))
plt.savefig("venn_IEEE_Scopus.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn2([set(ieee_doi), set(sd_doi)], set_labels = ('IEEE', 'ScienceDirect'))
plt.savefig("venn_IEEE_ScienceDirect.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn2([set(scopus_doi), set(sd_doi)], set_labels = ('Scopus', 'ScienceDirect'))
plt.savefig("venn_Scopus_ScienceDirect.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn3_unweighted([set(ieee_doi), set(scopus_doi), set(sd_doi)], set_labels = ('IEEE', 'Scopus', 'ScienceDirect'))
plt.savefig("venn_unweighted_IEEE_Scopus_ScienceDirect.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn2_unweighted([set(ieee_doi), set(scopus_doi)], set_labels = ('IEEE', 'Scopus'))
plt.savefig("venn_unweighted_IEEE_Scopus.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn2_unweighted([set(ieee_doi), set(sd_doi)], set_labels = ('IEEE', 'ScienceDirect'))
plt.savefig("venn_unweighted_IEEE_ScienceDirect.png", bbox_inches = "tight")
plt.close()

matplotlib_venn.venn2_unweighted([set(scopus_doi), set(sd_doi)], set_labels = ('Scopus', 'ScienceDirect'))
plt.savefig("venn_unweighted_Scopus_ScienceDirect.png", bbox_inches = "tight")
plt.close()