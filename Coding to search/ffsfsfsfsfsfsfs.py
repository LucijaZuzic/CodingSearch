import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib_venn 
import seaborn as sns
import numpy as np

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
    print(sd_names[i])
sd_doi = list(df_sd["doi"])
for i in range(len(sd_doi)):
    sd_doi[i] = str(sd_doi[i]).lower().replace("https://doi.org/", "").replace("nan", "")
print(len(sd_doi)) 
    
def find_names_for_doi(original_name, doi, names1, names2, names3, doi1, doi2, doi3):
    if len(doi) == 0:
        return original_name
    names = original_name
    for i in range(len(names1)):
        if doi1[i] == doi and len(names1[i]) != 0:
            names = names1[i]
    for i in range(len(names2)):
        if doi2[i] == doi and len(names2[i]) != 0:
            names = names2[i]
    for i in range(len(names3)):
        if doi3[i] == doi and len(names3[i]) != 0:
            names = names3[i]
    return names 

def find_dois_for_name(original_doi, name, names1, names2, names3, doi1, doi2, doi3):
    if len(name) == 0:
        return original_doi
    dois = original_doi
    for i in range(len(names1)):
        if names1[i] == name and len(doi1[i]) != 0:
            dois = doi1[i]
    for i in range(len(names2)):
        if names2[i] == name and len(doi2[i]) != 0:
            dois = doi2[i]
    for i in range(len(names3)):
        if names3[i] == name and len(doi3[i]) != 0:
            dois = doi3[i]
    return dois

print("fix IEEE")

for i in range(len(ieee_names)):
    new_doi = find_dois_for_name(ieee_doi[i], ieee_names[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    if ieee_doi[i] != new_doi:
        print(ieee_doi[i], new_doi, ieee_names[i])
    ieee_doi[i] = new_doi
    new_name = find_names_for_doi(ieee_names[i], ieee_doi[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    if ieee_names[i] != new_name:
        print(ieee_names[i], new_name, ieee_doi[i])
    ieee_names[i] = new_name 
    
print("fix SC")

for i in range(len(scopus_names)):
    new_doi = find_dois_for_name(scopus_doi[i], scopus_names[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    if scopus_doi[i] != new_doi:
        print(scopus_doi[i], new_doi, scopus_names[i])
    scopus_doi[i] = new_doi
    new_name = find_names_for_doi(scopus_names[i], scopus_doi[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    if scopus_names[i] != new_name:
        print(scopus_names[i], new_name, scopus_doi[i])
    scopus_names[i] = new_name 

print("fix SD")

for i in range(len(sd_names)):
    if len(sd_doi[i]) == 0:
        new_doi = find_dois_for_name(sd_doi[i], sd_names[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
        if sd_doi[i] != new_doi:
            print(sd_doi[i], new_doi, sd_names[i])
        sd_doi[i] = new_doi
    new_name = find_names_for_doi(sd_names[i], sd_doi[i], ieee_names, scopus_names, sd_names, ieee_doi, scopus_doi, sd_doi)
    if sd_names[i] != new_name:
        print(sd_names[i], new_name, sd_doi[i])
    sd_names[i] = new_name 