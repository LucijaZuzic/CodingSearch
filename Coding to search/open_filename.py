import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib_venn  
import urllib.request 

df_ieee = pd.read_csv("export2023_2.csv")
df_scopus = pd.read_csv("scopus.csv")
df_scopus2 = pd.read_csv("../SCOPUS_LATEST/scopus.csv") 
df_sd = pd.read_csv("ScienceDirect.csv")

def find_dois(all_names):
    all_dois = dict()
    for name in all_names:
        all_dois[name] = set()
    
    for i in range(len(df_ieee["Document Title"])): 
        tmp_name = str(df_ieee["Document Title"][i]).lower()
        tmp_doi = str(df_ieee["DOI"][i]).lower()
        if tmp_name in all_names and tmp_doi != 'nan':
            all_dois[tmp_name].add(tmp_doi) 
        
    for i in range(len(df_scopus["Title"])): 
        tmp_name = str(df_scopus["Title"][i]).lower()
        tmp_doi = str(df_scopus["DOI"][i]).lower() 
        if tmp_name in all_names and tmp_doi != 'nan':
            all_dois[tmp_name].add(tmp_doi) 
    for i in range(len(df_scopus2["Title"])): 
        tmp_name = str(df_scopus2["Title"][i]).lower()
        tmp_doi = str(df_scopus2["DOI"][i]).lower()
        if tmp_name in all_names and tmp_doi != 'nan':
            all_dois[tmp_name].add(tmp_doi) 
        
    for i in range(len(df_sd["title"])): 
        tmp_name = str(df_sd["title"][i]).lower()
        tmp_doi = str(df_sd["doi"][i]).replace("https://doi.org/", "").lower() 
        if tmp_name in all_names and tmp_doi != 'nan':
            all_dois[tmp_name].add(tmp_doi) 
        
    return all_dois
            
def make_venn2(sets, set_names):
    for a in range(len(sets)): 
        for b in range(a + 1, len(sets)):
            if len(sets[a].intersection(sets[b])) == 0:
                continue
            if len(sets[a].difference(sets[b])) == 0:
                continue
            if len(sets[b].difference(sets[a])) == 0:
                continue
            matplotlib_venn.venn2([sets[a], sets[b]], set_labels = (set_names[a], set_names[b]))
            plt.show()
            plt.close()
                
def make_venn3(sets, set_names):
    for a in range(len(sets)): 
        for b in range(a + 1, len(sets)): 
            if len(sets[a].intersection(sets[b])) == 0:
                continue
            if len(sets[a].difference(sets[b])) == 0:
                continue
            if len(sets[b].difference(sets[a])) == 0:
                continue
            for c in range(b + 1, len(sets)): 
                if len(sets[a].intersection(sets[c])) == 0:
                    continue
                if len(sets[a].difference(sets[c])) == 0:
                    continue
                if len(sets[c].difference(sets[a])) == 0:
                    continue
                if len(sets[b].intersection(sets[c])) == 0:
                    continue
                if len(sets[b].difference(sets[c])) == 0:
                    continue
                if len(sets[c].difference(sets[b])) == 0:
                    continue
                if len(sets[a].intersection(sets[b]).intersection(sets[c])) == 0:
                    continue
                if len(sets[a].difference(sets[b]).difference(sets[c])) == 0:
                    continue
                if len(sets[b].difference(sets[a]).difference(sets[c])) == 0:
                    continue
                if len(sets[c].difference(sets[a]).difference(sets[b])) == 0:
                    continue
                if len(sets[a].intersection(sets[b]).difference(sets[c])) == 0:
                    continue
                if len(sets[a].intersection(sets[c]).difference(sets[b])) == 0:
                    continue
                if len(sets[c].intersection(sets[b]).difference(sets[a])) == 0:
                    continue
                matplotlib_venn.venn3([sets[a], sets[b], sets[c]], set_labels = (set_names[a], set_names[b], set_names[c]))
                plt.show()
                plt.close()

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
    #plt.savefig(name_plot) 
    plt.show()
    plt.close()

def read_all_csvs(extension):
    save_dict = dict()
    my_files = os.listdir("keywords/")  
    for filename_mine in my_files:
        if extension in filename_mine: 
            tmp_csv = pd.read_csv("keywords/" + filename_mine)
            tmp_dict = dict()
            for i in range(len(tmp_csv["name"])):
                tmp_name = tmp_csv["name"][i]
                tmp_year = tmp_csv["year"][i]
                tmp_abstract = tmp_csv["abstract"][i]
                tmp_kw = tmp_csv["keywords"][i]
                tmp_dict[tmp_name] = {"year": tmp_year, "abstract": tmp_abstract, "keywords": tmp_kw}
            save_dict[filename_mine.replace(extension, "")] = tmp_dict
    return save_dict

def return_names(df_dict, kw_categories): 
    names_sets = list()
    for kw_category in kw_categories:
        names_sets.append(set())
        for name_one in df_dict[kw_category]:
            names_sets[-1].add(name_one)
    return names_sets

def return_years(df_dict, kw_categories): 
    years_dicts = dict()
    for kw_category in kw_categories:
        years_dicts[kw_category] = dict()
        for name_one in df_dict[kw_category]:
            tmp_year = int(df_dict[kw_category][name_one]["year"])
            if tmp_year not in years_dicts[kw_category]:
                years_dicts[kw_category][tmp_year] = 0
            years_dicts[kw_category][tmp_year] += 1
    return years_dicts
     
df_dict_none = read_all_csvs("_keywords.csv")
df_dict_maritime = read_all_csvs("_keywords_maritime.csv")
df_dict_anomaly = read_all_csvs("_keywords_anomaly.csv")
df_dict_detection = read_all_csvs("_keywords_detect.csv")
df_dict_maritime_anomaly = read_all_csvs("_keywords_maritime_anomaly.csv")
df_dict_maritime_detection = read_all_csvs("_keywords_maritime_detect.csv")
df_dict_anomaly_detection = read_all_csvs("_keywords_anomaly_detect.csv")
df_dict_maritime_anomaly_detection = read_all_csvs("_keywords_maritime_anomaly_detect.csv")

sets_cluster = ["DBSCAN", "kmeans", "GMM", "KDE", "OU"] 
sets_img = ["AIS", "radar", "camera"] 
sets_ML = ["CNN", "RNN", "VRNN", "NN", "learning", "artificial", "LSTM"] 
make_a_plot(return_years(df_dict_none, sets_ML), "ML_year_plot")
#make_venn2(return_names(df_dict_maritime_anomadf_dict_nonely_detection, sets_cluster), sets_cluster)
#make_venn3(return_names(df_dict_none, sets_img), sets_img)

all_keys = list(df_dict_none.keys())
#make_venn2(return_names(df_dict_none, all_keys), all_keys)
#make_venn3(return_names(df_dict_none, all_keys), all_keys)

for keyword_name in df_dict_none:
    print(keyword_name, len(return_names(df_dict_none, [keyword_name])[0]))

dict_all_names = dict()
for keyword_name in df_dict_none: 
    tmp_set_names = return_names(df_dict_none, [keyword_name])[0]
    tmp_dict_names_dois = find_dois(tmp_set_names)
    for name in tmp_dict_names_dois:
    	dict_all_names[name] = tmp_dict_names_dois[name] 
print(len(dict_all_names)) 
for name in dict_all_names: 
    if len(dict_all_names[name]) > 1:
        print(name, dict_all_names[name])

if not os.path.isdir("DOI"):
    os.mkdir("DOI")

merge_all_dois = dict() 
names_seen = set() 
all_lines_bib = ""
for name in dict_all_names: 
    for one_doi in dict_all_names[name]:  
        url = "http://dx.doi.org/" + one_doi
        req = urllib.request.Request(url, headers={"Accept": "text/bibliography; style=bibtex"})
	
        try:
            with urllib.request.urlopen(req) as response:
                for line in response:
                    new_line = line.decode()
                    
                    author_id = "author={"
                    pos_of_author = new_line.find(author_id)
                    end_of_author = new_line.find(",", pos_of_author)
                    author_name = new_line[pos_of_author + len(author_id):end_of_author].replace(" and ", "").replace(" ", "")
                    print("AUTHOR", author_name)
                    
                    pos_of_reference = new_line.find("{")
                    end_of_reference = new_line.find(",")
                    new_reference = new_line[pos_of_reference + 1:end_of_reference] + author_name 
                    print("REFERENCE", new_reference)
                    
                    count = 0
                    for old_name in names_seen:
                        if new_reference in old_name:
                            count += 1
                    print("COUNT", count)
                            
                    if count == 0: 
                        new_line = new_line[:end_of_reference] + author_name + new_line[end_of_reference:] 
                        open_save = open("DOI/" + new_reference + ".bib", "w", encoding = "utf-8")
                        open_save.write(new_line)
                        open_save.close()
                        merge_all_dois[one_doi] = new_reference
                        names_seen.add(new_reference)
                        all_lines_bib += new_line + "\n"
                    else:
                        new_line = new_line[:end_of_reference] + author_name + str(count) + new_line[end_of_reference:] 
                        open_save = open("DOI/" + new_reference + str(count) + ".bib", "w", encoding = "utf-8")
                        open_save.write(new_line)
                        open_save.close()
                        merge_all_dois[one_doi] = new_reference + str(count)
                        names_seen.add(new_reference + str(count))
                        all_lines_bib += new_line + "\n"
                    	
                    #print(new_line) 
        except:
            merge_all_dois[one_doi] = ""
            pass

open_save_bib = open("DOI/all_lines_bib.bib", "w", encoding = "utf-8")
open_save_bib.write(all_lines_bib)
open_save_bib.close()

save_new_doi = '"name","doi","file"\n'

for name in dict_all_names: 
    for one_doi in dict_all_names[name]:  
        save_new_doi += '"' + name + '",'
        save_new_doi += '"' + one_doi + '",' 
        save_new_doi += '"' + merge_all_dois[one_doi] + '"\n' 
        
open_save = open("DOI/doi_dict.csv", "w", encoding = "utf-8")
open_save.write(save_new_doi)
open_save.close()
