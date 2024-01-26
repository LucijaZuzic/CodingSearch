import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib_venn 
import seaborn as sns
import numpy as np

df_ieee = pd.read_csv("export2023_2.csv")
df_scopus = pd.read_csv("scopus.csv")
df_scopus2 = pd.read_csv("../SCOPUS_LATEST/scopus.csv") 
df_sd = pd.read_csv("ScienceDirect.csv")

def find_ref_file(all_names):
    df_ref_file = pd.read_csv("DOI/doi_dict.csv")
    all_file_extensions = dict()
    for i in range(len(df_ref_file["name"])): 
        tmp_name = str(df_ref_file["name"][i])
        tmp_doi = str(df_ref_file["doi"][i])
        tmp_file = str(df_ref_file["file"][i])
        if tmp_name in all_names:
            if not os.path.isfile("DOI/" + tmp_file + ".bib"):
                all_file_extensions[tmp_name] = ""
                continue
            a_file = open("DOI/" + tmp_file + ".bib", "r", encoding = "UTF-8")
            file_lines = a_file.readlines()
            content = ""
            for line in file_lines: 
                content += line.replace("\n", "")
            a_file.close()
            all_file_extensions[tmp_name] = content
    all_file_extensions_list = []
    for tmp_name in all_names:
        if tmp_name in all_file_extensions:
            all_file_extensions_list.append(all_file_extensions[tmp_name])
        else:
            all_file_extensions_list.append("")
    return all_file_extensions_list
        
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
            
def find_authors(all_names): 
    all_author = dict()
    for name in all_names:
        all_author[name] = set()
    
    for i in range(len(df_ieee["Document Title"])): 
        tmp_name = str(df_ieee["Document Title"][i]).lower()
        tmp_author = str(df_ieee["Authors"][i])
        if tmp_name in all_names:
            all_author[tmp_name].add(tmp_author) 
        
    for i in range(len(df_scopus["Title"])): 
        tmp_name = str(df_scopus["Title"][i]).lower()
        tmp_author = str(df_scopus["Authors"][i])
        if tmp_name in all_names:
            all_author[tmp_name].add(tmp_author) 
    for i in range(len(df_scopus2["Title"])): 
        tmp_name = str(df_scopus2["Title"][i]).lower()
        tmp_author = str(df_scopus2["Authors"][i])
        if tmp_name in all_names:
            all_author[tmp_name].add(tmp_author) 
        
    for i in range(len(df_sd["title"])): 
        tmp_name = str(df_sd["title"][i]).lower()
        tmp_author = str(df_sd["author"][i])
        if tmp_name in all_names:
            all_author[tmp_name].add(tmp_author) 
        
    return all_author
    
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
    miny = 0
    maxy = 0
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue
        minx = min(min(X_axis), minx)
        maxx = max(max(X_axis), maxx)
        Y_axis = list(dict_multi[dict_one_name].values())
        miny = min(min(Y_axis), miny)
        maxy = max(max(Y_axis), maxy)
    plt.vlines(np.arange(minx - 0.5, maxx + 1.5), miny, maxy, linestyle = 'dashed', color = '#c0c0c0')
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue
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

def return_years(df_dict, kw_categories, min_year = 0, max_year = 10000): 
    years_dicts = dict()
    for kw_category in kw_categories:
        years_dicts[kw_category] = dict()
        for name_one in df_dict[kw_category]:
            tmp_year = int(df_dict[kw_category][name_one]["year"])
            if tmp_year < min_year or tmp_year > max_year:
                continue
            if tmp_year not in years_dicts[kw_category]:
                years_dicts[kw_category][tmp_year] = 0
            years_dicts[kw_category][tmp_year] += 1
    return years_dicts

def return_intersection_data_frame(sets1, set_names1, sets2, set_names2):
    data_matrix = []
    ylabs = []
    for a in range(len(sets1)):        
        if len(sets1[a]) == 0:
            continue
        ylabs.append(set_names1[a])
        data_matrix.append([])
        xlabs = []
        for b in range(len(sets2)): 
            if len(sets2[b]) == 0:
                continue
            xlabs.append(set_names2[b])
            data_matrix[-1].append(len(sets1[a].intersection(sets2[b])))
    sns.heatmap(data_matrix, annot = True, xticklabels = xlabs, yticklabels = ylabs, fmt = 'g')
    plt.show() 
    plt.close()
     
def return_category(df_dict, kw_categories): 
    category_for_name = set() 
    for kw_category in kw_categories: 
        for name_one in df_dict[kw_category]:
            category_for_name.add(name_one)
    data_heatmap = []
    dict_heatmap = dict()
    for name_one in category_for_name:  
        row_heatmap = []
        row_heatmap_str = ""
        for kw_category in kw_categories:
            if name_one in df_dict[kw_category]:
                row_heatmap.append(1)
                row_heatmap_str += "1"
            else:
                row_heatmap.append(0)
                row_heatmap_str += "0"
        if row_heatmap_str not in dict_heatmap:
                dict_heatmap[row_heatmap_str] = set()
                data_heatmap.append(row_heatmap)
        dict_heatmap[row_heatmap_str].add(name_one) 
    labels_heatmap = []
    labels_bar_plot = []
    for row_num in range(len(data_heatmap)): 
        row_heatmap_str = ""
        row_barplot_str = ""
        for col_num in range(len(data_heatmap[row_num])): 
            if data_heatmap[row_num][col_num] == 1: 
                row_heatmap_str += "1"
                if row_barplot_str != "":
                    row_barplot_str += " + "
                row_barplot_str += kw_categories[col_num]
            else: 
                row_heatmap_str += "0"
        labels_heatmap.append(len(dict_heatmap[row_heatmap_str]))
        labels_bar_plot.append(row_barplot_str)
        
    sns.heatmap(data_heatmap, yticklabels = labels_heatmap, xticklabels = kw_categories, cbar = False)
    plt.show() 
    plt.close()
    
    plt.bar(range(len(labels_bar_plot)), labels_heatmap)
    plt.xticks(range(len(labels_bar_plot)), labels_bar_plot)
    plt.gca().set_xticklabels(labels_bar_plot, rotation = (45), va = 'top', ha = 'right')
    plt.show() 
    plt.close()
     
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
sets_ML = ["CNN", "RNN", "VRNN", "NN", "learning", "artificial", "LSTM", "ML", "DL"]  

sets_technique = ["learning", "ML", "artificial", "kalman", "gauss", "bayes", "autoencoder"]
sets_algorithm = ["HMM", "SVR", "SVM", "KNN", "GAM", "RF"]
sets_cluster = ["cluster", "DBSCAN", "kmeans", "GMM", "KDE", "OU"] 
sets_DL = ["DL", "NN", "CNN", "RNN", "VRNN", "LSTM"]
merge_kws = sets_technique + sets_algorithm + sets_cluster + sets_DL
return_category(df_dict_maritime_anomaly_detection, sets_DL)
sets_data = ["AIS", "visual", "camera", "radar", "SAR", "sonar"]
sets_variables = ["location", "speed", "heading", "course", "trajectory"]
sets_general = ["maritime", "anomaly", "detect", "analyze", "behaviour", "communication", "data_mining"]

dict_all_names = dict()
for keyword_name in df_dict_maritime_anomaly_detection: 
    tmp_set_names = return_names(df_dict_maritime_anomaly_detection, [keyword_name])[0]
    tmp_dict_names_dois = find_dois(tmp_set_names)
    for name in tmp_dict_names_dois:
    	dict_all_names[name] = tmp_dict_names_dois[name] 
print(len(dict_all_names)) 
for name in dict_all_names: 
    if len(dict_all_names[name]) > 1:
        print(name, dict_all_names[name])
        
file_bib = open("allbib.bib", "r", encoding="UTF-8")
all_lines_bib = file_bib.readlines()
file_bib.close()
all_lines_str = ""
for line in all_lines_bib:
    all_lines_str += line
 
new_refs_str = ""
num_new = 0
num_undef = 0
for keyword_name in df_dict_maritime_anomaly_detection: 
    names_some = return_names(df_dict_maritime_anomaly_detection, [keyword_name])[0]
    tmp_set_names = find_authors(names_some)
    tmp_set_refs = find_ref_file(names_some)
    tmp_set_find_dois = find_dois(names_some)
    new_tmp_set_names = dict()
    new_tmp_set_refs = []
    for num_paper in range(len(list(tmp_set_names.keys()))):
        name_paper = list(tmp_set_names.keys())[num_paper]
        ref_paper = tmp_set_refs[num_paper]
        dois_paper = tmp_set_find_dois[name_paper]
        if name_paper.lower() not in all_lines_str.lower():
            found_doi = False
            for doi in dois_paper:
                if doi.lower() in all_lines_str.lower():
                    found_doi = True
            if not found_doi:
                new_tmp_set_names[name_paper] = tmp_set_names[name_paper]
                new_tmp_set_refs.append(ref_paper)
    if len(new_tmp_set_names) < 20 and len(new_tmp_set_names) > 0 and keyword_name in merge_kws:
        new_refs_str += "#" + " " + keyword_name + " " + str(len(new_tmp_set_names)) + "\n"
        print("#", keyword_name, len(new_tmp_set_names))
        for x in range(len(new_tmp_set_refs)): 
            key = list(new_tmp_set_names.keys())[x]
            if new_tmp_set_refs[x] == "":
                add_names = ""
                for names_add in new_tmp_set_names[key]:
                    add_names += names_add + " "
                new_refs_str += "#" + " " + key + " " + add_names + "\n"
                print("#", key, add_names)  
                num_new += 1
                num_undef += 1
            else:
                find_start = new_tmp_set_refs[x].find("{") + 1
                find_end = new_tmp_set_refs[x].find(",")
                short_ref = new_tmp_set_refs[x][find_start:find_end]
                if short_ref not in new_refs_str:
                    new_refs_str += new_tmp_set_refs[x] + "\n"
                    print(new_tmp_set_refs[x])
                    num_new += 1
                else:
                    new_refs_str += "#" + " " + short_ref + "\n"
                    print("#", short_ref)
print(num_new, num_undef)

for keyword_name in df_dict_maritime_anomaly_detection: 
    names_some = return_names(df_dict_maritime_anomaly_detection, [keyword_name])[0]
    tmp_set_names = find_authors(names_some)
    tmp_set_refs = find_ref_file(names_some)
    tmp_set_find_dois = find_dois(names_some)
    new_tmp_set_names = dict()
    new_tmp_set_refs = []
    for num_paper in range(len(list(tmp_set_names.keys()))):
        name_paper = list(tmp_set_names.keys())[num_paper]
        ref_paper = tmp_set_refs[num_paper]
        dois_paper = tmp_set_find_dois[name_paper]
        if name_paper.lower() not in all_lines_str.lower():
            found_doi = False
            for doi in dois_paper:
                if doi.lower() in all_lines_str.lower():
                    found_doi = True
            if not found_doi:
                new_tmp_set_names[name_paper] = tmp_set_names[name_paper]
                new_tmp_set_refs.append(ref_paper)
    if len(new_tmp_set_names) < 100 and len(new_tmp_set_names) > 0 and keyword_name in merge_kws:
        print("#", keyword_name, len(new_tmp_set_names))  

#file_bib = open("allbib2.bib", "w", encoding="UTF-8")
#file_bib.write(new_refs_str)
#file_bib.close()

all_keys = list(df_dict_maritime_anomaly_detection.keys())
all_kws = return_names(df_dict_maritime_anomaly_detection, all_keys)        
num_sections = 4

for part_num1 in range(num_sections): 
    all_keys_div1 = all_keys[part_num1 * len(all_keys) // num_sections:(part_num1 + 1)* len(all_keys) // num_sections] 
    #make_a_plot(return_years(df_dict_maritime_anomaly_detection, all_keys_div1, 2000), "Plot " + str(part_num1))

#return_intersection_data_frame(all_kws, all_keys, all_kws, all_keys)
for part_num1 in range(num_sections): 
    all_keys_div1 = all_keys[part_num1 * len(all_keys) // num_sections:(part_num1 + 1)* len(all_keys) // num_sections]
    all_kws_div1 = all_kws[part_num1 * len(all_kws) // num_sections:(part_num1 + 1)* len(all_kws) // num_sections] 
    for part_num2 in range(part_num1, num_sections): 
        all_keys_div2 = all_keys[part_num2 * len(all_keys) // num_sections:(part_num2 + 1) * len(all_keys) // num_sections]
        all_kws_div2 = all_kws[part_num2 * len(all_kws) // num_sections:(part_num2 + 1) * len(all_kws) // num_sections]
        #return_intersection_data_frame(all_kws_div1, all_keys_div1, all_kws_div2, all_keys_div2)