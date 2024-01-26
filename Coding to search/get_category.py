import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

file_bib = open("allbibNEWEST.bib", "r", encoding="UTF-8")
all_lines_bib = file_bib.readlines()
file_bib.close()
  
all_entry = []
one_entry = ""
for line_num in range(len(all_lines_bib)):
    if "@" in all_lines_bib[line_num]:
        if len(one_entry.replace(" ", "")) != 0:
            all_entry.append(one_entry.replace("https://doi.org/", "").replace("\nand", " and"))
        one_entry = ""
    one_entry += all_lines_bib[line_num]

if len(one_entry.replace(" ", "")) != 0:
    all_entry.append(one_entry.replace("https://doi.org/", "").replace("\nand", " and"))
one_entry = ""

new_str = ""
new_str_author = "" 
entry_categories = dict() 
for entry in all_entry:  
        start_of_ref = entry.find("{")
        end_of_ref = entry.find(",")
        reference_original = entry[start_of_ref + 1:end_of_ref]  
        start_delimiter = "{"
        end_delimiter = "}"
        if entry.count("{") == 1 or entry.count("{") == 2:  
            start_delimiter = '"'
            end_delimiter = '"' 
        position_of_title = entry.lower().replace("booktitle", "bookttle").find("title")
        start_of_title = entry.find(start_delimiter, position_of_title) + 1
        end_of_title = entry.find(end_delimiter, start_of_title) 
        entry_categories[reference_original] = set()
print(len(all_entry))

papers = dict()
'''
papers["SVM"] = ["Mirghasemi2013APA", "2006Li", "2013Handayani", "Le2016TrajectoryPO", "MAZZARELLA2017110", "2019Fahn", "2017Sfyridis"]
papers["SVR"] = ["kim2019anomalous", "2016Kim"]
papers["OU"] = ["2016Millefiori", "2016Coraluppi",  "2018Coscia", "2018d'Afflisio1", "2019Forti", "2018Coscia1"]
papers["HMM"] = ["2018Toloue", "2014DuToit", "2014Shahir", "2004Franke", "2009Guo", "Auslander2012MaritimeTD"]
papers["DBSCAN"] = ["2015Cazzanti", "2015Sun", "zhao_shi_2019", "2018Coscia", "2018d'Afflisio1", "2018Lei", "2021Botts", "2021Prasad", "2021Pedroche", "2018Coscia1"]#, "2023Li"]
papers["GAM"] = ["2018Ford", "Ford2018LoiteringWI"]
papers["KNN"] = ["2018Virjonen", "2017Duca"]
papers["ANN"] = ["2014Mishra", "2020Singh"]
papers["RNN"] = ["2018Nguyen", "2022Nguyen", "2019Hoque", "zhao_shi_2019", "2020Nguyen"]
papers["BRNN"] = ["2020Xia"]
papers["LSTM"] = ["2019Hoque", "zhao_shi_2019"]
papers["VRNN"] = ["2020Nguyen"]
papers["CNN"] = ["2020Arasteh", "CHEN2020108182", "2022Djenouri", "2022Czaplewski", "2019Freitas"]#, "2022Velasco-Gallego2"]
papers["K-means"] = [ "2019Hanyang"]
papers["KDE"] = ["2013Pallotta"]
papers["GMM"] = ["2008Laxhammar"]
papers["BN"] = ["DABROWSKI2015116", "2014Castaldo", "2010Lane",   "2020Forti"]
papers["HBRS"] = ["2020Forti"]
papers["RF"] = ["Auslander2012MaritimeTD", "2022Szarmach"]

#papers["CRF"] = ["Auslander2012MaritimeTD"]
#papers["PAM"] = ["2017Mohammed"]
'''
excluded = ["GMM", "KDE", "K-means", "DBSCAN", "OU", "k-NN"]
papers["GMM"] = ["2015Anneken", "2008Riveiro1", "2008Laxhammar", "2009Laxhammar", "2002Kraiman", "DABROWSKI2015116", "2020Forti"]
papers["KDE"] = ["2015Anneken", "2009Laxhammar", "2013Pallotta", "2013Pallotta1", "2008Ristic", "2023Pohontu", "2020Loi"]
papers["K-means"] = [ "2019Hanyang"]
papers["DBSCAN"] = ["ZHANG2021107674", "2017Wang", "Li2017ADR",   
                    "2020Loi", "2017Fu", "2013Pallotta2", "2013Pallotta", 
                "2018d'Afflisio1",
                    "2013Pallotta1", "2014FernandezArguedas", 
                    "2014Pallotta", "2015Pallotta", "2019Shahir", 
                    "2018Lei", "2021Botts", "2021Prasad", "2021Pedroche",
                    "2008Laxhammar", "2019Forti", "2018Coscia1", "Varlamis2019ANA", 
                    "2015FernandezArguedas", "2018FernandezArguedas",
                    "2022Wei", "2014DuToit", "2015Cazzanti", "2015Sun", 
                    "zhao_shi_2019", "2018Coscia", "2023Li"]
papers["BN"] = ["DABROWSKI2015116", "2014Castaldo", "2010Lane",    "2012Ng", "handayani2015anomaly", "BOUEJLA201422"]
papers["autoencoder"] = ["2023Xie", "2020Liu", "2023Hu"]
papers["SVM"] = ["2013Handayani",   "Le2016TrajectoryPO", "2019Fahn", "MAZZARELLA2017110", "2017Sfyridis", "2015Shahir",  "2018Marzuki", "2018Dogru", "2022Wei" ]
papers["SVR"] = ["2015Kim", "kim2019anomalous", "2016Kim"] 
papers["RF"] = ["Woodill2020PredictingIF", "2020Singh1", "2022Szarmach", "2019Liang", "2018Marzuki", "2018Dogru"]
papers["OU"] = ["2018d'Afflisio1", "2016Millefiori", "2016Coraluppi",
                "2018Coscia1", "2020Forti", "2019Forti", "2018Forti",
                "2021d’Afflisio", "2021d’Afflisio1"] 
papers["GAM"] = ["2018Ford", "Ford2018LoiteringWI"]  
papers["HMM"] = ["2018Toloue", "2015Shahir", "Auslander2012MaritimeTD",
                 "2018Kroodsma", "2014DuToit", "2014Shahir"] 
papers["k-NN"] = ["2020Wang2", "2018Virjonen", "2017Duca", "2019Huang1"]
                #"2020Guo", "2019Huang1"]
papers["ANN"] = ["2020Singh"]

papers["RNN"] = ["2018Nguyen", "2022Nguyen", "2022Singh",
                 "2021Magnussen"]
papers["BRNN"] = ["2020Xia"]
papers["LSTM"] = ["2019Hoque", "zhao_shi_2019", "2019Liang", "2021Mehri",
                 "2021Karatas", "2023Mehri", "2023Wang"]
papers["VRNN"] = ["2020Nguyen", "2022Nguyen"]
papers["RNN"] = papers["RNN"] + papers["BRNN"] + papers["LSTM"] + papers["VRNN"]

papers["CNN"] = ["2020Arasteh", "CHEN2020108182", "2022Djenouri","2019Milios",
                 "2022Czaplewski", "2019Freitas", "2023Chen2"] 
papers["DL"] = ["2019Mantecon" ] 
papers["NN"] = ["2021Eljabu", "2022Liu5", "2018Dogru"]
 
for kw in papers:
    for paper_one in papers[kw]:
        entry_categories[paper_one].add(kw)

save_new_ref = '"reference","categories"\n'
 
has_category = 0
has_other = 0
for one_ref in entry_categories:   
    save_new_ref += '"' + one_ref + '","' 
    other_found = False
    for entry_category in entry_categories[one_ref]:
        save_new_ref += entry_category + ',' 
        if entry_category not in excluded:
            other_found = True
    if len(entry_categories[one_ref]) > 0:
        has_category += 1
        save_new_ref = save_new_ref[:-1]  
    if other_found:
        has_other += 1
    save_new_ref += '"\n' 
print(has_category)
print(has_other)

open_save = open("found_category.csv", "w", encoding = "utf-8")
open_save.write(save_new_ref)
open_save.close()

def list_to_set(list_old):
    make_set_new = set()
    for item_list_old in list_old:
        make_set_new.add(item_list_old)
    return make_set_new 
 
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

sets_all = []
set_names_all = []
sets_excluded = []
set_names_excluded = []
sets_not_excluded = []
set_names_not_excluded = []
for kw1 in papers:
    set1 = list_to_set(papers[kw1]) 
    sets_all.append(set1)
    set_names_all.append(kw1)
    if kw1 in excluded:
        sets_excluded.append(set1)
        set_names_excluded.append(kw1)
    else:
        sets_not_excluded.append(set1)
        set_names_not_excluded.append(kw1)
return_intersection_data_frame(sets_all, set_names_all, sets_all, set_names_all)
return_intersection_data_frame(sets_excluded, set_names_excluded, sets_not_excluded, set_names_not_excluded) 

def year_from_tag(tagged_reference):
    string_tag = tagged_reference.find("20")
    return(int(tagged_reference[string_tag:string_tag+4]))

def make_a_plot(dict_multi, name_plot, gray_line):
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
    if gray_line:
        plt.vlines(np.arange(minx - 0.5, maxx + 1.5), miny, maxy, linestyle = 'dashed', color = '#c0c0c0')
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue
        X_axis = [X_axis[i] + 1 / (len(dict_multi) + 1) * (counter_var - (len(dict_multi) - 1) / 2) for i in range(len(X_axis))]
        plt.bar(X_axis, list(dict_multi[dict_one_name].values()), 1 / (len(dict_multi) + 1), label = dict_one_name)
        counter_var += 1 
    plt.legend()
    plt.yticks(range(miny, maxy + 1), range(miny, maxy + 1))
    plt.xticks(range(minx, maxx + 1), range(minx, maxx + 1))
    plt.gca().set_xticklabels(range(minx, maxx + 1), rotation = (45), va = 'top', ha = 'right')
    #plt.savefig(name_plot) 
    plt.show()
    plt.close()

dict_years = dict()
for kw in papers:
    dict_years[kw] = dict()
    for paper in papers[kw]:
        year_for_paper = year_from_tag(paper)
        if year_for_paper not in dict_years[kw]:
            dict_years[kw][year_for_paper] = 0
        dict_years[kw][year_for_paper] += 1
excluded_dict_years = dict()
other_dict_years = dict()
for kw in dict_years:
    if kw in excluded:
        excluded_dict_years[kw] = dict_years[kw]
    else:
        other_dict_years[kw] = dict_years[kw]
        
dict_total = dict()
dict_total["Combined"] = dict()
for kw in dict_years:
    for year in dict_years[kw]:
        if year not in dict_total["Combined"]:
            dict_total["Combined"][year] = 0
        dict_total["Combined"][year] += dict_years[kw][year]
 
make_a_plot(dict_years, "NAME", True)
make_a_plot(excluded_dict_years, "NAME", True)
make_a_plot(other_dict_years, "NAME", True)
make_a_plot(dict_total, "NAME", False)

all_my_papers = set()
for kw in papers:
    for paper in papers[kw]:
    	all_my_papers.add(paper)
#print(all_my_papers)
    
papers_input = dict()
papers_input["AIS"] = ["2018Coscia", "2013Pallotta2", "2017Wang", "2020Xia", "2022Liu5", "Varlamis2019ANA", "Ford2018LoiteringWI", "2019Hanyang", "2020Nguyen", "2009Laxhammar", "2019Forti", "MAZZARELLA2017110", "ZHANG2021107674", "2017Sfyridis", "2023Hu",   "2015Shahir", "2022Nguyen", "2020Arasteh","2018Marzuki",
"2014DuToit", "2021Botts", "CHEN2020108182", "2018Coscia1", "2014Shahir", "2021Mehri", "2021Prasad", "2015Anneken", "2018Ford", "Li2017ADR", "2014Pallotta", "2015Pallotta",  "2023Chen2",  
"2008Ristic",  "2021Pedroche",  "2019Shahir", "2013Pallotta", "2019Liang", "2023Pohontu",  "2023Pohontu1","2019Mantecon", "2013Smith","2012Smith","2015Sun","2018Lei", "2021d’Afflisio", 
"2021d’Afflisio1", "2021Karatas", "2018Virjonen", "Woodill2020PredictingIF", "zhao_shi_2019", "2021Magnussen", "2018d'Afflisio1",
"2015FernandezArguedas", "2020Liu", "2023Xie", "2017Duca", "2022Szarmach", "2008Laxhammar", "2022Singh", "2014Castaldo","2019Milios",
"2014FernandezArguedas", "2019Hoque", "2013Handayani", "handayani2015anomaly", "2022Wei", "2018FernandezArguedas", "BOUEJLA201422",
"2020Singh1", "2023Li", "2020Singh", "2013Pallotta1", "2015Cazzanti", "2010Lane"
]
papers_input["SAR"] = ["2019Milios", "2012Vespe"]
papers_input["satellite"] = ["2019Young", "2022Verma", "2019Fahn"]
papers_input["ocean"] = ["DABROWSKI2015116", "ZHANG2021107674", "Woodill2020PredictingIF"]
papers_input["camera"] = ["2022Djenouri", "2022Czaplewski", "2019Freitas", "BOUEJLA201422"] 
papers_input["radar"] = ["2019Mantecon", "BOUEJLA201422", "2018d'Afflisio1", "2020Loi"]
papers_input["synthetic"] = ["2020Forti"]
papers_input["other"] = ["Auslander2012MaritimeTD",   "2002Kraiman"]
#papers_input["radar"] = ["2013Sermi"]
#papers_input["radar"] = ["2020Guo"] 
 
non_AIS = set() 
for paper in all_my_papers: 
    found_in_cat = False
    for cat in papers_input:
        if paper in papers_input[cat]:
            found_in_cat = True
            break
    if not found_in_cat:
    	non_AIS.add(paper)
print(non_AIS) 
 
table_of_papers = pd.read_csv("table_of_papers.csv")
papers_threat = dict() 
for word_num in range(len(table_of_papers["Detected threat"])):
    reference = table_of_papers["Reference"][word_num]
    threat = table_of_papers["Detected threat"][word_num]
    #supervised = table_of_papers["Supervised"][word_num]
    position = table_of_papers["Position"][word_num]
    speed = table_of_papers["Speed"][word_num]
    course = table_of_papers["Course"][word_num]
    if reference not in papers_threat:
        papers_threat[reference] = dict()
    papers_threat[reference][threat] = {"Position": position == "YES", "Speed": speed == "YES", "Course": course == "YES"}#"Supervised": supervised == "YES",  
#print(papers_threat)

non_feat = set() 
for paper in all_my_papers:  
    if paper not in papers_threat:
        non_feat.add(paper)
    else:
        found_not_nan = False
        for thrtyp in papers_threat[paper]:
            if str(thrtyp) != "nan":
                found_not_nan = True
                break
        if not found_not_nan:
    	    non_feat.add(paper) 
print(non_feat) 
 
non_my = set() 
for paper in papers_threat:  
    if paper not in all_my_papers: 
    	non_my.add(paper)
print(non_my) 
'''
"2021d’Afflisio"  "Unusual AIS behaviour"  

"2019Fahn"  "Unusual stop"
"2008Riveiro1" "Unusual speed"
"Le2016TrajectoryPO", "2014DuToit" "Unusual vessel trajectories"
papers_threat["Navigation status"] = ["2019Mantecon"] 
papers_threat["Predicting trajectories"] = ["2021Mehri"]  
papers_threat["Damaged AIS messages"] = ["2022Szarmach"]  
papers_threat["Damaged AIS messages"] = ["2022Szarmach"]  
'''
