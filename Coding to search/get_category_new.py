import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

if not os.path.isdir("Pictures/"):
    os.makedirs("Pictures/")

def return_intersection_data_frame(sets1, sets2, name_plot):
    data_matrix = []
    ylabs = []
    for a in list(sets1.keys()):        
        if len(sets1[a]) == 0:
            continue
        ylabs.append(a)
        data_matrix.append([])
        xlabs = []
        for b in list(sets2.keys()): 
            if len(sets2[b]) == 0:
                continue
            xlabs.append(b)
            data_matrix[-1].append(len(sets1[a].intersection(sets2[b])))
    plt.title(name_plot)
    sns.heatmap(data_matrix, annot = True, xticklabels = xlabs, yticklabels = ylabs, fmt = 'g')
    plt.savefig("Pictures/" + name_plot + "_intersection.png", bbox_inches="tight")
    #plt.show()  
    plt.close()
    
def make_a_plot_sizes(dict_multi, name_plot):
    sizes_dict_entries = []
    for dict_one_name in dict_multi:
        sizes_dict_entries.append(len(dict_multi[dict_one_name]))
    #plt.figure(figsize = (8, 12)) 
    plt.bar(range(0, len(dict_multi)), sizes_dict_entries) 
    step_size = 1
    if max(sizes_dict_entries) > 40:
        step_size = 3
    plt.yticks(range(0, max(sizes_dict_entries) + 1, step_size),  range(0, max(sizes_dict_entries) + 1, step_size))
    plt.xticks(range(0, len(dict_multi)), range(0, len(dict_multi)))
    plt.title(name_plot)
    plt.gca().set_xticklabels(list(dict_multi.keys()), rotation = (45), va = 'top', ha = 'right') 
    plt.savefig("Pictures/" + name_plot + "_size.png", bbox_inches="tight")
    #plt.show() 
    plt.close() 
    
def make_a_plot_years_combined(dict_multi, name_plot): 
    minx = 2000000000000
    maxx = 0
    miny = 0
    maxy = 0 
    
    dict_new_years = dict()
    #plt.figure(figsize = (10, 6)) 
    for dict_one_name in dict_multi: 
        for dict_ref_some in dict_multi[dict_one_name]:
            year_got = int(paper_details[dict_ref_some]["Year"]) 
            minx = min(year_got, minx)
            maxx = max(year_got, maxx) 
            
    for year_some in range(minx, maxx + 1): 
        dict_new_years[year_some] = 0
        
    all_papers_seen = set()
    for dict_one_name in dict_multi: 
        for dict_ref_some in dict_multi[dict_one_name]:
            if dict_ref_some in all_papers_seen:
                continue
            all_papers_seen.add(dict_ref_some)
            year_got = int(paper_details[dict_ref_some]["Year"])
            dict_new_years[year_got] += 1
             
    Y_axis = list(dict_new_years.values())
    miny = min(min(Y_axis), miny)
    maxy = max(max(Y_axis), maxy)
    plt.bar(range(minx, maxx + 1), list(dict_new_years.values()))
    plt.yticks(range(miny, maxy + 1), range(miny, maxy + 1))
    plt.xticks(range(minx, maxx + 1), range(minx, maxx + 1))
    plt.title(name_plot)
    plt.gca().set_xticklabels(range(minx, maxx + 1), rotation = (45), va = 'top', ha = 'right')
    plt.savefig("Pictures/" + name_plot + "_years_combined.png", bbox_inches="tight")
    #plt.show() 
    plt.close()

def random_colors(num_colors):
    colors_set = []
    for x in range(num_colors):
        string_color = "#"
        while string_color == "#" or string_color in colors_set:
            string_color = "#00"
            set_letters = "0123456789ABCDEF"
            for y in range(4):
                string_color += set_letters[np.random.randint(0, 16)]
        colors_set.append(string_color)
    return colors_set

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def make_a_plot_years(dict_multi, name_plot):
    counter_var = 0
    minx = 2000000000000
    maxx = 0
    miny = 0
    maxy = 0
    colors_use = random_colors(len(dict_multi))
    #two_colors = random_colors(2)
    #colors_use = get_color_gradient("#0000FF", "#FF0000", len(dict_multi))
    #colors_use = get_color_gradient(two_colors[0], two_colors[1], len(dict_multi))
    #plt.figure(figsize = (10, 6)) 
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue
        minx = min(min(X_axis), minx)
        maxx = max(max(X_axis), maxx)  
    heights_so_far = [0 for xval in range(minx, maxx + 1)]
    stuff_to_plot = []
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue 
        for year_one in dict_multi[dict_one_name]:
            heights_so_far[year_one - minx] += dict_multi[dict_one_name][year_one] 
            miny = min(heights_so_far[year_one - minx], miny)
            maxy = max(heights_so_far[year_one - minx], maxy)
        stuff_to_plot.append(heights_so_far.copy())
        counter_var += 1  
    for current_plt_num in range(len(stuff_to_plot)):
        X_axis_old = [xval for xval in range(minx, maxx + 1)]
        plt.bar(X_axis_old, stuff_to_plot[len(stuff_to_plot) - 1 - current_plt_num], 
                color = colors_use[current_plt_num],
                label = list(dict_multi.keys())[len(stuff_to_plot) - 1 - current_plt_num])
    if len(dict_multi) > 1:
        plt.legend(ncol = 3, loc='upper center', bbox_to_anchor=(0.5, -0.125))
    plt.yticks(range(miny, maxy + 1), range(miny, maxy + 1))
    plt.xticks(range(minx, maxx + 1), range(minx, maxx + 1))
    plt.title(name_plot)
    plt.gca().set_xticklabels(range(minx, maxx + 1), rotation = (45), va = 'top', ha = 'right')
    plt.savefig("Pictures/" + name_plot + "_years.png", bbox_inches="tight")
    #plt.show() 
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
    #plt.figure(figsize = (20, 6)) 
    if len(dict_multi) > 1:
        plt.vlines(np.arange(minx - 0.5, maxx + 1.5), miny, maxy + 1, linestyle = 'dashed', color = '#c0c0c0')
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue
        X_axis = [X_axis[i] + 1 / (len(dict_multi) + 1) * (counter_var - (len(dict_multi) - 1) / 2) for i in range(len(X_axis))]
        plt.bar(X_axis, list(dict_multi[dict_one_name].values()), 1 / (len(dict_multi) + 1), label = dict_one_name)
        counter_var += 1 
    if len(dict_multi) > 1:
        plt.legend(ncol = len(dict_multi) // 5 + 1, loc='upper center', bbox_to_anchor=(0.5, -0.125))
    plt.yticks(range(miny, maxy + 1), range(miny, maxy + 1))
    plt.xticks(range(minx, maxx + 1), range(minx, maxx + 1))
    plt.title(name_plot)
    plt.gca().set_xticklabels(range(minx, maxx + 1), rotation = (45), va = 'top', ha = 'right')
    plt.savefig("Pictures/" + name_plot + "_years.png", bbox_inches="tight")
    #plt.show() 
    plt.close()

def find_a_reference(text_references):
    file_bib = open(text_references + ".bib", "r", encoding="UTF-8")
    all_lines_bib = file_bib.readlines()
    file_bib.close()   
    concat_lines = "" 
    for line_one in all_lines_bib:
        concat_lines += line_one.replace("\n", "") 
    last_sign = 0
    dict_refs = dict()
    while concat_lines[last_sign:].count("@") > 0:
        first_sign = concat_lines[last_sign:].find("@") + last_sign
        last_sign = first_sign  
        tmp_entry = concat_lines[first_sign:last_sign]
        tmp_ref = tmp_entry[tmp_entry.find("{") + 1:tmp_entry.find(",")] 
        while last_sign <= len(concat_lines) and (tmp_entry.count("{") == 0 or tmp_entry.count("{") != tmp_entry.count("}")):
            last_sign += 1
            tmp_entry = concat_lines[first_sign:last_sign]
            tmp_ref = tmp_entry[tmp_entry.find("{") + 1:tmp_entry.find(",")]
        tmp_entry = tmp_entry[tmp_entry.find(",") + 1:]
        dict_refs[tmp_ref] = tmp_entry
        tmp_categories_old = tmp_entry.split(",")
        tmp_categories = []
        for category_val in tmp_categories_old:
            if "=" in category_val:
                tmp_categories.append(category_val + ",")
            else:
                tmp_categories[-1] += category_val + ","
        dict_categories = dict() 
        for category_val in tmp_categories: 
            tmp_category_and_val = category_val.split("=")
            tmp_category = tmp_category_and_val[0].strip()
            tmp_val = tmp_category_and_val[1].strip() 
            while len(tmp_val) > 0 and (tmp_val[-1] == "}" or tmp_val[-1] == "," or tmp_val[-1] == '"' or tmp_val[-1] == ' '):
                tmp_val = tmp_val[:-1]
            while len(tmp_val) > 0 and (tmp_val[0] == "{" or tmp_val[0] == "," or tmp_val[0] == '"' or tmp_val[0] == ' '):
                tmp_val = tmp_val[1:]
            tmp_val = tmp_val.strip() 
            dict_categories[tmp_category] = tmp_val 
        dict_refs[tmp_ref] = dict_categories
    return dict_refs
    
alg_category = dict()
alg_category["Supervised"] = {"SVM", "CNN", "RNN", "HMM", "SVR", "SVM", "LSTM", "RF", "VRNN", "GNN", "k-NN", "GP", "ANN"}
alg_category["Unsupervised"] = {"DBSCAN", "GAM", "MCM", "DBN", "BN", "OCSVM", "BRNN", "DBC", "SC", "DBSCAN LSTM", "DBSCAN KDE", "CRF", "OSVM", "AE", "KDE", "K-means", "GMM", "OU", "SOM"} 
alg_category["Not applicable"] = {"reinforcement", "filtering", "Lavielle", "parametric", "iGroup", "iDetect", "iGroup-iDetect"}

detection_category = dict()
detection_category["Intents"] = {"IUU fishing", "Smuggling",
                                 "Piracy", "Maritime vessel accidents",
                                 "Small vessel attacks", 
                                 "IUU fishing Trawler", 
                                 "IUU fishing Longliner", 
                                 "IUU fishing Pure Seiner"}
detection_category["Abnormal behaviors"] = {"U turn", "Spiral movement",
                                 "Zig-zag movement", "Drift",
                                 "Unusual vessel trajectories",
                                 "Unusual stop", "Unexpected port arrival",
                                 "Close approach", "Illegal zone entry",
                                 "Wrong dock", "Unexpected AIS activity",
                                 "Unusual AIS behavior",
                                 "Loitering"}
detection_category["Abnormal activities"] = {"Unusual position", "Unusual speed",
                                 "Unusual course"} 
detection_category["Extracting information"] = {"Damaged messages", "Ship type",
                                 "Predicting trajectories", "Sailing stability",
                                 "Destination port and region",
                                 "Object detection", "Navigation status"} 
 
dict_refs_bib = find_a_reference("my_bib_finally")
dict_refs_bib_short = find_a_reference("my_bib_finally_short") 

all_data = dict()
all_algorithms = dict() 
all_position_speed_course = dict()
all_threats = dict()
paper_details = dict()
   
table_of_papers = pd.read_csv("table_of_papers.csv")
for word_num in range(len(table_of_papers["Detected threat"])):
    reference = table_of_papers["Reference"][word_num]
    threat = table_of_papers["Detected threat"][word_num] 
    position = table_of_papers["Position"][word_num] == "YES"
    speed = table_of_papers["Speed"][word_num] == "YES"
    course = table_of_papers["Course"][word_num] == "YES"
    pos_speed_course = str(position * "P" + speed * "S" + course * "C") 
    if pos_speed_course == "":
        pos_speed_course = "N"
    
    algorithm_str = table_of_papers["Algorithm"][word_num].strip()
    algorithm_tmp = algorithm_str.split(" ")
    algorithm = []
    for alg in algorithm_tmp:
        if len(alg.strip()) > 0:
            algorithm.append(alg.strip())
            
    algorithm_str = ""
    for alg in sorted(algorithm):
        if algorithm_str != "":
            algorithm_str += " "
        algorithm_str += alg 
        
    data_str = table_of_papers["Data"][word_num].strip()
    data_tmp = data_str.split(" ")
    data = []
    for data_entry in data_tmp:
        if len(data_entry.strip()) > 0:
            data.append(data_entry.strip())
            
    data_str = ""
    for data_entry in sorted(data):
        if data_str != "":
            data_str += " "
        data_str += data_entry 
        
    authors_tmp = dict_refs_bib_short[reference]["author"].split(" and ")
    authors = []
    for author in authors_tmp:
        if len(author.strip()) > 0:
            authors.append(author.strip().split(",")[0])
    
    one_author = []
    if len(authors) == 1:
        one_author = authors[0]
    if len(authors) == 2:
        one_author = authors[0] + " and " + authors[1]
    if len(authors) > 2:
        one_author = authors[0] + " et al."
    
    #print(dict_refs_bib[reference])
    #print(dict_refs_bib[reference]["year"])

    paper_details[reference] = {"Title": dict_refs_bib[reference]["title"], "Author": one_author, "Year": int(dict_refs_bib[reference]["year"])}
    rest_of_data = {"Threat": threat, "Position Speed Course": pos_speed_course, "Data": data_str, "Algorithm": algorithm_str}
    #minus_threat = {"Position Speed Course": pos_speed_course, "Data": data, "Algorithm": algorithm}
    #minus_position = {"Threat": threat, "Data": data, "Algorithm": algorithm}
    #print(paper_details[reference])

    '''
    if algorithm_str not in all_algorithms:
        all_algorithms[algorithm_str] = dict()
    if reference not in all_algorithms[algorithm_str]:
        all_algorithms[algorithm_str][reference] = []
    all_algorithms[algorithm_str][reference].append(rest_of_data)
    '''

    for alg in algorithm:
        if alg not in all_algorithms:
            all_algorithms[alg] = dict()
        if reference not in all_algorithms[alg]:
            all_algorithms[alg][reference] = []
        all_algorithms[alg][reference].append(rest_of_data)
        
    '''
    if data_str not in all_data:
        all_data[data_str] = dict()
    if reference not in all_data[data_str]:
        all_data[data_str][reference] = []
    all_data[data_str][reference].append(rest_of_data)
    '''

    for data_entry in data: 
        if data_entry not in all_data:
            all_data[data_entry] = dict()
        if reference not in all_data[data_entry]:
            all_data[data_entry][reference] = []
        all_data[data_entry][reference].append(rest_of_data)
        
    if threat not in all_threats:
        all_threats[threat] = dict()
    if reference not in all_threats[threat]:
        all_threats[threat][reference] = []
    all_threats[threat][reference].append(rest_of_data)
    
    if pos_speed_course not in all_position_speed_course:
        all_position_speed_course[pos_speed_course] = dict()
    if reference not in all_position_speed_course[pos_speed_course]:
        all_position_speed_course[pos_speed_course][reference] = []
    all_position_speed_course[pos_speed_course][reference].append(rest_of_data)
    
def print_table(dictk, name_table, add_type = True, keyadd = "Threat", include_space = True, include_single = True):
    string_to_print_total = ""
    for keyd in dictk:  
        string_to_print = ""
        
        if " " in keyd:
            keyds_other = keyd.split(" ")
            for keyds_some in keyds_other:
                keyds = keyds_some.strip() 
                if len(keyds) < 1:
                    continue
                if keyds not in dictk:
                    continue
                for refd in dictk[keyds]:
                    if string_to_print == "":
                        string_to_print = str(keyds) + " & "
                    else:
                         string_to_print += ", "
                    added_str = ""
                    for threat_some in dictk[keyds][refd]:
                        added_str += threat_some[keyadd] + " "
                    string_to_print += paper_details[refd]["Author"] + add_type * " " + add_type * added_str + " \\cite{" + refd + "}"
            if not include_space:
                continue
                
        if not include_single and " " not in keyd:
            continue
            
        for ref in dictk[keyd]:
            if string_to_print == "":
                string_to_print = str(keyd) + " & "
            else:
                string_to_print += ", "
            added_str = ""
            for threat_some in dictk[keyd][ref]:
                added_str += threat_some[keyadd] + " "
            string_to_print += paper_details[ref]["Author"] + add_type * " " + add_type * added_str + " \\cite{" + ref + "}"
                    
        string_to_print += " \\\\ \\hline \n"
        string_to_print_total += string_to_print
    file_txt = open(name_table + ".txt", "w", encoding="UTF-8")
    file_txt.write(string_to_print_total)
    file_txt.close()   
    print(string_to_print_total)
    
def papers_from_table(dictk, include_space = True, include_single = True): 
    from_for_category = dict()
    for keyd in dictk:
        if " " in keyd:
            keyds_other = keyd.strip().split(" ")
            for keyds_some in keyds_other:
                keyds = keyds_some.strip() 
                if len(keyds) < 1:
                    continue
                if keyds not in dictk:
                    continue
                if keyds not in from_for_category:
                    from_for_category[keyds] = set() 
                from_for_category[keyds] = from_for_category[keyds].union(dictk[keyds].keys()) 
            if not include_space: 
                continue
        if not include_single and " " not in keyd:
            continue
        if keyd not in from_for_category:
            from_for_category[keyd] = set() 
        from_for_category[keyd] = from_for_category[keyd].union(dictk[keyd].keys()) 
    return(from_for_category)
    
def years_from_table(dictk, include_space = True, include_single = True): 
    years_for_category = dict()
    for keyd in dictk: 
        if " " in keyd:
            keyds_other = keyd.strip().split(" ")
            for keyds_some in keyds_other:
                keyds = keyds_some.strip()
                if len(keyds) < 1:
                    continue
                if keyds not in dictk:
                    continue
                if keyds not in years_for_category:
                    years_for_category[keyds] = dict() 
                for ref in dictk[keyds]:
                    if paper_details[ref]["Year"] not in years_for_category[keyds]:
                        years_for_category[keyds][paper_details[ref]["Year"]] = 0
                    years_for_category[keyds][paper_details[ref]["Year"]] += 1
            if not include_space:
                continue
        if not include_single and " " not in keyd:
            continue
        if keyd not in years_for_category:
            years_for_category[keyd] = dict() 
        for ref in dictk[keyd]:
            if paper_details[ref]["Year"] not in years_for_category[keyd]:
                years_for_category[keyd][paper_details[ref]["Year"]] = 0
            years_for_category[keyd][paper_details[ref]["Year"]] += 1
    return(years_for_category)
    
all_threats_by_detection = dict()
for threat_type in all_threats:
    category_sorted = "Other threats"
    for category_detected in detection_category:
        if threat_type in detection_category[category_detected]:
            category_sorted = category_detected
            break
    if category_sorted not in all_threats_by_detection:
        all_threats_by_detection[category_sorted] = dict()
    all_threats_by_detection[category_sorted][threat_type] = all_threats[threat_type]

all_algorithms_by_supervised = dict()
for algo_type in all_algorithms:
    category_sorted = "Other algorithms"
    categories_subset = set()
    algo_type_list = algo_type.split(" ")
    for category_detected in alg_category:
        if algo_type in alg_category[category_detected]:
            category_sorted = category_detected
        for algo_type2 in alg_category[category_detected]:
            if algo_type2 in algo_type_list:
                categories_subset.add(category_detected)
    if len(categories_subset) == 1:
        for algo_type2 in categories_subset:
            category_sorted = algo_type2
    if len(categories_subset) > 1:
        category_sorted = "Combined"
    if category_sorted not in all_algorithms_by_supervised:
        all_algorithms_by_supervised[category_sorted] = dict()
    all_algorithms_by_supervised[category_sorted][algo_type] = all_algorithms[algo_type]

print_table(all_algorithms, "Algorithm", False) 
for key_cat in all_algorithms_by_supervised: 
    print(key_cat)  
    print_table(all_algorithms_by_supervised[key_cat], key_cat, False)  
print_table(all_data, "Data source", False)
print_table(all_position_speed_course, "Variables", False)
print_table(all_threats, "Behavior", False)  
for key_cat in all_threats_by_detection: 
    print(key_cat)  
    print_table(all_threats_by_detection[key_cat], key_cat, False)  
     
make_a_plot_sizes(papers_from_table(all_data), "Data source") 
make_a_plot_years(years_from_table(all_data), "Data source") 
make_a_plot_years_combined(papers_from_table(all_data), "Data source") 

make_a_plot_sizes(papers_from_table(all_algorithms), "Algorithm") 
make_a_plot_years(years_from_table(all_algorithms), "Algorithm") 
make_a_plot_years_combined(papers_from_table(all_algorithms), "Algorithm") 

for key_cat in all_algorithms_by_supervised: 
    make_a_plot_sizes(papers_from_table(all_algorithms_by_supervised[key_cat]), key_cat)  
    make_a_plot_years(years_from_table(all_algorithms_by_supervised[key_cat]), key_cat)  
    make_a_plot_years_combined(papers_from_table(all_algorithms_by_supervised[key_cat]), key_cat)  
    
make_a_plot_sizes(papers_from_table(all_position_speed_course), "Variables") 
make_a_plot_years(years_from_table(all_position_speed_course), "Variables") 
make_a_plot_years_combined(papers_from_table(all_position_speed_course), "Variables")  

make_a_plot_sizes(papers_from_table(all_threats, True), "Behavior") 
make_a_plot_years(years_from_table(all_threats, True), "Behavior") 
make_a_plot_years_combined(papers_from_table(all_threats, True), "Behavior") 

for key_cat in all_threats_by_detection: 
    make_a_plot_sizes(papers_from_table(all_threats_by_detection[key_cat], True), key_cat) 
    make_a_plot_years(years_from_table(all_threats_by_detection[key_cat], True), key_cat) 
    make_a_plot_years_combined(papers_from_table(all_threats_by_detection[key_cat], True), key_cat)  
    
return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_algorithms), "Data source Algorithm")
for key_cat in all_algorithms_by_supervised:
    return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_algorithms_by_supervised[key_cat]), "Data source " + key_cat)
    
return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_position_speed_course), "Data source Variables")

return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_threats), "Data source Behavior")
for key_cat in all_threats_by_detection: 
    return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_threats_by_detection[key_cat]), "Data source " + key_cat)
    
return_intersection_data_frame(papers_from_table(all_algorithms), papers_from_table(all_position_speed_course), "Algorithm Variables")
for key_cat in all_algorithms_by_supervised:
    return_intersection_data_frame(papers_from_table(all_algorithms_by_supervised[key_cat]), papers_from_table(all_position_speed_course), key_cat + " Variables")
    
return_intersection_data_frame(papers_from_table(all_algorithms), papers_from_table(all_threats), "Algorithm Behavior")
for key_cat1 in all_algorithms_by_supervised:
    return_intersection_data_frame(papers_from_table(all_algorithms_by_supervised[key_cat1]), papers_from_table(all_threats), key_cat1 + " Behavior")
    for key_cat2 in all_threats_by_detection: 
        return_intersection_data_frame(papers_from_table(all_algorithms_by_supervised[key_cat1]), papers_from_table(all_threats_by_detection[key_cat2]), key_cat1 + " " + key_cat2)
        
for key_cat2 in all_threats_by_detection:
    return_intersection_data_frame(papers_from_table(all_algorithms), papers_from_table(all_threats_by_detection[key_cat2]), "Algorithm " + key_cat2)

return_intersection_data_frame(papers_from_table(all_position_speed_course), papers_from_table(all_threats), "Variables Behavior")
for key_cat in all_threats_by_detection: 
    return_intersection_data_frame(papers_from_table(all_position_speed_course), papers_from_table(all_threats_by_detection[key_cat]), "Variables " + key_cat)
