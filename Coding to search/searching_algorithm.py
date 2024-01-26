import os 

colnames = set()
colnames.add("publication type")
colnames.add("reference")
dict_papers = dict()
current_ref = ""

unique_url = set()
banned = set()
sd_dirs = os.listdir("../SD")
for sd_dir in sd_dirs:
    if sd_dir[0] != "S" or sd_dir[1] != "D":
        continue
    csv_files = os.listdir("../SD/" + sd_dir)  
    suma = 0
    for i in csv_files:
        if i[-1] == "b": 
            new_file = open("../SD/" + sd_dir + "/" + i, "r", encoding = "utf-8")
            new_lines = new_file.readlines()
            new_lines_merged = [""] 
            for j in new_lines:
                new_lines_merged[-1] += j.replace("\n", "")
                if j[0] == "@":
                    pub_type = j[1:j.find("{")]
                    doc_id = j[j.find("{") + 1:-2]
                    current_ref = doc_id
                    while current_ref in dict_papers:
                        current_ref = current_ref + "_1"
                    dict_papers[current_ref] = dict()  
                    dict_papers[current_ref]["publication type"] = pub_type  
                colname = j.split("{")[0]
                if colname.count(" = "): 
                    colnames.add(colname.replace(" = ", ""))
                    rest_of_line = j.split("{")[1]
                    dict_papers[current_ref][colname.replace(" = ", "")] = rest_of_line.replace("},\n", "").replace("}\n", "")
                if j[0] == "}":
                    new_lines_merged.append("")
                    if dict_papers[current_ref]["url"] in unique_url: 
                        banned.add(current_ref)
                        #print("BANNED", current_ref)
                    else:
                        print("NOT BANNED", current_ref) 
                    unique_url.add(dict_papers[current_ref]["url"])
            new_lines_merged = new_lines_merged[:-1] 
            new_file.close() 

print(len(dict_papers))
print(len(colnames))
write_to_file = ""
header_file = '"reference"'
all_lines = ""
for s in colnames:
    header_file += ',"' + s + '"'
for x in dict_papers.keys():
    if x in banned:
        continue
    line_entry = '"' + x + '"'
    for s in colnames:
        if s not in dict_papers[x]:
            dict_papers[x][s] = ""
        if dict_papers[x][s].count('"'):
            print("ERROR", dict_papers[x][s])
        line_entry += ',"' + dict_papers[x][s].replace('"', '\'') + '"'
    line_entry += "\n"
    all_lines += line_entry 

new_file_write = open("ScienceDirect.csv", "w", encoding = "utf-8")
new_file_write.write(header_file + "\n" + all_lines)
new_file_write.close()
