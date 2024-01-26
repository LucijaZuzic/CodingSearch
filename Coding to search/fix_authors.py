import os
import urllib.request

file_bib = open("my_bibliography.bib", "r", encoding="UTF-8")
all_lines_bib = file_bib.readlines()
file_bib.close()

if not os.path.isdir("NEW_DOI"):
    os.mkdir("NEW_DOI")

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
for entry in all_entry:
        if entry.lower().replace("booktitle", "bookttle").count("title") == 2: 
            print("error", entry)  
        start_delimiter = "{"
        end_delimiter = "}"
        if entry.count("{") == 1 or entry.count("{") == 2:  
            start_delimiter = '"'
            end_delimiter = '"'
        position_of_title = entry.lower().replace("booktitle", "bookttle").find("title")
        start_of_title = entry.find(start_delimiter, position_of_title) + 1
        end_of_title = entry.find(end_delimiter, start_of_title) 
        old_title = entry[start_of_title:end_of_title]
        new_title = entry[start_of_title:end_of_title]
        if new_title.count("@") > 0: 
            print("error", new_title, entry)
        set_replaced = set()
        for character in new_title:
            if character.isupper() and character not in set_replaced:
                set_replaced.add(character)
                new_title = new_title.replace(character, "{" + character + "}")

        entry_new = entry
        if entry.lower().count("pages") > 0: 
            position_of_pages = entry.lower().find("pages")
            start_of_pages = entry.find(start_delimiter, position_of_pages) + 1
            end_of_pages = entry.find(end_delimiter, start_of_pages) 
            old_pages_plus_start = entry[position_of_pages:end_of_pages + 2]
            old_pages = entry[start_of_pages:end_of_pages]
            new_pages = old_pages.replace(" ", "").replace("pages", "").replace("Pages", "")
            if "-" in new_pages and "--" not in new_pages:
                new_pages = new_pages.replace("-", "--")
            for x in new_pages:
                if not x.isdigit() and x != "-":
                    print(new_pages) 
                    new_pages = ""
                    break 
            if len(new_pages) == 0:
                entry_new = entry_new.replace(old_pages_plus_start, "")
                print(old_pages_plus_start)  
            else:
                entry_new = entry_new.replace(old_pages, new_pages)
 
        new_str += entry_new.replace(old_title, new_title) + "\n"

        position_of_author = entry.lower().find("author")
        start_of_author = entry.find(start_delimiter, position_of_author) + 1
        end_of_author = entry.find(end_delimiter, start_of_author) 
        old_author = entry[start_of_author:end_of_author]

        all_authors = old_author.split(" ")

        start_of_ref = entry.find("{")
        end_of_ref = entry.find(",")
        reference_original = entry[start_of_ref + 1:end_of_ref] 
  
        author_short = ""  
        for author_index in range(len(all_authors)): 
            if len(all_authors[author_index]) < 2: 
                continue
            if "." in all_authors[author_index]: 
                continue
            if all_authors[author_index] == "and": 
                continue
            if all_authors[author_index].replace(",", "").replace("-", "").lower() not in reference_original.replace("-", "").lower(): 
                continue
            author_short = all_authors[author_index].replace(",", "") + ","

            begin_search = author_index  + 1
            end_search = len(all_authors)
            if author_index != 0:
                begin_search = 0
                end_search = author_index
            for next_index in range(begin_search, end_search): 
                if all_authors[next_index - 1].count(".") != 0 and all_authors[next_index].count(".") == 0: 
                    break 
                if all_authors[next_index][0].islower():
                    break 
                if all_authors[next_index].count(".") != 0:
                    author_short += " " + all_authors[next_index].replace(",", "").replace(".", ". ").replace("  ", " ")
                else:
                    author_short += " " + all_authors[next_index][0] + "."
                    for letter_num in range(0, len(all_authors[next_index])):
                        if all_authors[next_index][letter_num - 1] == '-':
                            author_short += " " + all_authors[next_index][letter_num] + "."  

            break  

        new_names = author_short.strip()
        if old_author.count(" and ") == 1:  
            initial_pos = 0
            for x in range(old_author.count(" and ")):
                find_and = 0
                for next_index in range(initial_pos, len(all_authors)):
                    if all_authors[next_index] == "and":
                        find_and = next_index
                        break  
                if old_author.count(",") == 0:
                    next_surname = all_authors[-1].replace(",", "") + ","
                    begin_quest = find_and + 1
                    end_quest = len(all_authors) - 1 
                else:
                    next_surname = all_authors[find_and + 1].replace(",", "") + ","
                    begin_quest = find_and + 2
                    end_quest = len(all_authors)
                for next_index in range(begin_quest, end_quest):  
                    if all_authors[next_index].count(".") != 0:
                        next_surname += " " + all_authors[next_index].replace(",", "").replace(".", ". ").replace("  ", " ")
                    else:
                        next_surname += " " + all_authors[next_index][0] + "."
                        for letter_num in range(0, len(all_authors[next_index])):
                            if all_authors[next_index][letter_num - 1] == '-':
                                next_surname += " " + all_authors[next_index][letter_num] + "." 
                new_names += " and " + next_surname 
        if old_author.count(" and ") > 1:   
            initial_pos = 0
            for x in range(old_author.count(" and ")):
                find_and = 0
                for next_index in range(initial_pos, len(all_authors)):
                    if all_authors[next_index] == "and":
                        find_and = next_index
                        break
                initial_pos = find_and + 1
                following = len(all_authors)
                for next_index in range(initial_pos, len(all_authors)):
                    if all_authors[next_index] == "and":
                        following = next_index
                        break
                if old_author.count(",") == 0:
                    next_surname = all_authors[following - 1].replace(",", "") + ","
                    begin_quest = find_and + 1
                    end_quest = following - 1 
                else:
                    next_surname = all_authors[find_and + 1].replace(",", "") + ","
                    begin_quest = find_and + 2
                    end_quest = following
                for next_index in range(begin_quest, end_quest):  
                    if all_authors[next_index].count(".") != 0:
                        next_surname += " " + all_authors[next_index].replace(",", "").replace(".", ". ").replace("  ", " ")
                    else:
                        next_surname += " " + all_authors[next_index][0] + "."
                        for letter_num in range(0, len(all_authors[next_index])):
                            if all_authors[next_index][letter_num - 1] == '-':
                                next_surname += " " + all_authors[next_index][letter_num] + "." 
                new_names += " and " + next_surname  
            


        new_names = new_names.replace("  ", " ").strip().replace("-", " ").replace("St Hilaire", "St-Hilaire") 
        #print(reference_original)
        #print(all_authors)
        
        new_str_author += entry_new.replace(old_title, new_title).replace(old_author, new_names) + "\n"
  
file_bib = open("my_bibliography_fix_title.bib", "w", encoding="UTF-8")
file_bib.write(new_str)
file_bib.close()

file_bib = open("my_bibliography_fix_author.bib", "w", encoding="UTF-8")
file_bib.write(new_str_author)
file_bib.close()
'''
names_seen = set()
all_doi = dict() 
for line_num in range(len(all_lines_bib)):
    if "doi" in all_lines_bib[line_num].lower():  
        if "doi" not in all_lines_bib[line_num] and "DOI" not in all_lines_bib[line_num]: 
            continue
        start_delimiter = "{"
        end_delimiter = "}"
        if all_lines_bib[line_num].count("{") == 0:  
            start_delimiter = '"'
            end_delimiter = '"'
        position_of_doi = all_lines_bib[line_num].lower().find("doi")
        start_of_doi = all_lines_bib[line_num].find(start_delimiter, position_of_doi) + 1
        end_of_doi = all_lines_bib[line_num].find(end_delimiter, start_of_doi)
        doi = all_lines_bib[line_num][start_of_doi:end_of_doi].replace(" ", "")
        if len(doi) != 0:  
            all_doi[doi] = "" 
            url = "http://dx.doi.org/" + doi
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
                            open_save = open("NEW_DOI/" + new_reference + ".bib", "w", encoding = "utf-8")
                            open_save.write(new_line)
                            open_save.close()
                            all_doi[doi] = new_reference
                            names_seen.add(new_reference)
                            all_lines_bib += new_line + "\n"
                        else:
                            new_line = new_line[:end_of_reference] + author_name + str(count) + new_line[end_of_reference:] 
                            open_save = open("NEW_DOI/" + new_reference + str(count) + ".bib", "w", encoding = "utf-8")
                            open_save.write(new_line)
                            open_save.close()
                            all_doi[doi] = new_reference + str(count)
                            names_seen.add(new_reference + str(count))
                            all_lines_bib += new_line + "\n"
                            
                        #print(new_line) 
            except:
                all_doi[doi] = ""
                pass 
                
print(len(all_doi)) 
open_save_bib = open("NEW_DOI/all_lines_bib.bib", "w", encoding = "utf-8")
open_save_bib.write(all_lines_bib)
open_save_bib.close()

save_new_doi = '"doi","file"\n'
 
for one_doi in all_doi:   
    save_new_doi += '"' + one_doi + '",' 
    save_new_doi += '"' + all_doi[one_doi] + '"\n' 
        
open_save = open("NEW_DOI/doi_dict.csv", "w", encoding = "utf-8")
open_save.write(save_new_doi)
open_save.close()
'''