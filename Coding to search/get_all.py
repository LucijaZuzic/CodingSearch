import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import copy
from PIL import Image
from copy import deepcopy

#plt.rcParams.update({'font.size': 20}) 

def merge_images(image_names, name_total_plots):

    images = [] 
    sizes_x = [] 
    sizes_y = []  
    #Read the images 
    for order_num in range(len(image_names)): 
        if not os.path.isfile("Pictures/" + image_names[order_num].replace(" - ", " ") + "_intersection.png"):
            continue

        image = Image.open("Pictures/" + image_names[order_num].replace(" - ", " ") + "_intersection.png")
        x, y = image.size
        images.append(image)
        sizes_x.append(x)
        sizes_y.append(y)  
    if len(images) == 1:
        x_size = sum(sizes_x) 
        y_size = max(sizes_y)
        x_start = [0]
        y_start = [0]
    if len(images) == 2:
        x_size = sum(sizes_x) 
        y_size = max(sizes_y)
        x_start = [0, sizes_x[0]]
        y_start = [0, 0]
    if len(images) == 3:  
        x_size = max(sizes_x[0], sizes_x[2]) + sizes_x[1]
        y_size = max(sizes_y[0], sizes_y[1]) + sizes_y[2]
        x_start = [0, max(sizes_x[0], sizes_x[2]), 0]
        y_start = [0, 0, max(sizes_y[0], sizes_y[1])]
        if sizes_x[0] > sizes_x[2]:  
            x_start[2] += sizes_x[0] - sizes_x[2]
        if sizes_x[2] > sizes_x[0]:  
            x_start[0] += sizes_x[2] - sizes_x[0]
    if len(images) == 4: 
        x_size = max(sizes_x[0], sizes_x[2]) + max(sizes_x[1], sizes_x[3])
        y_size = max(sizes_y[0], sizes_y[1]) + max(sizes_y[2], sizes_y[3])
        x_start = [0, max(sizes_x[0], sizes_x[2]), 0, max(sizes_x[0], sizes_x[2])]
        y_start = [0, 0, max(sizes_y[0], sizes_y[1]), max(sizes_y[0], sizes_y[1])]
        
        if sizes_x[0] > sizes_x[2]:  
            x_start[2] += sizes_x[0] - sizes_x[2]
            
        if sizes_x[2] > sizes_x[0]:  
            x_start[0] += sizes_x[2] - sizes_x[0]
            
        if sizes_x[3] > sizes_x[1]:  
            x_start[1] += sizes_x[3] - sizes_x[1]
            
        if sizes_x[1] > sizes_x[3]:  
            x_start[3] += sizes_x[1] - sizes_x[3]

    new_image = Image.new('RGB',(x_size, y_size), (255,255,255))
	 
    for order_num in range(len(images)):
        if not os.path.isfile("Pictures/" + image_names[order_num].replace(" - ", " ") + "_intersection.png"):
            continue 
        new_image.paste(images[order_num], (x_start[order_num], y_start[order_num]))  
    new_image.save("Pictures/" + name_total_plots + "_intersection.png","PNG")   

if not os.path.isdir("Pictures/"):
    os.makedirs("Pictures/")

def add_margin(ax,x=0.05,y=0.05):
    # This will, by default, add 5% to the x and y margins. You 
    # can customise this using the x and y arguments when you call it.

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmargin = (xlim[1]-xlim[0])*x
    ymargin = (ylim[1]-ylim[0])*y

    ax.set_xlim(xlim[0]-xmargin,xlim[1]+xmargin)
    ax.set_ylim(ylim[0]-ymargin,ylim[1]+ymargin)

def return_intersection_data_frames(setsxs, setsys, name_plots, name_total_plots):

    fig, axes = plt.subplots(1, len(name_plots), figsize=(40, 10))

    for order_num in range(len(name_plots)):
        sets1 = copy.deepcopy(setsxs[order_num])
        sets2 = copy.deepcopy(setsxs[order_num])
        total_lens_a = set()
        total_lens_b = set()
        data_matrix = []
        ylabs = []

        for a in list(sets1.keys()):  
            total_len = 0
            for b in list(sets2.keys()): 
                total_len  += len(sets1[a].intersection(sets2[b])) 
            if total_len == 0:
                total_lens_a.add(a) 

        for b in list(sets2.keys()):  
            total_len = 0
            for a in list(sets1.keys()): 
                total_len  += len(sets1[a].intersection(sets2[b])) 
            if total_len == 0:
                total_lens_b.add(b) 

        for a in total_lens_a:   
            sets1.pop(a)

        for b in total_lens_b:   
            sets2.pop(b)

        if len(sets1) + len(sets2) == 0:   
            return

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
        s = plt.subplot(1, order_num + 1, order_num + 1)
        plt.title(name_plots[order_num])
        sns.heatmap(data_matrix, annot = True, xticklabels = xlabs, yticklabels = ylabs, fmt = 'g', cbar = False)
      
        # Check what the original limits were 
        x0,y0=s.get_xlim(),s.get_ylim()
  
        # Update the limits using set_xlim and set_ylim 
        add_margin(s,x=0.5,y=0.01) ### Call this after tsplot 

    plt.savefig("Pictures/" + name_total_plots + "_intersection.png", bbox_inches="tight")
    #plt.show()  
    plt.close()

def return_intersection_data_frame(setsx, setsy, name_plot):
    sets1 = copy.deepcopy(setsx)
    sets2 = copy.deepcopy(setsy)
    total_lens_a = set()
    total_lens_b = set()
    data_matrix = []
    ylabs = []

    for a in list(sets1.keys()):  
        total_len = 0
        for b in list(sets2.keys()): 
            total_len  += len(sets1[a].intersection(sets2[b])) 
        if total_len == 0:
            total_lens_a.add(a) 

    for b in list(sets2.keys()):  
        total_len = 0
        for a in list(sets1.keys()): 
            total_len  += len(sets1[a].intersection(sets2[b])) 
        if total_len == 0:
            total_lens_b.add(b) 

    for a in total_lens_a:   
        sets1.pop(a)

    for b in total_lens_b:   
        sets2.pop(b)

    if len(sets1) + len(sets2) == 0:   
        return

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
    sns.heatmap(data_matrix, annot = True, xticklabels = xlabs, yticklabels = ylabs, fmt = 'g', cbar = False)
    plt.savefig("Pictures/" + name_plot.replace(" - ", " ") + "_intersection.png", bbox_inches="tight")
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
    color_space = int(np.floor(16 ** 4 // (num_colors + 1)))
    for x in range(num_colors):
        #string_color = "#"
        #while string_color == "#" or string_color in colors_set:
            #string_color = "#00"
            #set_letters = "0123456789ABCDEF"
            #for y in range(4):
                #string_color += set_letters[np.random.randint(0, 16)]
        ord_of_color = color_space * (x + 1)
        cstr = str(hex(ord_of_color)).replace("0x", "").upper()
        while len(cstr) != 6:
            cstr = "0" + cstr
        colors_set.append("#" + cstr)
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
     
other_names = {"Not applicable": "Statistical Models",
               "Intents": "The Intent Which Caused\nThe Anomalous Behavior",
               "Abnormal activities": "Abnormal Activities\nin a Single Variable",
               "Abnormal behaviors": "Abnormal Behavior\nin Multiple Variables",
               "Data source": "Data Sources",
               "Extracting information": "Extracting Information",
               "Unsupervised": "Unsupervised Learning Algorithms",
               "Supervised": "Supervised Learning Algorithms",
               "Variables": "Position (P), Speed (S), and Course (C),\nor None of These (N) Variables",
               }

def make_a_plot_years_sizes(dict_multi, name_plot):
    name_plot_tmp = name_plot
    if name_plot in other_names:
        name_plot_tmp = other_names[name_plot]
    counter_var = 0
    minx = 2000000000000
    maxx = 0
    miny = 0
    maxy = 0
    new_dimensions = {"Not applicable": [[1, (10, 10), "upper left"], [1, (10, 10), "upper left"]],
                "Intents": [[1, (11, 10), "upper left"], [1, (10, 10), "upper left"]],
                "Abnormal activities": [[1, (10, 10), "upper left"], [1, (10, 12), "upper right"]],
                "Abnormal behaviors": [[1, (10, 24), "upper left"], [1, (10, 12), "upper left"]],
                "Data source": [[1, (10, 10), "upper left"], [1, (10, 10), "upper left"]],
                "Extracting information": [[1, (10, 13), "upper left"], [1, (10, 10), "upper left"]],
                "Unsupervised": [[2, (15, 15), "upper left"], [2, (15, 15), "upper left"]],
                "Supervised": [[2, (15, 15), "upper left"], [2, (15, 15), "upper left"]],
                "Variables": [[1, (10, 10), "upper left"], [1, (10, 10), "upper left"]],
                "Algorithm": [[2, (20, 20), "upper left"], [2, (20, 20), "upper left"]],
                "Behavior": [[2, (25, 25), "upper left"], [2, (20, 25), "upper left"]],
                }
    #colors_use_original = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    colors_use_original = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    #colors_use = random_colors(len(dict_multi))
    patterns_use_original = ["", "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    #two_colors_original = random_colors(2)
    #colors_use = get_color_gradient("#0000FF", "#FF0000", len(dict_multi))
    #colors_use = get_color_gradient(two_colors[0], two_colors[1], len(dict_multi))
    #plt.figure(figsize = (10, 6))
    colors_use_final = []
    patterns_use_final = []
    color_current_num = 0
    pattern_current_num = 0
    for pattern_color_num in range(len(dict_multi)):
        colors_use_final.append(colors_use_original[color_current_num])
        patterns_use_final.append(patterns_use_original[pattern_current_num])
        color_current_num += 1
        if color_current_num == len(colors_use_original):
            color_current_num = 0
            pattern_current_num += 1
            if pattern_current_num == len(patterns_use_original):
                pattern_current_num = 0
    for dict_one_name in dict_multi:
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue
        minx = min(min(X_axis), minx)
        maxx = max(max(X_axis), maxx)  
    heights_so_far = [0 for xval in range(minx, maxx + 1)]
    stuff_to_plot = []
    dict_multi_names = dict()
    for dict_one_name in dict_multi:
        dict_multi_names[dict_one_name] = 0
        X_axis = list(dict_multi[dict_one_name].keys())
        if len(X_axis) == 0:
            continue 
        for year_one in dict_multi[dict_one_name]:
            heights_so_far[year_one - minx] += dict_multi[dict_one_name][year_one] 
            dict_multi_names[dict_one_name] += dict_multi[dict_one_name][year_one] 
            miny = min(heights_so_far[year_one - minx], miny)
            maxy = max(heights_so_far[year_one - minx], maxy)
        stuff_to_plot.append(heights_so_far.copy())
        counter_var += 1
    sizes_use = new_dimensions[name_plot][0][1]
    plt.figure(figsize = sizes_use, dpi = 600)
    plt.rcParams.update({'font.size': 20})  
    dict_multi_colors = dict() 
    dict_multi_patterns = dict() 
    maxlabellen = 0
    ncols = new_dimensions[name_plot][0][0]
    locs = new_dimensions[name_plot][0][2]
    lblnms = dict()
    lblnms_no_num = dict()
    for current_plt_num in range(len(stuff_to_plot)):
        X_axis_old = [xval for xval in range(minx, maxx + 1)]
        dict_multi_colors[list(dict_multi.keys())[len(stuff_to_plot) - 1 - current_plt_num]] = colors_use_final[current_plt_num]
        dict_multi_patterns[list(dict_multi.keys())[len(stuff_to_plot) - 1 - current_plt_num]] = patterns_use_final[current_plt_num]
        lbl_no_num = list(dict_multi.keys())[len(stuff_to_plot) - 1 - current_plt_num]
        lbl = str(current_plt_num + 1) + " " + list(dict_multi.keys())[len(stuff_to_plot) - 1 - current_plt_num]
        if len(lbl) > 30:
            lbl = lbl.replace("+", "+\n")
        if len(lbl_no_num) > 30:
            lbl_no_num = lbl_no_num.replace("+", "+\n")
        lblnms[colors_use_final[current_plt_num] + patterns_use_final[current_plt_num]] = lbl
        lblnms_no_num[colors_use_final[current_plt_num] + patterns_use_final[current_plt_num]] = lbl_no_num
        edgecolor_use = "#000000"
        if colors_use_final[current_plt_num] == "#000000":
            edgecolor_use = "#F0E442"
        plt.bar(X_axis_old, stuff_to_plot[len(stuff_to_plot) - 1 - current_plt_num], 
                color = colors_use_final[current_plt_num],
                edgecolor = edgecolor_use,
                hatch = patterns_use_final[current_plt_num],
                label = lbl_no_num)
        maxlabellen = max(maxlabellen, len(str(current_plt_num + 1) + " " + list(dict_multi.keys())[len(stuff_to_plot) - 1 - current_plt_num]))
    plt.legend(ncol = ncols, loc = locs)
    plt.yticks(range(miny, maxy + 1), range(miny, maxy + 1))
    plt.xticks(range(minx, maxx + 1), range(minx, maxx + 1))
    plt.xlabel("Year")
    plt.ylabel("Number of Papers")
    plt.title(name_plot_tmp + "\n(Occurence by Year and Category)")
    plt.gca().set_xticklabels(range(minx, maxx + 1), rotation = (45), va = 'top', ha = 'right')
    plt.rc('xtick', labelsize = 16) 
    plt.rc('ytick', labelsize = 16) 

    plt.savefig("Pictures/" + name_plot + "_years_sizes_years.png", bbox_inches="tight")
    #plt.show() 
    plt.close()
    
    sizes_use = new_dimensions[name_plot][1][1]
    ncols = new_dimensions[name_plot][1][0]
    locs = new_dimensions[name_plot][1][2]
    plt.figure(figsize = sizes_use, dpi = 600) 
    plt.rcParams.update({'font.size': 20})  
    sizes_dict_entries = [] 
    width_bar = len(str(len(stuff_to_plot)))
    for current_plt_num in range(len(stuff_to_plot)):
        for dict_one_name in dict_multi_names:
            lbl_number_extract = int(lblnms[dict_multi_colors[dict_one_name] + dict_multi_patterns[dict_one_name]].split(" ")[0])
            if lbl_number_extract != current_plt_num + 1:
                continue
            sizes_dict_entries.append(dict_multi_names[dict_one_name])
            #print(dict_one_name, len(sizes_dict_entries) - 1, dict_multi_names[dict_one_name])
            edgecolor_use = "#000000"
            if colors_use_final[current_plt_num] == "#000000":
                edgecolor_use = "#F0E442"
            bar1 = plt.bar(current_plt_num * width_bar + 1, 
                        dict_multi_names[dict_one_name], 
                        color = dict_multi_colors[dict_one_name],
                        edgecolor = edgecolor_use,
                        hatch = dict_multi_patterns[dict_one_name],
                        label = lblnms[dict_multi_colors[dict_one_name] + dict_multi_patterns[dict_one_name]]) 
        
            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')
 
    step_size = 1
    if max(sizes_dict_entries) > 40:
        step_size = 3
    plt.legend(ncol = ncols, loc = locs)
    plt.yticks(range(0, max(sizes_dict_entries) + 1, step_size),  range(0, max(sizes_dict_entries) + 1, step_size))
    plt.xticks(range(1, 1 + len(dict_multi_names) * width_bar, width_bar), range(1, 1 + len(dict_multi_names), 1))
    plt.xlabel("Category")
    plt.ylabel("Number of Papers")
    plt.title(name_plot_tmp + "\n(Occurence by Category)")
    #plt.gca().set_xticklabels(range(1, 1 + len(dict_multi_names)), rotation = (45), va = 'top', ha = 'right')  
    plt.rc('xtick', labelsize = 16) 
    plt.rc('ytick', labelsize = 16) 
     
    plt.savefig("Pictures/" + name_plot + "_years_sizes_sizes.png", bbox_inches="tight")
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
     
def print_table_v2(dictk, name_table, lens): 
    string_to_print_total = "" 
    keyd_dict = dict(sorted(dictk.items(), key=lambda item: len(item[1]), reverse = True))
    keyd_list = list(keyd_dict.keys())
    for keyd_inde in range(0, len(keyd_list) + 1 - len(lens)):
        string_to_print = ["\\hline \n"] 
        maxnumref = 0 
        for keyd_offset in range(len(lens)):
            keyd = keyd_list[keyd_inde + keyd_offset]
            maxnumref = max(maxnumref, len(dictk[keyd]))
        for keyd_offset in range(len(lens)):
            keyd = keyd_list[keyd_inde + keyd_offset]
            num_refs_printed = 0
            string_to_print[0] += "\\multicolumn{" + str(lens[keyd_offset]) + "}{|c|}{" + str(keyd) + "}" 
            if keyd_offset == len(lens) - 1:
                string_to_print[0] += " \\\\ \\hline \n"
            else:
                string_to_print[0] += " & "
            line_number = 1
            refs = list(dictk[keyd].keys())
            while len(refs) != maxnumref:
                refs.append("")
            for ref in refs:
                if line_number == len(string_to_print):
                    string_to_print.append("")    
                if num_refs_printed % lens[keyd_offset] == lens[keyd_offset] - 1 and keyd_offset != len(lens) - 1: 
                    if lens[keyd_offset] == 1:
                        string_to_print[line_number] += "\\multicolumn{1}{|c|}{"
                    else:
                        string_to_print[line_number] += "\\multicolumn{1}{c|}{"
                if ref != "":
                    string_to_print[line_number] += paper_details[ref]["Author"] + " \\cite{" + ref + "}" 
                if num_refs_printed % lens[keyd_offset] == lens[keyd_offset] - 1 and keyd_offset != len(lens) - 1: 
                    string_to_print[line_number] += "}"
                if num_refs_printed % lens[keyd_offset] == lens[keyd_offset] - 1: 
                    if keyd_offset == len(lens) - 1:
                        string_to_print[line_number] += " \\\\ \n"
                    else:
                        string_to_print[line_number] += " & "
                    line_number += 1
                else:
                    string_to_print[line_number] += " & " 
                num_refs_printed += 1
             
            if line_number < len(string_to_print):
                while num_refs_printed % lens[keyd_offset] != lens[keyd_offset] - 1:
                    string_to_print[line_number] += " & " 
                    num_refs_printed += 1
                if num_refs_printed % lens[keyd_offset] == lens[keyd_offset] - 1 and keyd_offset != len(lens) - 1: 
                    if lens[keyd_offset] == 1:
                        string_to_print[line_number] += "\\multicolumn{1}{|c|}{}"
                    else:
                        string_to_print[line_number] += "\\multicolumn{1}{c|}{}" 
                if keyd_offset == len(lens) - 1:
                    string_to_print[line_number] += " \\\\ \n"
                else:
                    string_to_print[line_number] += " & "
        for line_number in range(len(string_to_print)):
            string_to_print_total += string_to_print[line_number]
    new_name = name_table 
    for x in lens:
        new_name += "_" + str(x)
    if not os.path.isdir(name_table + "_tables/"):
        os.makedirs(name_table + "_tables/")
    file_txt = open(name_table + "_tables/" + new_name + ".txt", "w", encoding="UTF-8")
    file_txt.write(string_to_print_total + "\\hline \n")
    file_txt.close()   

def print_table(dictk, name_table, add_type = True, keyadd = "Threat", include_space = True, include_single = True):   
    for one_num in range(1, 7): 
        list_num = [one_num]
        if sum(list_num) > 6:
            continue
        print_table_v2(dictk, name_table, list_num)    
        for two_num in range(1, one_num + 1): 
            list_num = [one_num, two_num]
            if sum(list_num) > 6:
                continue
            print_table_v2(dictk, name_table, list_num)   
            for three_num in range(1, two_num + 1):
                list_num = [one_num, two_num, three_num]
                if sum(list_num) > 6:
                    continue
                print_table_v2(dictk, name_table, list_num)   
                for four_num in range(1, three_num + 1):
                    list_num = [one_num, two_num, three_num, four_num]
                    if sum(list_num) > 6:
                        continue
                    print_table_v2(dictk, name_table, list_num) 
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
    if not os.path.isdir(name_table + "_tables/"):
        os.makedirs(name_table + "_tables/")
    file_txt = open(name_table + "_tables/" + name_table + ".txt", "w", encoding="UTF-8")
    file_txt.write(string_to_print_total)
    file_txt.close()   
    #print(string_to_print_total)
    
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
    all_threats_by_detection[category_sorted][threat_type] = copy.deepcopy(all_threats[threat_type])

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
    all_algorithms_by_supervised[category_sorted][algo_type] = copy.deepcopy(all_algorithms[algo_type])
    
def filter_dict(dict_filt):
    #print("New dict")
    fi = True
    while fi:
        #print("New round")
        fi = False
        lk = sorted(list(dict_filt.keys()))
        for c1 in range(len(lk)):
            for c2 in range(c1 + 1, len(lk)): 
                if lk[c1] not in dict_filt or lk[c2] not in dict_filt:
                    continue
                #print(lk[c1], lk[c2])
                s1 = set(dict_filt[lk[c1]].keys())
                s2 = set(dict_filt[lk[c2]].keys())
                inters = s1.intersection(s2)
                if len(inters) > 0:
                    #print(len(inters))
                    fi = True
                    dict_filt[lk[c1] + "+" + lk[c2]] = dict()
                    for refe in inters:  
                        dict_filt[lk[c1]  + "+" + lk[c2]][refe] = dict_filt[lk[c1]][refe]
                        dict_filt[lk[c1]].pop(refe)
                        dict_filt[lk[c2]].pop(refe)
                    if len(dict_filt[lk[c1]]) == 0:
                        dict_filt.pop(lk[c1]) 
                        #print("Remove", lk[c1]) 
                    if len(dict_filt[lk[c2]]) == 0:
                        dict_filt.pop(lk[c2]) 
                        #print("Remove", lk[c2]) 
    return dict_filt

all_algorithms = filter_dict(all_algorithms)        
print_table(all_algorithms, "Algorithm", False) 
for key_cat in all_algorithms_by_supervised: 
    #print(key_cat)  
    all_algorithms_by_supervised[key_cat] = filter_dict(all_algorithms_by_supervised[key_cat])  
    print_table(all_algorithms_by_supervised[key_cat], key_cat, False)  
all_data = filter_dict(all_data)
print_table(all_data, "Data source", False)
all_position_speed_course = filter_dict(all_position_speed_course)
print_table(all_position_speed_course, "Variables", False)
all_threats = filter_dict(all_threats)
print_table(all_threats, "Behavior", False)  
for key_cat in all_threats_by_detection: 
    #print(key_cat)  
    all_threats_by_detection[key_cat] = filter_dict(all_threats_by_detection[key_cat])  
    print_table(all_threats_by_detection[key_cat], key_cat, False)  
make_plots_true = True    
if make_plots_true:
    make_a_plot_years_sizes(years_from_table(all_data), "Data source")  
    
    make_a_plot_years_sizes(years_from_table(all_algorithms), "Algorithm")  
        
    make_a_plot_years_sizes(years_from_table(all_position_speed_course), "Variables")  
    
    make_a_plot_years_sizes(years_from_table(all_threats, True), "Behavior")  

    for key_cat in all_threats_by_detection:  
        print(key_cat)  
        make_a_plot_years_sizes(years_from_table(all_threats_by_detection[key_cat], True), key_cat)
        
    for key_cat in all_algorithms_by_supervised:  
        print(key_cat)  
        make_a_plot_years_sizes(years_from_table(all_algorithms_by_supervised[key_cat]), key_cat)    

    return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_algorithms), "Data source - Algorithm")
    setsxs = []
    setsys = []
    name_plots = []
    for key_cat in all_algorithms_by_supervised:
        return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_algorithms_by_supervised[key_cat]), "Data source - " + key_cat)
        setsxs.append(papers_from_table(all_data)) 
        setsys.append(papers_from_table(all_algorithms_by_supervised[key_cat]))
        name_plots.append("Data source - " + key_cat)
    #return_intersection_data_frames(setsxs, setsys, name_plots, "Data source Algorithm Multi")
    merge_images(name_plots, "Data source Algorithm Multi")

    return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_position_speed_course), "Data source - Variables")

    return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_threats), "Data source - Behavior")
    setsxs = []
    setsys = []
    name_plots = []
    for key_cat in all_threats_by_detection: 
        return_intersection_data_frame(papers_from_table(all_data), papers_from_table(all_threats_by_detection[key_cat]), "Data source - " + key_cat)
        setsxs.append(papers_from_table(all_data)) 
        setsys.append(papers_from_table(all_threats_by_detection[key_cat]))
        name_plots.append("Data source - " + key_cat)
    #return_intersection_data_frames(setsxs, setsys, name_plots, "Data source Behavior Multi")
    merge_images(name_plots, "Data source Behavior Multi")
    
    return_intersection_data_frame(papers_from_table(all_algorithms), papers_from_table(all_position_speed_course), "Algorithm - Variables")
    setsxs = []
    setsys = []
    name_plots = []
    for key_cat in all_algorithms_by_supervised:
        return_intersection_data_frame(papers_from_table(all_algorithms_by_supervised[key_cat]), papers_from_table(all_position_speed_course), key_cat + " - Variables")
        setsxs.append(papers_from_table(all_algorithms_by_supervised[key_cat])) 
        setsys.append(papers_from_table(all_position_speed_course))
        name_plots.append(key_cat + " - Variables")
    #return_intersection_data_frames(setsxs, setsys, name_plots, "Algorithm Variables Multi")
    merge_images(name_plots, "Algorithm Variables Multi")
    
    return_intersection_data_frame(papers_from_table(all_algorithms), papers_from_table(all_threats), "Algorithm - Behavior")
    for key_cat1 in all_algorithms_by_supervised:
        return_intersection_data_frame(papers_from_table(all_algorithms_by_supervised[key_cat1]), papers_from_table(all_threats), key_cat1 + " - Behavior")
        setsxs = [] 
        setsys = [] 
        name_plots = []      
        for key_cat2 in all_threats_by_detection: 
            return_intersection_data_frame(papers_from_table(all_algorithms_by_supervised[key_cat1]), papers_from_table(all_threats_by_detection[key_cat2]), key_cat1 + " - " + key_cat2)
            setsxs.append(papers_from_table(all_algorithms_by_supervised[key_cat1]))
            setsys.append(papers_from_table(all_threats_by_detection[key_cat2]))
            name_plots.append(key_cat1 + " - " + key_cat2) 
        #return_intersection_data_frames(setsxs, setsys, name_plots, key_cat1 + " Behavior Multi")
        merge_images(name_plots, key_cat1 + " Behavior Multi")

    setsxs = []
    setsys = []
    name_plots = []   
    for key_cat2 in all_threats_by_detection:
        return_intersection_data_frame(papers_from_table(all_algorithms), papers_from_table(all_threats_by_detection[key_cat2]), "Algorithm - " + key_cat2)
        setsxs.append(papers_from_table(all_algorithms)) 
        setsys.append(papers_from_table(all_threats_by_detection[key_cat2]))
        name_plots.append("Algorithm - " + key_cat2)
    #return_intersection_data_frames(setsxs, setsys, name_plots, "Algorithm Behavior Multi")
    merge_images(name_plots, "Algorithm Behavior Multi")

    return_intersection_data_frame(papers_from_table(all_position_speed_course), papers_from_table(all_threats), "Variables - Behavior")
    setsxs = []
    setsys = []
    name_plots = [] 
    for key_cat in all_threats_by_detection: 
        return_intersection_data_frame(papers_from_table(all_position_speed_course), papers_from_table(all_threats_by_detection[key_cat]), "Variables - " + key_cat)
        setsxs.append(papers_from_table(all_position_speed_course))
        setsys.append(papers_from_table(all_threats_by_detection[key_cat2]))
        name_plots.append("Variables - " + key_cat)
    #return_intersection_data_frames(setsxs, setsys, name_plots, "Variables Behavior Multi")
    merge_images(name_plots, "Variables Behavior Multi")

def get_pby(dict_y, indicator_all = False): 
    pby = dict()
    for c in years_from_table(dict_y, True):
        for y in years_from_table(dict_y, True)[c]:
            if y not in pby:
                pby[y] =0
            pby[y] += years_from_table(dict_y, True)[c][y]
    for y in sorted(list(pby.keys())):
        print(y, pby[y])
    if indicator_all:
        for y in sorted(list(pby.keys())):
            all_for_year = dict()
            for keyds in dict_y:
                for ref in dict_y[keyds]:
                    if paper_details[ref]["Year"] == y:
                        all_for_year[ref] = paper_details[ref]["Author"]
            sorted_vals = dict(sorted(all_for_year.items(), key=lambda item: len(item[1])))
            start_new = "\multirow{" + str(int(np.ceil(pby[y] / 2))) + "}{*}{" + str(y) + "}"
            list_ref = list(sorted_vals.keys())
            allstr = ""
            for ref_id in range(int(np.ceil(len(list_ref) / 2))):
                ref1 = list_ref[ref_id]
                ref2 = list_ref[- 1 - ref_id]
                p1 = paper_details[ref1]["Author"] + " \\cite{" + ref1 +"}"
                p2 = paper_details[ref2]["Author"] + " \\cite{" + ref2 +"}"
                if p1 != p2:
                    newstr = start_new + " & " + p1 + " & " + p2 + " \\\\\n"
                else:
                    newstr = start_new + " & " + p1 + " & \\\\\n"
                start_new = ""
                allstr += newstr
            print(allstr + "\\hline")
    print(sum(list(pby.values())))
        
get_pby(all_data, True)
#get_pby(all_position_speed_course)
#get_pby(all_threats)
for key_cat in all_threats_by_detection: 
    print(key_cat)
    get_pby(all_threats_by_detection[key_cat])
#get_pby(all_algorithms)
for key_cat in all_algorithms_by_supervised: 
    print(key_cat)
    get_pby(all_algorithms_by_supervised[key_cat]) 
 
def seen_in_both(dicts_dic): 
    seen_dict = dict()
    for key_cat in dicts_dic:
        for cat in dicts_dic[key_cat]:
            for r in dicts_dic[key_cat][cat]:
                if r not in seen_dict:
                    seen_dict[r] = set()
                seen_dict[r].add(key_cat)
    for r in seen_dict:
        if len(seen_dict[r]) > 1:
            print(r, seen_dict[r])
seen_in_both(all_threats_by_detection)
seen_in_both(all_algorithms_by_supervised)
