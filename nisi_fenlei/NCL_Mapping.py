# -*- coding: UTF-8 -*-
import os
import sys
import re
# import threading
import difflib
import csv
from load_nice_modify_info  import load_modify_data
from load_each_nice_version_info import load_all_version_data

"""
    all_modify = {"year_time": modify_info}
    scuh as all_modify = {"2013": {"0101":[...,]}, "2014":{"0101":[...,]}}
"""
all_modify = load_modify_data()

"""
all_nice_version_dict = {"2013-10":(nice_8th2002_dict, first_8th2002_dict, second_8th2002_dict),...}
nice_8th2002_dict = {"第一类":{"0101":[氨010061,无水氨010066,...]}}
first_8th2002_dict = {"第一类":用于工业、科学、摄影、农业、园艺、森林的化学品..,...}
second_8th2002_dict = {"0101":"工业气体，单质",...}
"""
all_nice_version_dict = load_all_version_data()




def nice_mapping(code, name):
    results = []
    for time in all_nice_version_dict:
        # print("search %s"% time)
        nice_dict, first_dict, second_dict = all_nice_version_dict[time]
        for first in nice_dict:
            # print(nice_dict[first].keys())
            for second in nice_dict[first]:
                for third_code, third_name in nice_dict[first][second]:
                    # print(third_code, third_name)
                    if code and name:
                        if code == third_code or name in third_name:
                            results.append((time, first, first_dict[first], second, 
                                second_dict[second], third_code, third_name))
                            # print("time :{}, position:{}:{},{}:{},({},{})".format(time, first, 
                                # first_dict[first], second, second_dict[second], third_code, third_name))
                    elif code and not name:
                        if code == third_code:
                            results.append((time, first, first_dict[first], second, 
                                second_dict[second], third_code, third_name))
                            # print("time :{}, position:{}:{},{}:{},({},{})".format(time, first, 
                                # first_dict[first], second, second_dict[second], third_code, third_name))
                    elif not code and name:
                        if name == third_name:
                            results.append((time, first, first_dict[first], second, 
                                second_dict[second], third_code, third_name))
                            # print("time :{}, position:{}:{},{}:{},({},{})".format(time, first, 
                                # first_dict[first], second, second_dict[second], third_code, third_name))
    return results


def search_change(text):
    # text == code or name
    changed = []
    for time in all_modify:
        # print(time)
        for second_pos in all_modify[time]:
            for modify_info in all_modify[time][second_pos]:
                if text in modify_info or text in all_modify[time][second_pos]:
                    # print("时间：{}, 类别位置：{},修改内容：{}".format(time,second_pos,modify_info))
                    changed.append((time, second_pos, modify_info))
    return changed

# 展示数据 根据用户选择的时间
def get_all_version():
    # version_time
    return [version for version in all_nice_version_dict]

def show_nice(select_time):
    Nice_data = {}
    for time in all_nice_version_dict:
        # print(time)
        if select_time == time:
            return all_nice_version_dict[time]

def get_modify_time():
    # modify_time
    return [time for time in all_modify]

def show_modify(select_time):
    Modify =  {}
    Modify["删除"] = []
    Modify["修改"] = []
    Modify["新增"] = []
    Modify["移动"] = []
    for time in all_modify:
        # print(time)
        if select_time == time:
            for second_pos in all_modify[time]:
                for modify_info in all_modify[time][second_pos]:
                    if "删除" in modify_info:
                        Modify["删除"].append(modify_info)
                    elif "改为" in modify_info or "修改" in modify_info:
                        Modify["修改"].append(modify_info)
                    elif "新增" in modify_info or "增加" in modify_info:
                        Modify["新增"].append(modify_info)
                    elif "移入" in modify_info:
                        Modify["移动"].append(modify_info)
            return Modify

def parse_args(text):
    text = text.strip()
    sub_str = re.compile(r' |  |   |    |')
    find_num_code = re.compile(r'\d{4,6}|[A-Z][0-9]{7}')
    match_code  = find_num_code.findall(text)
    if match_code:
        code = match_code[0]
        name = text.replace(code, "")
        name = sub_str.sub("", name)
        # for item in text.split(code):
        #     if item:
        #         name+=item
        return code, name
    else:
        code = None
        name = sub_str.sub("", text)
        return code, name

def similarity_name(name1, name2):
    # singe_name1 = [i for i in name1 if i]
    # singe_name2 = [i for i in name2 if i]
    return difflib.SequenceMatcher(None, name1, name2).quick_ratio()


# nice分类的增删改判断
def judge_first_name(class1, name_class1, class2, name_class2):
    if first_class1 == first_class2:
        print(" judge class {} {}",format(first_class1, first_class2))
        if name_class1 == name_class2:
            return True
        else:
            print("change class name {} {}".format(name_class1,name_class2))
            return False
def judge_second_name(class1, name_class1, class2, name_class2):
    if first_class1 == first_class2:
        print(" judge class {} {}",format(first_class1, first_class2))
        if name_class1 == name_class2:
            return True
        else:
            print("change class name {} {}".format(name_class1,name_class2))
            return False


# 第三小类更改判断
def judge_third_name(third_class_list1, third_class_list2, position):
    """
        third_class_list1:前版本
        third_class_list2:后版本
    """
    # 类别标题
    third_modified = []
    third_deleted = []
    third_increased = []


    # 查找类别中分类名不同，但分类代码相同的类别
    duplicated_code1 = set()
    duplicated_code2 = set()
    # codes1 = [third_code1 for third_name1, third_code1 in third_class_list1]
    # codes2 = [third_code2 for third_name2, third_code2 in third_class_list2]
    # names1 = [third_name1 for third_name1, third_code1 in third_class_list1]
    # names2 = [third_name2 for third_name2, third_code2 in third_class_list2]
    codes1, names1, codes2, names2 = [], [], [], []
    for third_code1, third_name1 in third_class_list1:
        names1.append(third_name1)
        codes1.append(third_code1)
    for third_code2, third_name2 in third_class_list2:
        names2.append(third_name2)
        codes2.append(third_code2)
    # print(codes2)
    for i, third_code1 in enumerate(codes1):
        if third_code1 in codes1[i+1:]:
            duplicated_code1.add(third_code1)

    for i, third_code2 in enumerate(codes2):
        if third_code2 in codes2[i+1:]:
            duplicated_code2.add(third_code2)

    # mapping
    no_more = []
    for third_code1, third_name1 in third_class_list1:
        # not change 
        if third_code1 in codes2 and third_name1 in names2:
            change = 0
            no_more.append(third_code1)
    change = 1
    for third_code1, third_name1 in third_class_list1:
        # not change 
        if third_code1 in codes2 and third_name1 in names2:
            change = 0
        # modify
        elif third_code1 in duplicated_code1:
            if third_code1 not in no_more and third_code1 in codes2:
                for i, code in enumerate(codes2):
                    if third_code1 == code:
                        third_name2 = names2[i]
                # print("{} change class name {} between {}".format(third_code1, third_name1, third_name2))
                third_modified.append([position, third_code1, third_name1, third_name2])
            elif third_code1 not in codes2:
                # print("Delete {}:{}".format(third_code1, third_name1))
                third_deleted.append([position, third_code1, third_name1])

        elif third_code1 in codes2 and third_name1 not in names2 and third_code1 not in duplicated_code1:
            change =1
            for i, code in enumerate(codes2):
                if third_code1 == code:
                    third_name2 = names2[i]
            # print("{} change class name {} between {}".format(third_code1, third_name1, third_name2))
            third_modified.append([position, third_code1, third_name1, third_name2])
        elif third_code1 not in codes2 and third_name1 in names2 and third_code1 not in duplicated_code1:
            change = 1
            for i, name in enumerate(names2):
                if third_name1 == name:
                    third_code2 = codes2[i]
                    
            # print("{} change class code {} between {}".format(third_name1, third_code1, third_code2))
            third_modified.append([position, third_name1, third_code1, third_code2])
        # delete
        elif third_code1 not in codes2 and third_name1 not in names2 and third_code1 not in duplicated_code1:
            # print("Delete {}:{}".format(third_code1, third_name1))
            third_deleted.append([position, third_code1, third_name1])

    for third_code2, third_name2 in third_class_list2:
        # add
        if third_name2 not in names1 and third_code2 not in codes1 and third_code2 not in duplicated_code1:
            # print("Add {}:{}".format(third_code2, third_name2))
            third_increased.append([position, third_code2, third_name2])
            
    # if not change:
    #     return True
    # else:
    #     return False
    # print(third_modified, third_increased, third_deleted)
    return third_modified, third_increased, third_deleted

def version_mapping(version1, version2):
    all_third_modified = []
    all_third_deleted = []
    all_third_increased = []
    results = []
    find_num = re.compile(r'\d{4}')
    if version1 in all_nice_version_dict and version2 in all_nice_version_dict:
        year1 = int(find_num.findall(version1)[0])
        year2 = int(find_num.findall(version2)[0])
    if year1<year2:
        before_version = version1
        after_version = version2
    else:
        before_version = version2
        after_version = version1

    before_nice, before_first, before_second = all_nice_version_dict[before_version]
    after_nice, after_first, after_sencond = all_nice_version_dict[after_version]
    # 类别标题
    class_title = []
    # print("类别标题")
    # for before_class, after_class in zip(before_first, after_first):
    #     print(before_class, before_class)
    #     if before_first[before_class]!=after_first[after_class]:
    #         print("位置: {} 类别标题 {} 改为 {} ".format(before_class, before_first[before_class], after_first[after_class]))
    #         class_title.append([before_class, before_first[before_class], after_first[after_class]])
    # results.append(class_title)
    for before_class in before_first:
        for after_class in after_first:
            if before_class == after_class:
                print(before_class, before_class)
                if before_first[before_class]!=after_first[after_class]:
                    # print("位置: {} 类别标题 {} 改为 {} ".format(before_class, before_first[before_class], after_first[after_class]))
                    class_title.append([before_class, before_first[before_class], after_first[after_class]])
    results.append(class_title)
    # 标题
    title_increased = []
    title_modified = []
    title_deleted = []
    # print("标题")
    for before_code in before_second:
        if before_code in after_sencond: 
            if before_second[before_code] != after_sencond[before_code]:
                # print("位置: {} 类别标题 {} 改为 {}".format(before_code, before_second[before_code], after_sencond[before_code]))
                title_modified.append([before_code, before_second[before_code], after_sencond[before_code]])
        else:
            # print("位置: {} 删除类别 {} ".format(before_code, before_second[before_code]))
            title_deleted.append([before_code, before_second[before_code]])

    for after_code in after_sencond:
        if after_code not in before_second:
            # print("位置: {} 增加类别 {} ".format(after_code, after_sencond[after_code]))
            title_increased.append([after_code, after_sencond[after_code]])
    


    # 小类
    for before, after in zip(before_nice, after_nice):
        print(before, after)
        for second_before in before_nice[before]:
            for second_after in after_nice[after]:
                if second_before == second_after:
                    if before_nice[before][second_before] and not after_nice[after][second_after]:
                        print("位置: {} 删除类别 {} ".format(second_before, before_second[second_before]))
                        title_deleted.append([second_before, before_second[second_before]])
                        # print(before_nice[before][second_before],after_nice[after][second_after])
                        # continue
                    else:
                        # print(before_nice[before][second_before][0], after_nice[after][second_after][0])
                        # if not before_nice[before][second_before] or not after_nice[after][second_after]:
                        #     print("None", second_before, second_after)
                        result = judge_third_name(before_nice[before][second_before], after_nice[after][second_after], before_code)
                        all_third_modified.extend(result[0])
                        all_third_increased.extend(result[1])
                        all_third_deleted.extend(result[2])
    results.append((title_modified, title_increased, title_deleted))
    results.append((all_third_modified, all_third_increased, all_third_deleted))
    return results



if __name__ == '__main__':

    # print(search_change("060032"))
    # # print(show_modify())
    # nice_mapping(code, name)
    # show_modify("2014")
    # show_nice("2002")
    # nice_mapping("010165", "")
    # code, name = parse_args("对称二苯硫脲 010533")
    # nice_mapping(code, name)
    # str1 = "不属别类的木、软木、苇、藤、柳条、角、骨、象牙、鲸骨、贝壳、琥珀、珍珠母、海泡石 制品，这些材料的代用品"
    # str2 = "不属别类的木、软木、苇、藤、柳条、角、骨、象牙、鲸骨、贝壳、琥珀、珍珠母、海泡石 制品，这些材料的代用品或塑料制品。"
    # print(similarity_name("迪安*", "安迪"))
    # sub_simbol = re.compile(r'、|，|；|')
    # all_time = [i for i in all_nice_version_dict.keys()]
    # for i in range(len(all_time)):
    #     # print("search %s"% time)
    #     if i<len(all_time)-1:
    #         print(all_time[i],all_time[i+1])
    #         nice_dict1, first_dict1, second_dict1 = all_nice_version_dict[all_time[i]]
    #         nice_dict2, first_dict2, second_dict2 = all_nice_version_dict[all_time[i+1]]
    #     for first1, first2 in zip(nice_dict1, nice_dict2):
    #         # str1 = sub_simbol.sub("", first_dict1[first1])
    #         # str2 = sub_simbol.sub("", first_dict2[first2])
    #         # ratio = similarity_name(str1,str2)
    #         # print(first1, first2)
    #         # ratio = similarity_name(first_dict1[first1],first_dict2[first2])
    #         # if ratio<1:
    #         #     print(ratio)
    #         #     print(first_dict1[first1])
    #         #     print(first_dict2[first2])
    #         for second1, second2 in zip(nice_dict1[first1], nice_dict2[first2]):
    #             if second1!=second2:
    #                 # print("*****")
    #                 print(second1, second2)
    #         ratio = similarity_name(second_dict1[second1], second_dict2[second2])
    #         if ratio<1:
    #             print(ratio)
    #             print(second_dict1[second1])
    #             print(second_dict2[second2])

    # print(all_nice_version_dict["第九版(2007)"][0]["第四十五类"]["4506"])
    # local_path = os.getcwd()
    # version_mapping_path = os.path.join(local_path, "version_mapping")
    # if not os.path.exists(version_mapping_path):
    #     os.mkdir(version_mapping_path)
    # with open(version_mapping_path,'w', newline="") as f:
    #     writer = csv.writer(f)
    #     writer.write(["位置", ])
    class_title, title, third = version_mapping("第九版(2007)", "第十版(2012)")
    # print([i for i in all_nice_version_dict])
    # print(title)
    # print(bool(title))
    # for i1, i2, i3 in title:
    # print(bool(title[0]))
# <!-- {% for modified, increased, deleted in title %} -->