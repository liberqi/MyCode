import re
import os
import sys

local_path = os.getcwd()
modify_path = os.path.join(local_path, "nice_modify")
def extract_modify_info(modify_file):
    modify_dict = {}
    find_num_code = re.compile(r'^\d+')
    # sub_str = re.compile(r'\d{1}\.|“|”|第 \d{1} 段|第.+部分，|')
    sub_part = re.compile(r'| ')
    # sub_str = re.compile(r'\d{1}\.|第 \d{1} 段|第.+部分，|第.{3,4}部分')
    sub_str = re.compile(r'\d{1}\.|')
    find_special = re.compile(r'\d{1,2} \d{4} ')
    # skip_str = re.compile(r' \d{1,2} / \d{1,2} ')
    with open(modify_file, encoding='utf-8') as f:
        for line in f:
            line_list = line.split("。")
            # print(line_list)
            if "位置" in line or "修改内容" in line:continue
            for part in line_list:
                if "注释" in part: continue
                # part = part.replace(" ", "")
                part = sub_part.sub("", part)
                part = part.strip("\n")
                part = sub_str.sub("", part).strip()
                # print(part)
                codes = find_num_code.findall(part)
                # print(codes)
                if codes and len(codes[0])==4:
                    code = codes[0]
                elif "标题" in part or "类别标题" in part:
                    code = "标题"
                if codes and len(codes[0])<=2:
                    # print(part)
                    # print("类 %s" % codes[0])
                    pass
                    # modify_dict[codes[0]] = {}
                if 2<len(part)<=4:
                    if code not in modify_dict:
                        modify_dict[code] = []
                        # print("add code %s" % code)
                    # elif "类别标题" in part:
                    #     code = part.strip()
                        modify_dict[code] = []
                    else:
                        # print("repeat %s" % code)
                        pass
                    # print(codes)
                elif len(part)>4:
                    # print(part)
                    c = find_special.findall(part)
                    if c:
                        # print(part, c)
                        code = c[0].strip()[-4:]
                        # print("spcial %s" % code)
                    if code not in modify_dict:
                        modify_dict[code] = []
                        # print("add code %s" % code)
                    # elif "标题" in part:
                    #     modify_dict["类别标题"] = []
                    if part.startswith(code):
                        modify_dict[code].append(part.replace(code,"",1).replace(" ",""))
                    else:
                        modify_dict[code].append(part.replace(" ",""))
    return modify_dict


def load_modify_data():
    # load each modify information in dict
    find_time = re.compile(r'\d{4}-\d{1,2}')
    all_modify = {}
    find_year = re.compile('\d{4}')
    for f in sorted(os.listdir(modify_path), key=lambda x: find_year.findall(x)[0]):
        print(f)
        time = find_time.findall(f)[0]
        file_path = os.path.join(modify_path, f)
        if "2013" in f:
            modify_2013th_dict = extract_modify_info(file_path) 
            # all_modify["2013"] = modify_2013th_dict
            all_modify[time] = modify_2013th_dict
        if "2014" in f:
            modify_2014th_dict = extract_modify_info(file_path)
            # all_modify["2014"] = modify_2014th_dict
            all_modify[time] = modify_2014th_dict
        if "2015" in f:
            modify_2015th_dict = extract_modify_info(file_path)
            # all_modify["2015"] = modify_2015th_dict
            all_modify[time] = modify_2015th_dict
        if "2016" in f:
            modify_2016th_dict = extract_modify_info(file_path)
            # all_modify["2016"] = modify_2016th_dict
            all_modify[time] = modify_2016th_dict
        if "2017" in f:
            modify_2017th_dict = extract_modify_info(file_path)
            # all_modify["2017"] = modify_2017th_dict
            all_modify[time] = modify_2017th_dict
        if "2018" in f: 
            modify_2018th_dict = extract_modify_info(file_path)
            # all_modify["2018"] = modify_2018th_dict
            all_modify[time] = modify_2018th_dict

    return all_modify

if __name__ == '__main__':
    # main()
    load_modify_data()