# -*- coding: UTF-8 -*-
import os
import sys
import re
import random
import xml.etree.ElementTree as ET
import zipfile

local_path = os.getcwd()
nice_path = os.path.join(local_path, "nice_classes_ecah_version")

"""
  NCL Description：
  第一层是商品和服务类别,中文第一类, 第二类…表示,共45个类别, 
  第二层是商品和服务类似群,代码
  采用四位数字，前两位数字表示商品和服务类别，后面两位数字表示类似群号，如“0304”即表示表示第三类商品的第4类似群；
  第三层是商品和服务项目，代码采用六位数字，前两位表示商品和服务类别，后面四位数字为商品或服务项目编码，如“120092”为第十二类第92号商品，六位数字前面加“C”的代码表示未列入《国际分类》的我国常用商品和服务项目，如“C120008”为国内第十二类第8号商品。
  第四层的代码用中文〔一〕、〔二〕……表示各类似群中的某一部分；
  第五层的代码在各类似群后面的“注”中出现，用1、2……去分各条说明。


  Files Format：
  第一类:用于工业、科学、摄影、农业、园艺和林业的化学品；未加工...

"""

"""
    Data Structure:

      before:
      before = [nice_classes_dict, nice_first_classes, nice_second_classes, nice_third_classes]
      nice_classes_dict = {'第一类'：{('工业气体，单质', '0101'): [("氨*",'010061'), ...]}, ...}
      nice_first_classes = {'第一类':'用于工业、科学、摄影、农业、园艺和林业的化学品...', ...}
      nice_second_classes = {'0101':'工业气体，单质', ...}
      nice_third_classes = {'氨*','010061', ...}

"""


replace_str = ["（一）", "（二）", "（三）", "（四）", "（五）", "（六）", "（七）", "（八）", "（九）",
"（十）", "（十一）", "（十二）", "（十三）", "（十四）", "（十五）", "（十六）", "（十七）", "（十八）",
"（十九）", "（二十）", "（移入45类）", "【", "】", "。", "[", "]"]


remove_str = "（删除）"

jugment_str = ["*", "※"]

nsmap = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}


def qn(tag):
    """
    Stands for 'qualified name', a utility function to turn a namespace
    prefixed tag name into a Clark-notation qualified tag name for lxml. For
    example, ``qn('p:cSld')`` returns ``'{http://schemas.../main}cSld'``.
    Source: https://github.com/python-openxml/python-docx/
    """
    prefix, tagroot = tag.split(':')
    uri = nsmap[prefix]
    return '{{{}}}{}'.format(uri, tagroot)


def xml2text(xml):
    """
    A string representing the textual content of this run, with content
    child elements like ``<w:tab/>`` translated to their Python
    equivalent.
    Adapted from: https://github.com/python-openxml/python-docx/
    """
    text = u''
    root = ET.fromstring(xml)
    for child in root.iter():
        if child.tag == qn('w:t'):
            t_text = child.text
            text += t_text if t_text is not None else ''
        elif child.tag == qn('w:tab'):
            text += '\t'
        elif child.tag in (qn('w:br'), qn('w:cr')):
            text += '\n'
        elif child.tag == qn("w:p"):
            text += '\n\n'
    return text


def process(docx, img_dir=None):
    text = u''

    # unzip the docx in memory
    zipf = zipfile.ZipFile(docx)
    filelist = zipf.namelist()

    # get header text
    # there can be 3 header files in the zip
    header_xmls = 'word/header[0-9]*.xml'
    for fname in filelist:
        if re.match(header_xmls, fname):
            text += xml2text(zipf.read(fname))

    # get main text
    doc_xml = 'word/document.xml'
    text += xml2text(zipf.read(doc_xml))

    # get footer text
    # there can be 3 footer files in the zip
    footer_xmls = 'word/footer[0-9]*.xml'
    for fname in filelist:
        if re.match(footer_xmls, fname):
            text += xml2text(zipf.read(fname))
    zipf.close()
    return text.strip()

    
def spcial_data(third_class):
    if "*" or "※" in third_class:
        return "".join(third_class.split())

def split_nice(text):
    find_num_code = re.compile(r'\d{4,6}|[A-Z][0-9]+')
    find_special = re.compile(r'（(.+)）')
    sub_str = re.compile(r' 。|\.| |  |   |    |')
    code = find_num_code.findall(text)

    spcial_data = find_special.findall(text)
    if spcial_data and spcial_data[0] in "，" and code:
        print("spcial line: %s"% text)
    text = text.replace(remove_str, "")
    for s in replace_str:
        if s in text:
            text = text.replace(s, "")
            # print("replace:",text,s)

    if code and len(code[0])>3:
        code = code[0]
        item = text.replace(code, "")
        # for item in text.split(code):
        if item:
            # name = item.replace("     ", "")
            # name = name.replace("    ", "")
            # name = name.replace("   ", "")
            # name = name.replace("  ", "")
            # name = name.replace(" ", "")
            name = sub_str.sub("",item.strip())
        return code, name
        
    
def get_nice(text):
    find_num_code = re.compile(r'\d+|[A-Z][0-9]+')
    third_nice = []
    if split_nice(text):
        code, name = split_nice(text)
        if len(code)==4:
            return code, name
        elif len(code)>=6:
            for third in text.split("，"): 
                if split_nice(third):
                   name, code = split_nice(third)
                   name = spcial_data(name)
                   third_nice.append((name, code))
                else:
                    print("spcial line: {} -- {}".format(text, third))
                spcial_if = find_num_code.findall(name)
                if spcial_if and len(spcial_if)>3:
                    print("wrong line %s"%name)

            return third_nice
    else:
        return None


def get_first(text):
     if "第" and "类" in text and ":" in text:
        first_class, name_class, = text.split(":")
        print(first_class)
        return first_class, name_class

def remove_ilege(files_path):
    for fi in os.listdir(files_path):
        if "result" not in fi:
            save_path = os.path.join(files_path, "result")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, fi)
            with open(os.path.join(files_path,fi),encoding='utf-8') as f, open(save_file,'w',encoding='utf-8') as sf:
                for line in f:
                    if line.endswith('，  '):
                        sf.write(lines[:-1]+"\n")
                    else:
                        sf.write(line)

def extract_8th_edition_classes(nice_8th_file):
    nice_8th_classes_dict = {}
    return nice_8th_classes_dict



def extract_9th_edition_classes():
    nice_9th_classes_dict = {}

    return nice_9th_classes_dict

def extract_10th_edition_classes():
    nice_10th_classes_dict = {}

    return nice_10th_classes_dict

def extract_11th_edition_classes():
    nice_11th_classes_dict = {}

    return nice_11th_classes_dict



def save(nice_classes_dict, file_name):
    """"""
    with open(file_name, 'w') as f:
        for key in nice_classes_dict.keys():
            pass



def judgment_niceName(id, before_class_name, after_class_name):
    """"""
    return True if before_class_name==after_class_name else False


def search_nice(year, nice_code):
    """search nice code map to new code"""
    pass



# def main(data_path, num_thread):
#     path = '/'.join(data_path.split('/')[:-1])
#     save_path = os.path.join(path,'results')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     # Thread Assignment Task
#     # files_space = np.linspace(0,len(all_files),num_thread+1,dtype=np.int)

#     # thread begin
#     workers = [threading.Thread(target=process_data, args=(files_list, save_path)) for _ in num_thread]
#     for thread in workers:
#         # thread.daemon = True  # make interrupting the process with ctrl+c easier #设置线程为后台线程
#         thread.start()

#     # 阻塞线程
#     for thread in workers:
#         thread.join()
#     print("all files finished! results save in %s"%save_path)


def word2txt(nice_word, save_file):
    text = process(nice_word)
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(text)

def extract_nice_from_txt(nice_txt):
    nice_dict = {}
    first_class_dict = {}
    second_class_dict = {}
    # first_second_class = {}
    sub_symbol = re.compile(r'。|\.| |  |    ')
    with open(nice_txt, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            try:
                if not line or "本类新增项目" in line: continue
                if "类似" in line or "近似" in line: continue
                if "移来" in line or "交叉检索" in line: continue
                if "注：" in line or "移入" in line:continue
                # if "###" in line: continue
                if "第" and "类" in line and ":" in line:
                    first_class, name_class, = line.split(":")
                    # print("\n")
                    # print(first_class, "***********")
                    first_class_dict[first_class] = sub_symbol.sub("", name_class)   
                    if first_class not in nice_dict:
                        nice_dict[first_class] = {}
                if get_nice(line):
                    result = get_nice(line)
                    if not isinstance(result, list):
                        second_code, second_name = result
                        # print(second_name, second_code)

                        # second_name = second_name
                        # print("process {}-{}".format(me, second_code))
                        second_class_dict[second_code] = sub_symbol.sub("", second_name.strip())
                        nice_dict[first_class][second_code] = []
                        if not second_name:
                            print("miss: {} {}".format(line, second_code))
                    elif isinstance(result, list):
                        # print(result)
                        nice_dict[first_class][second_code].extend(result)
                        
                    else:
                        print("except line %s"%line)
                        pass
            except Exception as e:
                print(line)
                raise e
    return nice_dict, first_class_dict, second_class_dict



def get_nice_8th2002():
    nice_8th2002_file = os.path.join(nice_path, "第八版(2002).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_8th2002_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_9th2007():
    nice_9th2007_file = os.path.join(nice_path, "第九版(2007).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_9th2007_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_10th2012():
    nice_10th2012_file = os.path.join(nice_path, "第十版(2012).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_10th2012_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_10th2013():
    nice_10th2013_file = os.path.join(nice_path, "第十版(2013).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_10th2013_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_10th2014():
    nice_10th2014_file = os.path.join(nice_path, "第十版(2014).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_10th2014_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_10th2015():
    nice_10th2015_file = os.path.join(nice_path, "第十版(2015).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_10th2015_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_10th2016():
    nice_10th2016_file = os.path.join(nice_path, "第十版(2016).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_10th2016_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_11th2017():
    nice_11th2017_file = os.path.join(nice_path, "第十一版(2017).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_11th2017_file)
    return nice_dict, first_class_dict, second_class_dict

def get_nice_11th2018():
    nice_11th2018_file = os.path.join(nice_path, "第十一版(2018).txt")
    nice_dict, first_class_dict, second_class_dict = extract_nice_from_txt(nice_11th2018_file)
    return nice_dict, first_class_dict, second_class_dict


# 各版本nice分类（收集到的）
def load_nice_version_data():
    nice_8th2002_dict, first_8th2002_dict, second_8th2002_dict = get_nice_8th2002()
    nice_9th2007_dict, first_9th2007_dict, second_9th2007_dict = get_nice_9th2007()
    nice_10th2012_dict, first_10th2012_dict, second_10th2012_dict = get_nice_10th2012()
    nice_10th2013_dict, first_10th2013_dict, second_10th2013_dict = get_nice_10th2013()
    nice_10th2014_dict, first_10th2014_dict, second_10th2014_dict = get_nice_10th2014()
    nice_10th2015_dict, first_10th2015_dict, second_10th2015_dict = get_nice_10th2015()
    nice_10th2016_dict, first_10th2016_dict, second_10th2016_dict = get_nice_10th2016()
    nice_11th2017_dict, first_11th2017_dict, second_11th2017_dict = get_nice_11th2017() 
    nice_11th2018_dict, first_11th2018_dict, second_11th2018_dict = get_nice_11th2018()

def load_all_version_data():
    all_nice_version_dict = {}
    find_year = re.compile('\d{4}')
    for f in sorted(os.listdir(nice_path), key=lambda x: find_year.findall(x)[0]):
        print(f)
        if "2002" in f:
            version_number = f.replace(".txt","")
            # nice_8th2002_dict, first_8th2002_dict, second_8th2002_dict = get_nice_8th2002()
            # all_nice_version_dict["2002"] = get_nice_8th2002()
            all_nice_version_dict[version_number] = get_nice_8th2002()
        elif "2007" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2007"] = get_nice_9th2007()
            all_nice_version_dict[version_number] = get_nice_9th2007()
        elif "2012" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2012"] = get_nice_10th2012()
            all_nice_version_dict[version_number] = get_nice_10th2012()
        elif "2013" in f:
            version_number = f.replace(".txt","")
            all_nice_version_dict[version_number] = get_nice_10th2013()
            # all_nice_version_dict["2013"] = get_nice_10th2013()
        elif "2014" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2014"] = get_nice_10th2014()
            all_nice_version_dict[version_number] = get_nice_10th2014()
        elif "2015" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2015"] = get_nice_10th2015()
            all_nice_version_dict[version_number] = get_nice_10th2015()
        elif "2016" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2016"] = get_nice_10th2016()
            all_nice_version_dict[version_number] = get_nice_10th2016()
        elif "2017" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2017"] = get_nice_11th2017()
            all_nice_version_dict[version_number] = get_nice_11th2017()
        elif "2018" in f:
            version_number = f.replace(".txt","")
            # all_nice_version_dict["2018"] = get_nice_11th2018()
            all_nice_version_dict[version_number] = get_nice_11th2018()
    return all_nice_version_dict

if __name__ == '__main__':
    # main()
    # print(load_all_version_data())
    # get_nice_10th2012()
    nice_dict, first_dict, second_dict = get_nice_9th2007()
    print(nice_dict["第一类"]['0102'])
    for i in nice_dict["第一类"]:
        print(nice_dict["第一类"][i])
    # a = os.listdir(nice_path)
    # find_year = re.compile('\d{4}')
    # print(sorted(a,key=lambda x: find_year.findall(x)[0]))

    # sort(key=lambda x: int(find_year.findall(x)[0]))