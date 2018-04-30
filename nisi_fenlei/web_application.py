import os.path
import random
import re
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
from NCL_Mapping import nice_mapping
from NCL_Mapping import search_change
from NCL_Mapping import get_modify_time
from NCL_Mapping import get_all_version
from NCL_Mapping import show_modify
from NCL_Mapping import show_nice
from NCL_Mapping import version_mapping


find_year = re.compile('\d{4}')
all_versions = get_all_version()
all_versions = sorted(all_versions, key=lambda x: find_year.findall(x)[0])

define("port", default=8000, help="run on the given port", type=int)

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

class NiceHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('mapping.html', search_text=None, results=None, modify_info=None)
    def post(self):
        search_text = self.get_argument('search_text')
        # year = self.get_argument('year')
        if search_text:
            code, name = parse_args(search_text)
            results = nice_mapping(code, name)
            modify_info = search_change(search_text)
            self.render('mapping.html', search_text=search_text, results=results, modify_info=modify_info)
        else:
            self.render('mapping.html', search_text=None, results=None, modify_info=None)


class MappingHandler(tornado.web.RequestHandler):
    def get(self):
        # all_versions = get_all_version()
        self.render('show_data.html', all_versions=all_versions, class_title=None, title=None, third=None)
    def post(self):
        # all_versions = get_all_version()
        version1 = self.get_argument("version1")
        version2 = self.get_argument("version2")
        results = version_mapping(version1, version2)
        class_title, title, third = results
        self.render('show_data.html', all_versions=all_versions, class_title=class_title, title=title, third=third)

class ModifyPageHandler(tornado.web.RequestHandler):
    def post(self):
        modify_times = get_modify_time() 
        select_modify_time = self.get_argument("select_modify_time")
        if select_modify_time:
            self.render("show_modify.html", modify_times=modify_times, select_modify_time=select_nice_time)
        self.render("show_modify.html", modify_times=modify_times)    


def main():
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[
                (r"/", MappingHandler),
                (r"/show_modify", MappingHandler),
                (r"/show_nice", NiceHandler),
            ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True,)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()        

if __name__ == "__main__":
    main()