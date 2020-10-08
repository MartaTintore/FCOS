import json
import ast

def main():


    s = open('ms_coco_classnames.txt', 'r').read()
    dic = ast.literal_eval(s)
    with open('ms_coco_classnames.json', 'w') as fp:
        json.dump(dic, fp)
    
if __name__=='__main__':
    main()
