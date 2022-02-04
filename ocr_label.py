import os

file_path='D:/Library/deep-text-recognition-benchmark/data'

data=[]
all_data=[]
with open(os.path.join(file_path,'gt.txt'),'r',encoding='utf-8') as f:
    lines=f.readlines()
    for line in lines:
        if not line:
            break
        all_data.append(line.strip()) 
        tmp=line.split('\\')
        raw=tmp[1].split('.')
        if '-' in raw[0]:
            raw=raw[0].split('-')
            data.append(raw[0])
        else:
            data.append(raw[0])
        
with open(os.path.join(file_path,'gt2.txt'),'w',encoding='utf-8') as f:
    for ix,line in enumerate(all_data):
        if not line:
            break
        f.write(line+'\t'+data[ix]+'\n')