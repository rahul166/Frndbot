###code clean up task is there to be done
##Also this X,y,decode_target needed to be saved as .npy so model can use them 



import pandas as pd 

import numpy as np

data = pd.read_csv('./fr.txt', sep='\t')
data

lst_sc_ids=data['scene_id']
lst_sc_ids

# pd.set_option('display.max_colwidth', -1)
# tt_dic={}
whole_data=[]
x_prev=[]
y_prev=[]
for i in range(1, 401):
    sc_data=data[data['scene_id']==str(i)]
    le=len(sc_data)
    val=sc_data['line'].values
    if len(val)<=13:
      continue
    whole_data.append(val)
    x_prev.append(val[0:le-1])
    y_prev.append(val[le-1])


print(x_prev[2])
print(y_prev[2])

#chars to id mapping
text=" "
for arr in whole_data:
  for i in arr:
    text+=i
    
chars = sorted(list(set(text)))
chars_size = len(chars)

chars2id = dict((c, i) for i, c in enumerate(chars))
chars2id

pd.set_option('display.max_colwidth', -1)

# vectorize our data 
len_per_section = 50
skip = 50
sections = []
next_chars = []
# print(len(x_prev))
for i in range(0, len(x_prev)):
    sections.append(x_prev[i])
    next_chars.append(y_prev[i]) 
print(next_chars)  

len_sections_y=len(max(next_chars,key=len))


len_sections_x=0
size_list=[]
for i in sections:
  size_list.append(i.shape[0])
  max_legth=len(max(i,key=len))
  if len_sections_x<max_legth:
    len_sections_x=max_legth
print(len_sections_x)
num_sentences=min(size_list)
print(num_sentences)


# vectorize our chars 
y = np.zeros((len(sections), len_sections_y, chars_size))
X = np.zeros((len(sections),num_sentences,len_sections_x,chars_size))

for i, section in enumerate(next_chars):
    for j, char in enumerate(section):

        y[i][j][chars2id[char]] = 1
for i,evry in enumerate(sections):
  for j,section in enumerate(evry[:num_sentences]):
    for k ,char in enumerate(section):
      X[i][j][k][chars2id[char]]=1
# print("X :", X)
# print("y:", y)

print("X:",X.shape,"Y:",y.shape)
# y.shape
# print("y:", y[1][15][700])




###################################################################



# vectorize our data 
len_per_section = 50
skip = 50
sections = []
next_chars = []

sec_length= len(x_prev)
# print(len(x_prev))
for i in range(0, len(x_prev)):
    sections.append(x_prev[i])
    next_chars.append(y_prev[i]) 
print(next_chars)  

len_sections_y=len(max(next_chars,key=len))
# max_sente_arr=max(next_chars)
# print(max_sente)


len_sections_x=0
size_list=[]
for i in sections:
  size_list.append(i.shape[0])
#   print(len(max(i,key=len)))
  max_legth=len(max(i,key=len))
  if len_sections_x<max_legth:
    len_sections_x=max_legth
print(len_sections_x)
num_sentences=min(size_list)
print(num_sentences)
# vectorize our chars 
y = np.zeros((len(sections), len_sections_y, chars_size))
X = np.zeros((len(sections),num_sentences,len_sections_x,chars_size))
decode_target=np.zeros((len(sections),len_sections_y,chars_size))
# print(X)
# print(X.shape)
# print(X[1][1])

for i, section in enumerate(next_chars):
    for j, char in enumerate(section):
#       if j>3:
#         break
#         print(j,char)
        y[i][j][chars2id[char]] = 1
        if j>0:
          decode_target[i][j-1][chars2id[char]]=1          



for i,evry in enumerate(sections):
  for j,section in enumerate(evry[:num_sentences]):
    for k ,char in enumerate(section):
      X[i][j][k][chars2id[char]]=1

      
    
    
    
print(decode_target)
print("x:",X.shape,"y:",y.shape,"D:",decode_target.shape)    

