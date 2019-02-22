import math

with open ('log.txt', 'r') as f:
    hey = []
    for line in f.readlines():
        score = line.strip().split(':')[-1]
        hey.append(score)
    fa = open('hey.csv','w+')
    for each in hey:
        fa.write(each+',')
    fa.close()
    print ('done')