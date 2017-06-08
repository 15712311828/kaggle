import csv
from functools import cmp_to_key
def loadTrainFile():
    l=[];
    with open('train.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    label=[]
    data=[]
    for i in l:
        label.append(i[0])
        data.append(i[1:])
    return [[int(i) for i in j] for j in data],[int(i) for i in label]
def normalize(list):
    m=len(list)
    n=len(list[0])
    for i in range(m):
        for j in range(n):
            if(list[i][j]!=0):
                list[i][j]=1
    return
def loadTestFile():
    l=[];
    with open('test.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    return [[int(i) for i in j] for j in l]
def knn(data,label,testVector,k):
    res=[]
    m=len(data)
    n=len(data[0])
    for i in range(m):
        distance=0;
        for j in range(n):
            if(data[i][j]!=testVector[j]):
                distance=distance+1;
        res.append([distance,label[i]])
    res.sort(key = lambda a:a[0])
    labels=[]
    for i in range(10):
        labels.append(0)
    for i in range(k):
        labels[res[i][1]]=labels[res[i][1]]+1;
    return labels.index(max(labels))
def writeCsv(l):
    with open('res.csv', 'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImageId','Label'])
        j=1;
        for i in l:
            writer.writerow([j,i])
            j=j+1;
    return
data,label=loadTrainFile()
normalize(data)
testData=loadTestFile()
normalize(testData)
j=0
res=[]
for i in testData:
    a=knn(data,label,i,20)
    res.append(a)
    j=j+1
    print(j)
writeCsv(res)

input("push any key to exit")
