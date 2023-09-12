from collections import Counter

n = 9
m = 34

lis = [6, 6, 6, 8, 8, 8, 5, 5, 1]
for idx, i in enumerate(lis):
    lis[idx] = int(i)
c = Counter(lis)
# print(c)

pre = []
las = []
for k, v in c.items():
    if v >= 3:
        pre.append(k)
    if v >= 2:
        las.append(k)
if len(pre)==0 or len(las) == 0:
    print("0, 0")
else:
    res = {}
    pres = 0
    lass = 0
    for i in pre:
        pres = 3*i
        for j in las:
            if j!=i:
                lass = 2* j
                res[(i, j)] = pres+lass
    curpre = 0
    curlas = 0
    # print(res)
    for k,v in res.items():
        if v>m:
            continue
        else:
            # print(k)
            if k[0] > curpre:
                curpre = k[0]
                curlas = k[1]
    print(str(curpre) + ", " + str(curlas))


