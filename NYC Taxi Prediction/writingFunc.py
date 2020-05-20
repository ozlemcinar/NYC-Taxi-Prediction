import sys
import csv
#used this func to pass outputs from newcsv file to html as points

flag = False
if len(sys.argv)>2:
    if sys.argv[2]=='-raw':
        flag = True

with open(sys.argv[1], 'rb') as f:
    reader = csv.reader(f, dialect='excel')
    for row in reader:
        if flag:
            print("%s %s" % (row[2], row[1]))
        else:
            print ("{lat:%s, lng:%s}," % (row[2], row[1]))
