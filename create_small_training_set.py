import csv

list = []
with open('./data/train_r6_test4.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    ii = 0;
    max = 2

    for row in reader: # one doc
        content = row[0]
        topicLabel = row[1]
        topicName = row[2]
        if ii == 0:
            prev_name = topicName
        if ii < max:
            list.append(row)
        ii += 1
        if topicName != prev_name:
            ii = 0
        prev_name = topicName

print (len(list))

with open("./data/train_small.csv",'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for line in list:
        writer.writerow(line)