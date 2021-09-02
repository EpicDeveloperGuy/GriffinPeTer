import csv

peter = ''

with open('Family_Guys.csv', newline='\n') as fg:
    reader = csv.reader(fg)
    for row in reader:
        if (row[0] == 'Peter'):
            row[1] = "".join(c for c in row[1] if ord(c)<128)
            peter = peter + row[1] + '\n'

with open('peter.data', 'w') as pd:
    pd.write(peter)