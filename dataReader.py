import sys

def dataReader(filename):
    dataset = []
    with open(filename, 'r') as reader:
        for i, line in enumerate(reader.readlines()):
            sys.stdout.write("\r Loading {0}%".format(round(i/30,2)))
            row = line.strip().split(' ')
            row = list(map(int, row))
            dataset.append(row[1:])
    return dataset

# dataset = dataReader("P53.ds")
# print(len(dataset))
# print(len(dataset[0]))
