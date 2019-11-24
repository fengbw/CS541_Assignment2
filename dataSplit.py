def data_split():
    with open('P53.ds', 'r') as reader:
        for i, line in enumerate(reader.readlines()):
            if i < 3000:
                with open('P53_test.ds', 'w') as writer:
                    writer.write(line)
            if i >= 3000:
                with open('P53_train.ds', 'w') as writer:
                    writer.write(line)
#         row = line.strip().split(' ')
#         row = list(map(int, row))
#         dataset.append(row[1:])
# print(len(dataset[0]))
if __name__ == "__main__":
    data_split()
