def data_split():
    with open('P53.ds', 'r') as reader:
        for i, line in enumerate(reader.readlines()):
            if i < 3000:
<<<<<<< HEAD
                with open('P53_test.ds', 'a') as writer:
                    writer.write(line)
            if i >= 3000:
                with open('P53_train.ds', 'a') as writer:
=======
                with open('P53_test.ds', 'w') as writer:
                    writer.write(line)
            if i >= 3000:
                with open('P53_train.ds', 'w') as writer:
>>>>>>> 092ffb91a5163dad1ef8f994e405595994bdad03
                    writer.write(line)
#         row = line.strip().split(' ')
#         row = list(map(int, row))
#         dataset.append(row[1:])
# print(len(dataset[0]))
if __name__ == "__main__":
    data_split()
