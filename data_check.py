from numpy import load

data = load('data_cifar100.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])