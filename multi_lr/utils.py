import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch




if __name__ == "__main__":
    col_pic = 2
    row_pic = 4
    x_size = 15
    y_size = 15
    B = 3

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize((x_size, y_size))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    data = torch.zeros(B, x_size * col_pic, y_size * row_pic)
    labels = []

    for t in range(B):
        pic = torch.zeros(x_size * col_pic, y_size * row_pic)
        label = torch.zeros(10)
        for i in range(col_pic):
            for j in range(row_pic):
                pic[x_size*i:x_size*(i+1), y_size*j:y_size*(j+1)] = train_dataset[t*4+i*2+j][0][0]
                label[train_dataset[t*4+i*2+j][1]] = 1
        data[t] = pic
        labels.append(label)



