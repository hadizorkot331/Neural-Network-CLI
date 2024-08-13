import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

def get_args():
    parser = argparse.ArgumentParser(description="CLI to train deep neural netwroks")
    parser.add_argument("data_dir", help="Directory that contains training data", required=True)
    parser.add_argument("--save_dir", help="Directory to save checkpoint")
    parser.add_argument("--arch", help="Architecture of model: vgg or densenet")
    parser.add_argument("--learning_rate", help="Learning rate to be applied")
    parser.add_argument("--hidden_units", help="Number of nodes in hidden layer")
    parser.add_argument("--epochs", help="Number of epochs to train for")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    args_dict = {
        "data_dir":args.data_dir,
        "save_dir": (args.save_dir + "/checkpoint.pth") if args.save_dir is not None else "checkpoint.pth",
        "arch": args.arch if args.arch is not None else "vgg",
        "learning_rate": float(args.learning_rate) if args.learning_rate is not None else 0.005,
        "hidden_units": int(args.hidden_units) if args.hidden_units is not None else  4096,
        "epochs": int(args.epochs) if args.epochs is not None else 3,
        "gpu": "cuda" if args.gpu else "cpu"
    }

    return args_dict

def process_data(data_dir):
    train_dir, valid_dir, test_dir = data_dir + "/train", data_dir + "/valid", data_dir + "/test"

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {"training":transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                        "validation":transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                        "testing":transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        "training":datasets.ImageFolder(train_dir, transform=data_transforms["training"]),
        "validation":datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "testing":datasets.ImageFolder(test_dir, transform=data_transforms["testing"])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "training":torch.utils.data.DataLoader(image_datasets["training"], batch_size=32, shuffle=True),
        "validation":torch.utils.data.DataLoader(image_datasets["validation"], batch_size=32, shuffle=True),
        "testing":torch.utils.data.DataLoader(image_datasets["testing"], batch_size=32, shuffle=True),
    }

    return dataloaders

def create_model(arch, hidden_units):
    if arch == "vgg":
        model = models.vgg11(pretrained = True)
        input_nodes = 25088
    elif arch == "densenet":
        model = models.densenet121(pretrained=True)
        input_nodes = 1024
    else:
        raise Exception("Architecture chosen is not one of the options")
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(input_nodes, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1)
                                     )

    return model

def train_model(model, learning_rate, epochs, device, trainloader, validloader):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    running_loss = 0
    total_batches = len(trainloader) * epochs
    batches_completed = 0

    for epoch in epochs:
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches_completed += 1
            if batches_completed % 10 == 0:
                print(f"{batches_completed} / {total_batches} batches completed")
        else:
            test_loss = 0
            accuracy = 0

            model.eval()
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    logps = model(inputs)
                    test_loss += criterion(logps, labels)
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor))

            model.train()
            print("Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)),
                f"{epoch}/{epochs} epochs completed."  
            )


def test_accuracy(model, testloader, device):
    test_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model(inputs)
            test_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor if device == 'cpu' else torch.cuda.FloatTensor))
            
            
    model.train()
    return accuracy / len(testloader)

def save_model(model, save_dir):
    checkpoint = {
            'model': model.cpu(),
            'features': model.features,
            'classifier': model.classifier,
            'state_dict': model.state_dict()}
    
    torch.save(checkpoint, save_dir)

def main():
    args = get_args()
    loaders = process_data(args["data_dir"])

    print("Creating model...")
    model = create_model(arch=args["arch"], hidden_units=args["hidden_unit"])

    print("Training model...")
    train_model(model, args["learning_rate"], args["epochs"], args["device"], loaders["training"], loaders["validation"])
    print("Training Complete")

    print("Testing accuracy...")
    accuracy = test_accuracy(model, loaders["testing"])
    print(f"Accuracy is {accuracy * 100}%")


    print(f"Saving Model to {args['save_dir']}")
    save_model(model, args["save_dir"])

print(get_args())