import torch

DEVICE_IDS = [0, 1]


def train(model, dataloader, testloader):
    optimizer = model.optimizer()
    lr_scheduler = model.lr_scheduler()
    num_epochs = model.num_epochs

    # run on gpu if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on {}'.format(device))
    model.to(device)
    model = torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)

    print("start training for {} epochs".format(num_epochs))
    total_steps = 1
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            # run on gpu if available.
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            loss = torch.nn.functional.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log/print progress
            if total_steps % 10 == 0:
                with torch.no_grad():
                    # argmax : max returns the values and the indices, we want the indices
                    _, predictions = torch.max(output, 1)
                    accuracy = torch.sum(predictions == labels) / predictions.size(0) * 100

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {} %'
                          .format(epoch + 1, total_steps, loss.item(), accuracy.item()))

            total_steps += 1

        if testloader is not None:
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs_test, labels_test in testloader:
                    # run on gpu if available.
                    inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                    output = model(inputs_test)
                    _, predictions = torch.max(output, 1)
                    total += predictions.size(0)
                    correct += torch.sum(predictions == labels_test)

                print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
            lr_scheduler.step()

    print('training complete')


def test(model, dataloader):
    # run on gpu if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('testing on {}'.format(device))
    model.to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            # run on gpu if available.
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, predictions = torch.max(output, 1)
            total += predictions.size(0)
            correct += torch.sum(predictions == labels)

        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
