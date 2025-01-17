import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
import argparse
from dataloader import get_train_loader, get_test_loader  # Import the custom data loaders
from model import CNNExperiment  # Import your model

# Default parameters
save_file = 'snoutnet_weights.pth'
n_epochs = 30
batch_size = 32
plot_file = 'snoutnet_loss_plot.png'
transform = 'none'


def train(n_epochs, optimizer, model, loss_fn, train_loader, test_loader, scheduler, device, save_file=None,
          plot_file=None):
    print('Training ...')
    model.train()

    losses_train = []
    losses_val = []

    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)
        loss_train = 0.0

        # Training loop
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device).float()  # Ensure targets are float

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # Update learning rate
        scheduler.step(loss_train)

        losses_train += [loss_train / len(train_loader)]  # changed line to match lab1

        # Validation loop (no gradient update)
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(device)
                targets = targets.to(device).float()

                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                loss_val += loss.item()

        losses_val += [loss_val / len(test_loader)]  # see a few lines above

        model.train()  # Set the model back to training mode

        print(
            f'{datetime.datetime.now()} Epoch {epoch}, Training loss: {loss_train / len(train_loader)}, Validation loss: {loss_val / len(test_loader)}')

        # Save model weights
        if save_file is not None:
            torch.save(model.state_dict(), save_file)

        # Plot loss
        if plot_file is not None:
            plt.figure(figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='Train Loss')
            plt.plot(losses_val, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc=1)
            print('Saving plot to', plot_file)
            plt.savefig(plot_file)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def main():
    global save_file, n_epochs, batch_size, plot_file

    print('Running main ...')

    # Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', metavar='state', type=str, help='Parameter file (.pth)')
    argParser.add_argument('-e', metavar='epochs', type=int, help='Number of epochs [30]')
    argParser.add_argument('-b', metavar='batch size', type=int, help='Batch size [32]')
    argParser.add_argument('-p', metavar='plot', type=str, help='Output loss plot file (.png)')
    argParser.add_argument('-t', metavar='transform', type=str, help='Transform type [flip] or [saturate]')

    args = argParser.parse_args()

    if args.s is not None:
        save_file = args.s
    if args.e is not None:
        n_epochs = args.e
    if args.b is not None:
        batch_size = args.b
    if args.p is not None:
        plot_file = args.p
    if args.p is not None:
        transform = args.t

    print('\t\tn epochs =', n_epochs)
    print('\t\tbatch size =', batch_size)
    print('\t\tsave file =', save_file)
    print('\t\tplot file =', plot_file)
    print('\t\ttransform =', transform)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('\t\tusing device', device)

    # Initialize the model
    model = CNNExperiment()  # Replace with your model initialization if needed
    model.to(device)
    model.apply(init_weights)

    # Use the custom data loaders
    train_loader = get_train_loader(batch_size=batch_size, transform_type=transform)
    test_loader = get_test_loader(batch_size=batch_size, transform_type=transform)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss_fn = nn.MSELoss()

    train(
        n_epochs=n_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,  # Pass the test loader
        scheduler=scheduler,
        device=device,
        save_file=save_file,
        plot_file=plot_file
    )


if __name__ == '__main__':
    main()
