import os
from os.path import join
from utils import *
from Network import *
import matplotlib.pyplot as plt


def run():
    train_window = 24
    x_full, y_full, x_train, y_train, x_val, y_val = load_flatbeam_data(train_window)
    # train(x_train, y_train, x_val, y_val, train_window)

    plot_graph(x_val, y_val)

def train(x_train, y_train, x_val, y_val, train_window):
    # hyperparams
    train_window = train_window
    input_size = 1
    enc_seq_len = train_window
    dec_seq_len = train_window // 3
    output_sequence_length = 1

    dim_val = 10
    dim_attn = 5
    lr = 0.002
    epochs = 3

    n_heads = 3

    n_decoder_layers = 3
    n_encoder_layers = 3

    batch_size = 15

    # init network and optimizer
    t = Transformer(dim_val, dim_attn, input_size, dec_seq_len, output_sequence_length, n_decoder_layers,
                    n_encoder_layers, n_heads)
    optimizer = torch.optim.Adam(t.parameters(), lr=lr)

    # keep track of loss for graph
    losses = []
    for e in range(epochs):
        out = []
        for i in range(0, x_train.size(0), batch_size):
            if i + batch_size > x_train.size(0):
                x, y = x_train[i:, :, 0], y_train[i:, 1]
            else:
                x, y = x_train[i:(i + batch_size), :, 1], y_train[i:(i + batch_size), 0]
            output = t(x.unsqueeze(2))

            loss = F.mse_loss(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            # out.append([net_out.detach().numpy(), Y])
            losses.append(loss)
            print(i, loss)
            # break
    print("saving")
    # torch.save(t.state_dict(), "test_model")
    torch.save(t, "test_model")


def plot_graph(x_val, y_val):
    t = torch.load("test_model")
    t.eval()
    ypred = t(x_val[:, :, 0].unsqueeze(2))

    plt.figure(figsize = (16, 8))

    plt.plot(y_val[:,0], linewidth = 3, alpha = 0.5)
    plt.plot(ypred.detach().numpy()[:,0])

    plt.xlim([0,500])
    plt.show()

def evaluate(model, X_test, Y_test):
    # model.eval()
    with torch.no_grad():
        output = model(X_test)
        test_loss = F.mse_loss(output, Y_test)
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
        return test_loss.item()

if __name__ == '__main__':
    run()
