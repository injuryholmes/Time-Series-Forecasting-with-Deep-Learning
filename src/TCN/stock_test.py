
parser = argparse.ArgumentParser(description='Stock')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false', default=False,
                    help='use CUDA (default: False)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


use_columns = ['Close', 'Volume']

input_channels = len(use_columns)

channel_sizes = [32, 32, 32, 32]
ONE_DAY = 1
model = TCN(input_channels, ONE_DAY,  channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

def train(ep):
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # output:[64,10]
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0



if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        if epoch % 10 == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr