import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from loguru import logger
import itertools
from easydict import EasyDict as edict

from dataloader import get_dataloaders
from model import MetaCNN

def main(args):
    log_path = f"meta_logs/model_{args.model_id}.log"
    logger.add(log_path)
    logger.info(args)
    train_loader, valid_loader, test_loader = get_dataloaders(
        desired_columns=args.desired_columns,
        train_ratio=args.train_ratio,
        window_size=args.window_size,
        batch_size=args.batch_size,
        device=args.device,
        full_seq=True
    )
    model = MetaCNN(
        input_shape=[args.window_size, len(args.desired_columns)],
        channels=args.channels,
        kernels=args.kernels,
        dilations=args.dilations
    )
    model = model.to(args.device)
    logger.info(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("pytorch_total_params", pytorch_total_params)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    patience = 0
    min_loss = np.inf
    train_loss = []
    val_loss = []
    for epoch in range(args.epochs):
        model.train()

        epoch_loss = []
        for _, batch in enumerate(train_loader):
            features = batch["features"].to(args.device)
            targets = batch["targets"].to(args.device)
            pointwise_target = targets[:, targets.size(1) // 2]
            print(targets)
            print(pointwise_target)
            
            # print(features)
            # print(targets)

            sequence_prediction, pointwise_prediction = model(features)   
            sequence_prediction = sequence_prediction.squeeze()
            pointwise_prediction = pointwise_prediction.squeeze()
            
            sequence_loss = criterion(sequence_prediction, targets)
            pointwise_loss = criterion(pointwise_prediction, pointwise_target)
            
            loss = sequence_loss*args.alpha + (1-args.alpha)*pointwise_loss
            epoch_loss.append(loss.item())
            
            model.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
            optimizer.step()
            return
         
        train_loss.append(np.mean(epoch_loss))
        epoch_loss = []
        torch.cuda.empty_cache()

        # Validate
        model.eval()
        with torch.no_grad():
            epoch_val_loss = []
            for _, batch in enumerate(valid_loader):
                features = batch["features"].to(args.device)
                targets = batch["targets"].to(args.device)
                out = model(features)
                out = out.squeeze()
                loss = criterion(out, targets)

                epoch_val_loss.append(loss.item())
            val_loss.append(np.mean(epoch_val_loss))
            epoch_val_loss = []
        if epoch % 1 == 0:
            logger.info('Epoch: {}, Train Loss: {:.4e}, Valid Loss: {:.4e}'.format(epoch, train_loss[-1], val_loss[-1]))
        
        if epoch % 50 == 0:
            if epoch > 0:
                path = f"meta_checkpoints/model_{args.model_id}/model_{args.model_id}_{epoch}.pt"
                
                if not os.path.isdir(f"meta_checkpoints/model_{args.model_id}"):
                    os.mkdir(f"meta_checkpoints/model_{args.model_id}")

                state = {
                    "args": args,
                    "state_dict": model.state_dict()
                }
                torch.save(state, path)
                
        torch.cuda.empty_cache() ## 캐시 비워주기 자주 해줘야함

    # now test
    model.eval()

    test_loss = []
    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            features = batch["features"].to(args.device)
            targets = batch["targets"].to(args.device)
            out = model(features)
            out = out.squeeze()
            loss = criterion(out, targets)

            test_loss.append(loss.item())
    mean_test_loss = np.mean(test_loss)

    logger.info(f"Final test loss: {mean_test_loss}")

    return train_loss, val_loss, test_loss


def try_alot():
    window_sizes = [7]
    lrs = [5e-5]
    col_lengths = [11,12,13]
    columns_list = []
    
    full_columns = ['CALI', 'DRHO',
       'DT', 'GR', 'NPHI', 'PEF', 'RACEHM', 'RACELM', 'RHOB', 'ROP',
       'RPCEHM', 'RPCELM', 'RT']
    
    for col_length in col_lengths:
        for one_combo in itertools.combinations(full_columns, col_length):
            columns_list.append(one_combo)
    
    count = 1
    for window_size in window_sizes:
        for lr in lrs:
            for columns in columns_list:
                for repeat_count in range(1, 5):
                    logger.remove()
                    args = {
                        "batch_size": 128,
                        "train_ratio": "9:1:2",
                        "window_size": window_size,
                        "desired_columns": columns,
                        "epochs": 201,
                        "lr": lr,
                        "weight_decay": 1e-8, 
                        "device": "cuda:1",
                        "channels": [32, 64, 32],
                        "kernels": [(3,3), (3,3), (3,3)],
                        "dilations": [1, 1, 1],
                        "alpha": 0.5,
                        "model_id": f"{count}_{repeat_count}",
                    }
                    args = edict(args)
                    train_loss, val_loss, test_loss = main(args)

                    train_dict = {
                        "model_id": args.model_id,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "test_loss": test_loss,
                        "args": args
                    }
                    f = open(f"meta_results/model_{args.model_id}.pkl", "wb")
                    pickle.dump(train_dict, f)
                    f.close()
                count += 1

if __name__ == "__main__":
    try_alot()
    
