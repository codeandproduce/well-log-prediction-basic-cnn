import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from loguru import logger
from easydict import EasyDict as edict

from dataloader import get_dataloaders
from model import BasicConvNet

def main():
    args = {
        "batch_size": 16,
        "train_ratio": "9:1:2",
        "window_size": 9,
        "desired_columns": ["BS", "CALI", "DRHO", "DT", "GR", "NPHI", "PEF", "RACEHM", "RACELM", "RHOB", "ROP", "RPCEHM", "RPCELM", "RT"],
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-8, 
        "device": "cpu",
    }

    args = edict(args)

    train_loader, valid_loader, test_loader = get_dataloaders(
        desired_columns=args.desired_columns,
        train_ratio=args.train_ratio,
        window_size=args.window_size,
        batch_size=args.batch_size,
        device=args.device
    )
    model = BasicConvNet([args.window_size, len(args.desired_columns)])
    model = model.to(args.device)
    
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
            out = model(features)   
            out = out.squeeze(dim=-1)
            loss = criterion(out, targets)
            epoch_loss.append(loss.item())
            
            model.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
            optimizer.step()
            break
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
                loss = criterion(out, targets)

                epoch_val_loss.append(loss.item())
            val_loss.append(np.mean(epoch_val_loss))
            epoch_val_loss = []
        if epoch % 1 == 0:
            logger.info('Epoch: {}, Train Loss: {:.4e}, Valid Loss: {:.4e}'.format(epoch, train_loss[-1], val_loss[-1]))
        # Update minimum loss
        if min_loss > train_loss[-1]:
            patience = 0
            min_loss = train_loss[-1]
            #torch.save(model.state_dict(), path)
        else:
            patience += 1

        # Early stop when patience become larger than 10
        if patience > 10:
            logger.info("STOPP")
            break
            
        torch.cuda.empty_cache() ## 캐시 비워주기 자주 해줘야함
        
if __name__ == "__main__":
    main()
    
