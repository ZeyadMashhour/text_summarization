import torch
import torch.nn.functional as F

def train_fn(data_loader, model, optimizer, device):
    model.train()
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = F.mse_loss(outputs.squeeze(-1), targets.float())

        loss.backward()

        optimizer.step()

def eval_fn(data_loader, model, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = F.mse_loss(outputs.squeeze(-1), targets.float())

            total_loss += loss.item()

        return total_loss / len(data_loader)
