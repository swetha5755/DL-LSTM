# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The task is to identify and classify named entities such as person names, locations, organizations, etc., from a given text using an LSTM model. The dataset consists of sentences where each word is labeled with its corresponding entity tag (like B-PER, I-LOC, O, etc.).

## DESIGN STEPS
### STEP 1: 
Load and preprocess the dataset by tokenizing the text and converting words and labels into numerical format.

### STEP 2: 
Pad the input sequences to make all sentences of equal length.


### STEP 3: 

Build the LSTM model with embedding layer, LSTM layer, and a dense output layer for classification.

### STEP 4: 

Compile the model using appropriate loss function (like categorical cross-entropy) and optimizer.

### STEP 5: 

Train the model using the training data and validate it using validation data.

### STEP 6: 

Evaluate the model performance and use it to predict named entities in new text.



## PROGRAM

### Name: Swetha S

### Register Number: 212224040344

```python
class BiLSTMTagger(nn.Module):
  def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
    super(BiLSTMTagger, self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim)
    self.dropout=nn.Dropout(0.1)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.fc=nn.Linear(hidden_dim*2,tagset_size)
  def forward(self,x):
    x = self.embedding(x)
    x = self.dropout(x)
    x,_=self.lstm(x)
    return self.fc(x)

model =BiLSTMTagger(len(word2idx)+1,len(tag2idx)).to(device)
loss_fn =nn.CrossEntropyLoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss=0
      for batch in train_loader:
        input_ids=batch["input_ids"].to(device)
        labels=batch["labels"].to(device)
        optimizer.zero_grad()
        outputs=model(input_ids)
        loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss)
      model.eval()
      val_loss=0
      with torch.no_grad():
        for batch in test_loader:
          input_ids=batch["input_ids"].to(device)
          labels=batch["labels"].to(device)
          outputs=model(input_ids)
          loss=loss_fn(outputs.view(-1,len(tag2idx)),labels.view(-1))
          val_loss+=loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f},Val Loss = {val_loss:.4f}")  
      
    return train_losses,val_losses 

```

### OUTPUT

## Loss Vs Epoch Plot

<img width="581" height="501" alt="image" src="https://github.com/user-attachments/assets/a2a07f46-5808-4866-8383-d0cd4f21129c" />


### Sample Text Prediction
<img width="288" height="410" alt="image" src="https://github.com/user-attachments/assets/0f9550bb-756d-483a-bd4d-340fc21dca93" />


## RESULT
This program has been executed successfully.
