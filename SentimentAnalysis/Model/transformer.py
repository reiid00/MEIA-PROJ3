import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class SelfAttention(nn.Module):

    def __init__(self, emb_size, heads,device):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads
        self.device = device

        # Verify head_dim size
        assert (self.head_dim * heads == emb_size), "Embed size must be divisible by heads"

        self.values = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size) # should be (heads*self.head_dim, embed_size) but heads*self.head_dim should be equal to embed_size

        # Initialize bias terms with zeros
        nn.init.zeros_(self.values.bias)
        nn.init.zeros_(self.keys.bias)
        nn.init.zeros_(self.queries.bias)
    
    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # get len

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # (N, values_len, heads, head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) # (N, key_len, heads, head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim) # (N, query_len, heads, head_dim)
        
        # We have:
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # We want
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # QK.T ( part of attention )
        #print("Energy before mask output:", has_nan_or_inf(energy))
        if mask is not None:
            # Upper left triangle is being changed to 0s in order to prevent insight of the following words / tokens and try to predict them
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            #print("Energy after mask output:", has_nan_or_inf(energy))
        
        # Basically (QK.T(energy) / emb_size(dk) ** 0.5 )
        attention = torch.softmax(energy / (self.emb_size ** (1 / 2)), dim=3) # Normalizing across key_len

        #print("Attention output:", has_nan_or_inf(attention))
        # We have:
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, values_len, heads, head_dim)
        # We want:
        # output shape : (N, query_len, heads, head_dim)
        # After einsum
        # Flatten the last two dimensions
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        #print("Output after einsum:", has_nan_or_inf(out))

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):

    def __init__(self, emb_size, heads, dropout, forward_expansion, device):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(emb_size, heads, device)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion*emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*emb_size, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x))
        
        return out

class Encoder(nn.Module):

    def __init__(self, src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_length, emb_size)
        self.layers = nn.ModuleList([TransformerBlock(emb_size, heads, dropout=dropout, forward_expansion=forward_expansion, device=device) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        pos = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(pos))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

class DecoderBlock(nn.Module):

    def __init__(self, emb_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(emb_size, heads, device)
        self.norm = nn.LayerNorm(emb_size)
        self.transformer_block = TransformerBlock(emb_size, heads, dropout=dropout, forward_expansion=forward_expansion, device=device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):

    def __init__(self, target_vocab_size, emb_size, num_layers, heads, forward_expansion, dropout, device, max_len):
        super(Decoder,self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_len, emb_size)
        self.layers = nn.ModuleList([DecoderBlock(emb_size, heads, dropout=dropout, forward_expansion=forward_expansion, device=device) for _ in range(num_layers)])
        self.fc_out = nn.Linear(emb_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, target_mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(pos))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx, emb_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda',max_len=100):
        super(Transformer,self).__init__()

        self.encoder = Encoder(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len)

        self.decoder = Decoder(target_vocab_size, emb_size, num_layers, heads, forward_expansion, dropout, device, max_len)

        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape

        target_mask = torch.tril(torch.ones((target_len,target_len))).expand(N,1 , target_len, target_len)

        return target_mask.to(self.device)

    def forward(self, src, target):

        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(target, enc_src, src_mask, target_mask)

        return out
    

# Differences From Transformer to Emotional Analysis Model

# 1. Removed the Transformer, and Decoder class and created a new EmotionAnalysisModel class. Emotion analysis is typically a classification task, 
#  and doesn't need the decoder part of the transformer for this. Only needs the encoder to generate a meaningful representation of the input text.
# 2. The new EmotionAnalysisModel class has the Encoder and a fully connected layer (fc_out). The Encoder generates the input text's representation, 
# and the fully connected layer maps this representation to the number of emotion classes you have.
# 3. In the forward method, removed the target-related parts since they are not needed for a classification task. Also added a line to calculate 
# the mean of the encoder output along the sequence dimension. This provides a single vector representation of the input text, which is then passed through 
# the fully connected layer to produce the final output.

class EmotionAnalysisModel(nn.Module):
    def __init__(self, src_vocab_size, num_classes, src_pad_idx, emb_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda', max_len=100):
        super(EmotionAnalysisModel, self).__init__()

        self.encoder = Encoder(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len)
        self.fc_out = nn.Linear(emb_size, num_classes)
        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, src):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        enc_src_mean = enc_src.mean(dim=1)
        out = self.fc_out(enc_src_mean)
        return out


# Changes from normal EmotionAnalysisModel

# 1. Removed the 'fc_out' layer and replaced it with a 'classifier' layer to be consistent with the naming in the BertForMultilabelSequenceClassification class.
# 2. Added the dropout layer just before the classifier to regularize the model.
# 3. Changed the forward method signature to match the BertForMultilabelSequenceClassification class. I added an optional 'labels' argument, which is used to 
#  calculate the loss when provided.
# 4. Added the BCEWithLogitsLoss loss function for multilabel classification. This loss function combines a sigmoid activation and binary cross-entropy loss, 
# making it suitable for multilabel classification tasks

class MultilabelSequenceClassificationTransformer(nn.Module):
    def __init__(self, src_vocab_size, num_classes, src_pad_idx, emb_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda', max_len=100):
        super(MultilabelSequenceClassificationTransformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len)
        self.classifier = nn.Linear(emb_size, num_classes)
        self.pre_classifier = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.num_classes = num_classes
        self._reset_parameters()
        torch.autograd.set_detect_anomaly(True)
    
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                fan_in = p.size(-1)
                nn.init.normal_(p, mean=0, std=np.sqrt(1 / fan_in))
            elif 'bias' in name:
                nn.init.zeros_(p)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, src: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        #assert not has_nan_or_inf(src.flatten()), "Input IDs contain NaN or infinity values."
        #assert not has_nan_or_inf(labels), "Input IDs contain NaN or infinity values."
        src_mask = self.make_src_mask(src)
        #print("Src Mask:", has_nan_or_inf(src_mask))
        enc_src = self.encoder(src, src_mask)
        #print("Encoder output:", has_nan_or_inf(enc_src))
        enc_src_mean = enc_src.mean(dim=1)
        pooled_output = self.pre_classifier(enc_src_mean)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        #print("Pooled output:", has_nan_or_inf(pooled_output))
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        #print("Logits output:", has_nan_or_inf(logits))

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return (loss, logits)
    
def has_nan_or_inf(tensor):
    return torch.isnan(tensor.detach()).any() or torch.isinf(tensor.detach()).any()
