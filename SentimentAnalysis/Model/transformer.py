import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

# Define a SelfAttention class
class SelfAttention(nn.Module):


    def __init__(self, emb_size, heads):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        # Verify head_dim size
        assert (self.head_dim * heads == emb_size), "Embed size must be divisible by heads"

        # Initialize linear layers for values, keys, queries, and output
        self.values = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size) # should be (heads*self.head_dim, embed_size) but heads*self.head_dim should be equal to embed_size
    
    def forward(self, values, keys, queries, mask=None):
        # Get the batch size and lengths of values, keys, and queries
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # get len

        # Pass values, keys, and queries through linear layers
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Reshape values, keys, and queries to have separate dimensions for heads and head_dim
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # (N, values_len, heads, head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) # (N, key_len, heads, head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim) # (N, query_len, heads, head_dim)

        # Calculate the energy (dot product) between queries and keys
        # We have:
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # We want
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # QK.T ( part of attention )
        #print("Energy before mask output:", has_nan_or_inf(energy))

        # If a mask is provided, apply it to the energy
        if mask is not None:
            # Upper left triangle is being changed to 0s in order to prevent insight of the following words / tokens and try to predict them
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            #print("Energy after mask output:", has_nan_or_inf(energy))
        
        # Normalize the energy by the square root of emb_size and apply softmax
        # Basically (QK.T(energy) / emb_size(dk) ** 0.5 )
        attention = torch.softmax(energy / (self.emb_size ** (1 / 2)), dim=3) # Normalizing across key_len

        #print("Attention output:", has_nan_or_inf(attention))

        # Multiply the attention weights by the values and sum along the key_len dimension
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

        # Pass the output through a fully connected layer
        out = self.fc_out(out)
        return out
    
# Define a TransformerBlock class
class TransformerBlock(nn.Module):

    def __init__(self, emb_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # Instantiate a SelfAttention layer to model relationships between tokens in the sequence
        self.attention = SelfAttention(emb_size, heads)

        # Add LayerNorm layers for input stabilization and improving training efficiency
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        # Define a feed-forward neural network for additional non-linear transformation
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion*emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*emb_size, emb_size)
        )
        # Add dropout for regularization and preventing overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Calculate self-attention scores and apply them to the input values
        attention = self.attention(value, key, query, mask)

        # Perform residual connection (add original input to the self-attention output)
        # and apply LayerNorm to stabilize the input for the next layer
        x = self.dropout(self.norm1(attention + query))

        # Pass the output through the feed-forward network for further non-linear transformation
        forward = self.feed_forward(x)

        # Perform another residual connection (add the output of the self-attention layer
        # to the output of the feed-forward network) and apply LayerNorm
        out = self.dropout(self.norm2(forward+x))
        
        return out

# Define an Encoder class
class Encoder(nn.Module):


    def __init__(self, src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder,self).__init__()
        self.emb_size = emb_size
        self.device = device

        # Create word embeddings to map token indices to continuous vectors
        self.word_embedding = nn.Embedding(src_vocab_size, emb_size)

        # Create position embeddings to incorporate positional information into the model
        self.position_embedding = nn.Embedding(max_length, emb_size)

        # Instantiate multiple TransformerBlocks in a ModuleList for the desired depth
        self.layers = nn.ModuleList([
            TransformerBlock(
                emb_size, heads, dropout=dropout,
                forward_expansion=forward_expansion
            ) for _ in range(num_layers)
        ])

        # Add dropout for regularization and preventing overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Get the batch size and sequence length
        N, seq_length = x.shape

        # Generate position indices and expand them to match the input shape
        pos = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Combine word and position embeddings and apply dropout
        out = self.dropout(self.word_embedding(x) + self.position_embedding(pos))
        
        # Pass the input through the TransformerBlock layers
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

# Define a Decoder class
class Decoder(nn.Module):

    def __init__(self, target_vocab_size, emb_size, num_layers, heads, forward_expansion, dropout, device, max_len):
        super(Decoder,self).__init__()
        self.emb_size = emb_size
        self.device = device

        # Create word embeddings to map token indices to continuous vectors
        self.word_embedding = nn.Embedding(target_vocab_size, emb_size)

        # Create position embeddings to incorporate positional information into the model
        self.position_embedding = nn.Embedding(max_len, emb_size)

        # Instantiate multiple DecoderBlock layers in a ModuleList for the desired depth
        self.layers = nn.ModuleList([DecoderBlock(emb_size, heads, dropout=dropout, forward_expansion=forward_expansion, device=device) for _ in range(num_layers)])
        
        # Add a fully connected output layer to project the embeddings to target vocabulary size
        self.fc_out = nn.Linear(emb_size, target_vocab_size)

        # Add dropout for regularization and preventing overfitting
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, target_mask):
        # Get the batch size and sequence length
        N, seq_len = x.shape

        # Generate position indices and expand them to match the input shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        # Combine word and position embeddings and apply dropout
        x = self.dropout(self.word_embedding(x) + self.position_embedding(pos))

        # Pass the input through the DecoderBlock layers with the Encoder output and masks
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)
        
        # Project the output of the DecoderBlock layers to the target vocabulary size
        out = self.fc_out(x)

        return out

# Define a DecoderBlock class
class DecoderBlock(nn.Module):

    def __init__(self, emb_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        # Instantiate a SelfAttention layer for masked self-attention in the Decoder
        self.attention = SelfAttention(emb_size, heads, device)

        # Add LayerNorm for input stabilization and improving training efficiency
        self.norm = nn.LayerNorm(emb_size)

        # Instantiate a TransformerBlock for cross-attention between Encoder and Decoder
        self.transformer_block = TransformerBlock(emb_size, heads, dropout=dropout, forward_expansion=forward_expansion)

        # Add dropout for regularization and preventing overfitting
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, target_mask):

        # Calculate masked self-attention scores and apply them to the input
        attention = self.attention(x, x, x, target_mask)

        # Perform residual connection (add original input to the self-attention output)
        # and apply LayerNorm to stabilize the input for the next layer
        query = self.dropout(self.norm(attention + x))

        # Pass the output through the TransformerBlock for cross-attention
        # between the Encoder output and Decoder self-attention output
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, target_vocab_size, src_pad_idx, target_pad_idx, emb_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda',max_len=100):
        super(Transformer,self).__init__()

        # Instantiate an Encoder to process the source sequence
        self.encoder = Encoder(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len)

        # Instantiate a Decoder to generate the target sequence
        self.decoder = Decoder(target_vocab_size, emb_size, num_layers, heads, forward_expansion, dropout, device, max_len)

        # Store padding indices and device for mask creation
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
    
    # Create a source mask for the Encoder to ignore padding tokens
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    # Create a target mask for the Decoder to prevent attention beyond the current token
    def make_target_mask(self, target):
        N, target_len = target.shape

        target_mask = torch.tril(torch.ones((target_len,target_len))).expand(N,1 , target_len, target_len)

        return target_mask.to(self.device)

    def forward(self, src, target):
        # Create masks for the source and target sequences
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)

        # Pass the source sequence through the Encoder
        enc_src = self.encoder(src, src_mask)

         # Pass the target sequence and Encoder output through the Decoder
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

        # Instantiate an Encoder to process the input sequence
        self.encoder = Encoder(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len)

        # Add a fully connected layer to map the Encoder output to the number of emotion classes
        self.fc_out = nn.Linear(emb_size, num_classes)

        # Store padding index and device for mask creation
        self.src_pad_idx = src_pad_idx
        self.device = device

    # Create a source mask for the Encoder to ignore padding tokens
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, src):
        # Create a mask for the input sequence
        src_mask = self.make_src_mask(src)

        # Pass the input sequence through the Encoder
        enc_src = self.encoder(src, src_mask)

        # Calculate the mean of the Encoder output along the sequence dimension
        enc_src_mean = enc_src.mean(dim=1)

        # Pass the mean vector through the fully connected layer to obtain the final output
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

        

        # Instantiate an Encoder to process the input sequence
        self.encoder = Encoder(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len)

        # Instantiate an EmotionFeatureExtractor to extract emotion features from the encoded sequence
        self.emotion_extractor = EmotionFeatureExtractor(emb_size)

        # Add a fully connected layer to map the concatenated features to the number of classes
        self.classifier = nn.Linear(emb_size + 1, num_classes)

        # Add a pre-classifier fully connected layer to transform encoded sequence representation
        self.pre_classifier = nn.Linear(emb_size, emb_size)

        # Add a dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Store padding index, device, and number of classes for mask creation and other uses
        self.src_pad_idx = src_pad_idx
        self.device = device

        # Reset parameters and enable anomaly detection for debugging
        self.num_classes = num_classes
        self._reset_parameters()
        torch.autograd.set_detect_anomaly(True)
    
    # Initialize weights and biases with appropriate initial values
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                fan_in = p.size(-1)
                nn.init.normal_(p, mean=0, std=np.sqrt(1 / fan_in))
            elif 'bias' in name:
                nn.init.zeros_(p)

    # Create a source mask for the Encoder to ignore padding tokens
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def forward(self, src: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        #assert not has_nan_or_inf(src.flatten()), "Input IDs contain NaN or infinity values."
        #assert not has_nan_or_inf(labels), "Input IDs contain NaN or infinity values."

        # Create a mask for the input sequence
        src_mask = self.make_src_mask(src)
        #print("Src Mask:", has_nan_or_inf(src_mask))

        # Pass the input sequence through the Encoder
        enc_src = self.encoder(src, src_mask)
        #print("Encoder output:", has_nan_or_inf(enc_src))

        # Calculate the mean of the Encoder output along the sequence dimension
        enc_src_mean = enc_src.mean(dim=1)

        # Extract sentiment scores using the EmotionFeatureExtractor
        sentiment_scores = self.emotion_extractor(enc_src_mean)

        # Transform the mean vector using the pre-classifier layer and apply ReLU activation
        pooled_output = self.pre_classifier(enc_src_mean)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)

        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        #print("Pooled output:", has_nan_or_inf(pooled_output))

        # Concatenate the pooled output with the sentiment scores
        pooled_output = torch.cat((pooled_output, sentiment_scores), dim=1) # Add sentiment scores to the pooled output

        # Pass the concatenated features through the classifier layer to obtain the final logits
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        #print("Logits output:", has_nan_or_inf(logits))

        # Calculate the loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return (loss, logits)
    

### ------------- TRANSFORMER WITH LOCAL SELF-ATTENTION --------------- ###

#  Attention is limited to a fixed-sized window around the current position.
#  Can help the model focus on local patterns, which might be relevant to emotion analysis.

# Explanation:

# LocalSelfAttention is a specialized attention mechanism designed to focus on a limited context in the input sequence. 
# It restricts the attention to a local window of tokens around the current token, 
# reducing the computational cost and promoting better handling of long sequences.
# By focusing on a smaller range of tokens, the model is less likely to be influenced by irrelevant information, 
# which can help improve its performance in certain tasks, like in this case, emotional analysis.
class LocalSelfAttention(nn.Module):
    def __init__(self, emb_size, heads, device, k=16):
        super(LocalSelfAttention, self).__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads
        self.device = device

        # Local context window size
        self.k = k

        # Verify head_dim size
        assert (self.head_dim * heads == emb_size), "Embed size must be divisible by heads"

        # Linear transformations for values, keys, and queries
        self.values = nn.Linear(emb_size, emb_size)
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.fc_out = nn.Linear(emb_size, emb_size)

    def forward(self, values, keys, queries, mask=None):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Apply linear transformations
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Reshape to obtain separate heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Apply local attention
        if self.k > 0:
            max_k = self.k // 2
            min_k = -max_k
            local_mask = torch.ones((key_len, key_len), device=self.device)
            local_mask = local_mask.tril(max_k).triu(min_k)
            local_mask = local_mask.view(1, 1, key_len, key_len)
            
            if mask is not None:
                mask = mask * local_mask  # Combine the local mask and the regular mask
            else:
                mask = local_mask
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.emb_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerLocalAttentionBlock(nn.Module):
    def __init__(self, emb_size, heads, device, forward_expansion, dropout, k=16): # Add the k parameter
        super(TransformerLocalAttentionBlock, self).__init__()
        self.attention = LocalSelfAttention(emb_size, heads, device, k) # Replace SelfAttention with LocalSelfAttention
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),
            nn.Linear(forward_expansion * emb_size, emb_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)

        # Apply residual connection and layer normalization
        x = self.norm1(attention + query)
        x = self.dropout(x)
        forward = self.feed_forward(x)

        # Apply residual connection and layer normalization
        out = self.norm2(forward + x)
        out = self.dropout(out)

        return out

class EncoderLocalAttention(nn.Module):

    def __init__(self, src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len, k=16): # Add the k parameter
        super(EncoderLocalAttention,self).__init__()
        self.emb_size = emb_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_len, emb_size)
        self.layers = nn.ModuleList(
            [
                TransformerLocalAttentionBlock(
                    emb_size,
                    heads,
                    device,
                    forward_expansion,
                    dropout,
                    k, # Pass the k parameter to the TransformerBlock
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        pos = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(pos))
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

class MultilabelLocalAttentionSequenceClassificationTransformer(nn.Module):
    def __init__(self, src_vocab_size, num_classes, src_pad_idx, emb_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device='cuda', max_len=100,k=16):
        super(MultilabelLocalAttentionSequenceClassificationTransformer, self).__init__()
        self.encoder = EncoderLocalAttention(src_vocab_size, emb_size, num_layers, heads, device, forward_expansion, dropout, max_len,k)
        self.emotion_extractor = EmotionFeatureExtractor(emb_size)
        self.classifier = nn.Linear(emb_size + 1, num_classes)
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
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        enc_src_mean = enc_src.mean(dim=1)
        sentiment_scores = self.emotion_extractor(enc_src_mean)
        pooled_output = self.pre_classifier(enc_src_mean)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        pooled_output = torch.cat((pooled_output, sentiment_scores), dim=1) # Add sentiment scores to the pooled output
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        return (loss, logits)



# Computes sentiment scores for each token and uses these scores to modulate the attention mechanism

class EmotionFeatureExtractor(nn.Module):
    def __init__(self, emb_size):
        super(EmotionFeatureExtractor, self).__init__()
        self.sentiment_fc = nn.Linear(emb_size, 1)

    def forward(self, x):
        sentiment_scores = torch.sigmoid(self.sentiment_fc(x))
        return sentiment_scores
    
def has_nan_or_inf(tensor):
    return torch.isnan(tensor.detach()).any() or torch.isinf(tensor.detach()).any()
