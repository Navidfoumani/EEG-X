import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Models.Attention import *
from Models.position_embedding import elec_location_Embedding, PositionalEmbedding

class EEG_X(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
         channel_size: number of EEG channels
         seq_len: number of timepoints in a window
        """
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2] 
        self.Layer_Norm = nn.LayerNorm(channel_size) # individual sample normalizing across channels 
        emb_size = config['emb_size']  # d_x
        # Embedding Layer -----------------------------------------------------------
        # config['pooling_size'] = 2  # Max pooling size in input embedding
        # seq_len = int(seq_len / config['pooling_size'])  # Number of patches (l)
        # self.InputEmbedding = InputEmbedding(config)  # input (Batch,Channel, length) -> output (Batch, l, d_x)
        # seq_len = int(seq_len / config['patch_size']) * channel_size  # Number of patches (l)
        # self.InputEmbedding = InputEmbedding_e2rsd_freq(config)  # input (Batch,Channel, length) -> output (Batch, l, d_x)
        # self.InputEmbedding = EEGPatchEmbedding(emb_size, config['patch_size'], 128)
        # self.InputEmbedding = EEGSTFTProcessor(window_size=config['patch_size'], overlap=0.5)

        # self.InputEmbedding = Linear_Embedding(config)

        self.InputEmbedding = Linear_Embedding_freq(config)

        new_seq_len = int((seq_len - config['patch_size'])/(config['patch_size']/4)) + 1
        self.LocationEncoding = elec_location_Embedding(emb_size)
        self.PositionalEncoding = PositionalEmbedding(seq_len, emb_size)
        
        # -------------------------------------------------------------------------
        self.momentum = config['momentum']
        self.device = config['device']
        self.mask_ratio = config['mask_ratio']
        self.mask_len = int(config['mask_ratio'] * new_seq_len)
        self.mask_token = nn.Parameter(torch.randn(emb_size, ))
        self.contex_encoder = Encoder(config)
        self.target_encoder = copy.deepcopy(self.contex_encoder)
        # self.Decoder = Decoder(config)
        self.Decoder = Linear_Embedding_Decoder(config)
        self.Predictor = Predictor(emb_size, config['num_heads'], config['dim_ff'], 1, config['pre_layers'])
        self.predict_head = nn.Linear(emb_size, config['num_labels'])
        self.Norm = nn.LayerNorm(emb_size)
        self.Norm2 = nn.LayerNorm(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def copy_weight(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = param_a.data

    def momentum_update(self):
        with torch.no_grad():
            for (param_a, param_b) in zip(self.contex_encoder.parameters(), self.target_encoder.parameters()):
                param_b.data = self.momentum * param_b.data + (1 - self.momentum) * param_a.data

    def linear_prob(self, x):
        with (torch.no_grad()):
            patches = self.InputEmbedding(x)
            patches = self.Norm(patches)
            patches = patches + self.LocationEncoding(patches)
            patches = patches + self.PositionalEncoding(patches)
            patches = self.Norm2(patches)
            patches = patches.reshape(patches.size(0), -1, patches.size(-1))  # B, Channel*patch_number, embedding_dim
            out = self.contex_encoder(patches)
            out = out.transpose(2, 1)
            out = self.gap(out)
            return out.squeeze()

    def pretrain_forward(self, x, x_c):
        patches = self.InputEmbedding(x)  # (Batch, l, d_x)
        patches = self.Norm(patches)
        patches = patches + self.LocationEncoding(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)

        rep_mask_token = self.mask_token.repeat(patches.shape[0], patches.shape[1], patches.shape[2], 1)
        rep_mask_token = rep_mask_token + self.LocationEncoding(rep_mask_token)
        rep_mask_token = rep_mask_token + self.PositionalEncoding(rep_mask_token)

        patches = patches.reshape(patches.size(0), -1, patches.size(-1))  # B, Channel*patch_number, embedding_dim
        rep_mask_token = rep_mask_token.reshape(rep_mask_token.size(0), -1, rep_mask_token.size(-1)) # B, Channel*patch_number, embedding_dim

        index = np.arange(patches.shape[1])
        index_chunk = Semantic_Subsequence_Preserving(index, 2, 1-self.mask_ratio)
        v_index = np.unique(np.concatenate(index_chunk))
        m_index = np.setdiff1d(index, v_index)

        visible = patches[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex = self.contex_encoder(visible)
        with torch.no_grad():
            target = self.target_encoder(patches)
            target_rep_mask = target[:, m_index, :]
        target_prediction = self.Predictor(rep_contex, rep_mask_token)

        # Prepare inputs for the decoder
        decoder_input = torch.zeros_like(patches)
        decoder_input[:, v_index, :] = rep_contex
        decoder_input[:, m_index, :] = target_prediction

        reconstruct = self.Decoder(decoder_input)

        X_c_Norm = self.Layer_Norm(x_c.permute(0, 2, 1)).permute(0, 2, 1)
        return [target_rep_mask, target_prediction, X_c_Norm, reconstruct]


    def forward(self, x):
        patches = self.InputEmbedding(x)
        patches = self.Norm(patches)
        patches = patches + self.LocationEncoding(patches)
        patches = patches + self.PositionalEncoding(patches)
        patches = self.Norm2(patches)
        patches = patches.reshape(patches.size(0), -1, patches.size(-1))  # B, Channel*patch_number, embedding_dim
        out = self.contex_encoder(patches)
        return self.predict_head(torch.mean(out, dim=1))


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2] 
        self.Norm = nn.BatchNorm1d(channel_size)
        emb_size = config['emb_size']  # d_x (input embedding dimension)
        k = 40
        # Embedding Layer -----------------------------------------------------------
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(channel_size, 1))
        self.spatial_padding = nn.ReflectionPad2d((int(np.floor((k - 1) / 2)), int(np.ceil((k - 1) / 2)), 0, 0))
        self.spatialwise_conv1 = nn.Conv2d(in_channels=1, out_channels=emb_size, kernel_size=(1, k))
        self.spatialwise_conv2 = nn.Conv2d(in_channels=emb_size, out_channels=1, kernel_size=(1, k))
        self.SiLU = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, config['pooling_size']), stride=(1, config['pooling_size']))

    def forward(self, x):
        out = x.unsqueeze(1)
        out = self.depthwise_conv(out)  # (bs, embedding, 1 , T)
        out = out.transpose(1, 2)  # (bs, 1, embedding, T)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv1(out)  # (bs, 1, embedding, T)
        out = self.SiLU(out)
        out = self.maxpool(out)  # (bs, 1, embedding, T // m)
        out = self.spatial_padding(out)
        out = self.spatialwise_conv2(out)
        out = out.squeeze(1)  # (bs, embedding, T // m)
        out = out.transpose(1, 2)  # (bs, T // m, embedding)
        patches = self.SiLU(out)
        return patches


class Linear_Embedding(nn.Module):
    """ EEG to Patch Embedding"""
    def __init__(self, config):
        super().__init__()
        patch_size=config['patch_size'] 
        patch_stride = int(config['patch_size']/2)
        embed_dim=config['emb_size']

        self.Norm = nn.BatchNorm1d(config['Data_shape'][1])
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), stride=(1, patch_stride))
        
    def forward(self, x):
        # x: B,C,T
        x = self.Norm(x)
        x = x.unsqueeze(1)# B, 1, C, T
        x = self.proj(x) # B, embed_dim, C, num_patches
        # Transpose to: B, num_patches, embed_dim, C
        x = x.permute(0, 2, 3, 1)  # B, C, num_patches, embed_dim
        # Reshape to: B, T*C, D
        # x = x.reshape(x.size(0), -1, x.size(-1))  # B, T*C, D
        return x


class EEGSTFTProcessor(nn.Module):
    def __init__(self, window_size=64, overlap=0.75):
        super().__init__()
        self.window_size = window_size
        self.hop_length = int(window_size * (1 - overlap))  # 50% overlap
        self.n_fft = window_size

    def forward(self, x):
        """
        x: (batch_size, channels, length)
        Returns: (batch_size, channels, num_windows)
        """
        batch_size, channels, length = x.shape
        num_windows = (length - self.window_size) // self.hop_length + 1

        # Prepare to collect the mean energy for each window
        stft_energy = torch.zeros((batch_size, channels, num_windows), device=x.device)

        # Apply STFT to each channel independently
        for ch in range(channels):
            # Perform STFT on the current channel across all batches
            spectral = torch.stft(
                x[:, ch, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.window_size,
                return_complex=True,
                center=False
            )
            # Compute the magnitude (energy) and average across frequency bins
            stft_energy[:, ch, :] = spectral.abs().mean(dim=1)

        return stft_energy


class InputEmbedding_e2rsd_freq(nn.Module):
    def __init__(self, config):
        super().__init__()
        channel_size = config['Data_shape'][1]
        emb_size = config['emb_size']  # d_x (input embedding dimension)

        # Device Layer --------------------------------------------------------------
        self.n_fft = 128 # config['n_fft']
        self.patch_size = config['patch_size']
        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        # channel token, N_channels >= each dataset's channel size
        self.channel_tokens = nn.Embedding(channel_size, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(channel_size)), requires_grad=False
        )
    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = int(self.patch_size/4),
            win_length = self.patch_size,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, subjects=None):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            # channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb) # Orig biot
            channel_emb = channel_spec_emb + channel_token_emb # To match EEG2Rep
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        out = torch.cat(emb_seq, dim=1)

        return out


class Linear_Embedding_freq(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_size = config['emb_size']  # d_x (input embedding dimension)
        # Device Layer --------------------------------------------------------------
        self.n_fft = config['patch_size']
        self.hope_length = config['patch_stride']
        self.patch_embedding = PatchFrequencyEmbedding(emb_size=emb_size, n_freq=self.n_fft // 2 + 1)

    def stft(self, sample):
        spectral = torch.stft( 
            input = sample.squeeze(1),
            n_fft = self.n_fft,
            hop_length = self.hope_length,
            win_length = self.n_fft,
            center = False,
            onesided = True,
            return_complex = True,
        )
        return torch.abs(spectral)

    def forward(self, x):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            channel_emb = channel_spec_emb 
            emb_seq.append(channel_emb)
        out = torch.stack(emb_seq, dim=1)
        return out


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=32, n_freq=128):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        d_model = config['emb_size']
        attn_heads = config['num_heads']
        d_ffn = config['dim_ff']
        layers = config['layers']
        dropout = config['dropout']
        enable_res_parameter = True
        # TRMs
        self.TRMs = nn.ModuleList(
            [TransformerBlock(d_model, attn_heads, d_ffn, enable_res_parameter, dropout) for i in range(layers)])

    def forward(self, x):
        for TRM in self.TRMs:
            x = TRM(x, mask=None)
        return x


class Predictor(nn.Module):
    def __init__(self, d_model, attn_heads, d_ffn, enable_res_parameter, layers):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList(
            [CrossAttnTRMBlock(d_model, attn_heads, d_ffn, enable_res_parameter) for i in range(layers)])

    def forward(self, rep_visible, rep_mask_token):
        for TRM in self.layers:
            rep_mask_token = TRM(rep_visible, rep_mask_token)
        return rep_mask_token
    

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.channels = config['Data_shape'][1]  # Number of output channels (14)
        self.emb_size = config['emb_size']  # Input embedding size
        self.seq_len_out = config['Data_shape'][2]  # Desired output length (256)
        patch_stride = int(config['patch_size']/2)
        
        # Convolutional Layers
        self.conv_to_channels = nn.Conv1d(self.emb_size, self.channels, kernel_size=1)  # Change channels from 32 to 14
        self.upsample = nn.ConvTranspose1d(self.channels, self.channels, kernel_size=config['pooling_size'], stride = patch_stride)  # Upsample length from 128 to 256
        self.SiLU = nn.SiLU(inplace=True)

    def forward(self, x):
        # x shape is (batch, 128, 32) 
        batch_size, length, emb_size = x.size()
        
        # Change shape to (batch, emb_size, length) for Conv1d processing
        out = x.permute(0, 2, 1)  # Shape: (batch, 32, 128)

        # First layer: Reduce channels from 32 to 14
        out = self.conv_to_channels(out)  # Shape: (batch, 14, 128)
        
        # Apply SiLU activation
        out = self.SiLU(out)
        
        # Second layer: Upsample from 128 to 256
        out = self.upsample(out)  # Shape: (batch, 14, 256)

        # Final output is (batch, 14, 256)
        return out


class Linear_Embedding_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size=config['patch_size'] 
        self.patch_stride = config['patch_stride']
        embed_dim=config['emb_size']
        self.output_size = [config['Data_shape'][1], config['Data_shape'][2]]
        # Transposed Conv2d to reverse the projection (Conv2d)
        self.deproj = nn.ConvTranspose2d(embed_dim, 1, kernel_size=(1, self.patch_size), stride=(1, self.patch_stride))

    def forward(self, x):
        """
        x: B, T*C, embed_dim (Embedding output)
        Returns: B, C, T (Reconstructed raw series)
        """
        # Reshape the embeddings back into (B, embed_dim, C, num_patches)
        B, _, D = x.shape
        C, L = self.output_size
        num_patches = int((L - self.patch_size)/self.patch_stride) + 1
        # Reshape x to (B, embed_dim, C, num_patches)
        x = x.reshape(B, D, C, num_patches)

        # Apply the transposed convolution to get back the original signal
        x_reconstructed = self.deproj(x)  # (B, 1, C, T)

        # Remove extra dimension and reshape to (B, C, T)
        x_reconstructed = x_reconstructed.squeeze(1)  # B, C, T
        
        return x_reconstructed


def Semantic_Subsequence_Preserving(time_step_indices, chunk_count, target_percentage):
    # Get the total number of time steps
    total_time_steps = len(time_step_indices)
    # Calculate the desired total time steps for the selected chunks
    target_total_time_steps = int(total_time_steps * target_percentage)

    # Calculate the size of each chunk
    chunk_size = target_total_time_steps // chunk_count

    # Randomly select starting points for each chunk with minimum distance
    start_points = [random.randint(0, total_time_steps - chunk_size)]
    # Randomly select starting points for each subsequent chunk with minimum distance
    for _ in range(chunk_count - 1):
        next_start_point = random.randint(0, total_time_steps - chunk_size)
        start_points.append(next_start_point)

    # Select non-overlapping chunks using indices
    selected_chunks_indices = [time_step_indices[start:start + chunk_size] for start in start_points]

    return selected_chunks_indices