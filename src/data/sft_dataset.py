"""
SFT Dataset for Pebble fine-tuning.

Loads tokenized conversations and creates loss masks so that
the training loss is only computed on assistant responses,
not on system prompts, user messages, or tool results.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    
    Each sample is a full conversation tokenized into a fixed-length window.
    A loss mask indicates which positions should contribute to the loss
    (only assistant responses).
    """
    
    def __init__(self, data_path: str, seq_len: int, 
                 assistant_token_id: int, end_token_id: int,
                 tool_result_start_id: int, tool_result_end_id: int,
                 bos_token_id: int):
        """
        Args:
            data_path: Path to the .bin file.
            seq_len: Sequence length for training.
            assistant_token_id: Token ID for <|assistant|>.
            end_token_id: Token ID for <|end|>.
            tool_result_start_id: Token ID for <|tool_result_start|>.
            tool_result_end_id: Token ID for <|tool_result_end|>.
            bos_token_id: Token ID for <|bos|>.
        """
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)
        
        self.assistant_id = assistant_token_id
        self.end_id = end_token_id
        self.tool_result_start_id = tool_result_start_id
        self.tool_result_end_id = tool_result_end_id
        self.bos_id = bos_token_id
        
        # Find conversation boundaries (BOS positions)
        # We sample whole conversations, not random windows
        self.conv_starts = []
        for i in range(self.n_tokens):
            if self.data[i] == self.bos_id:
                self.conv_starts.append(i)
        
        self.conv_starts = np.array(self.conv_starts)
    
    def __len__(self):
        return len(self.conv_starts)
    
    def __getitem__(self, idx):
        """
        Get a conversation with loss mask.
        
        Returns:
            input_ids: (seq_len,) LongTensor
            target_ids: (seq_len,) LongTensor  
            loss_mask: (seq_len,) BoolTensor — True where loss should be computed
        """
        start = self.conv_starts[idx]
        
        # Find end of this conversation (next BOS or end of data)
        if idx + 1 < len(self.conv_starts):
            end = self.conv_starts[idx + 1]
        else:
            end = self.n_tokens
        
        # Extract tokens (pad or truncate to seq_len + 1)
        conv_len = end - start
        if conv_len > self.seq_len + 1:
            # Truncate
            chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        else:
            # Pad with zeros (will be masked out anyway)
            chunk = np.zeros(self.seq_len + 1, dtype=np.int64)
            chunk[:conv_len] = self.data[start : end].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1].copy())   # input
        y = torch.from_numpy(chunk[1:].copy())     # target (shifted)
        
        # Build loss mask: True only on assistant response tokens
        # Walk through tokens and track state
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        in_assistant = False
        in_tool_result = False
        
        for i in range(self.seq_len):
            token = x[i].item()
            
            if token == self.assistant_id:
                in_assistant = True
                continue  # don't compute loss on the <|assistant|> token itself
            
            if token == self.tool_result_start_id:
                in_tool_result = True
                in_assistant = False
                continue
            
            if token == self.tool_result_end_id:
                in_tool_result = False
                continue
            
            if token == self.end_id and in_assistant:
                mask[i] = True  # include the <|end|> token (model must learn to produce it)
                in_assistant = False
                continue
            
            if token == self.bos_id:
                in_assistant = False
                in_tool_result = False
                continue
            
            if token == 0 and i > 0 and x[i-1].item() == 0:
                # Padding region
                break
            
            if in_assistant and not in_tool_result:
                mask[i] = True
        
        return x, y, mask