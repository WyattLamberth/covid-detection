import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Args:
        batch: List of (image_tensor, caption_sequence) tuples

    Returns:
        images: Tensor of shape [batch_size, 3, 224, 224]
        captions: LongTensor of shape [batch_size, max_seq_len]
    """
    images, captions = zip(*batch)  # unzip list of tuples

    # Stack image tensors into [B, 3, 224, 224]
    images = torch.stack(images, dim=0)

    # Convert list of lists into padded tensor [B, max_len]
    captions = [torch.tensor(seq, dtype=torch.long) for seq in captions]
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0)  # 0 = <pad>

    return images, padded_captions