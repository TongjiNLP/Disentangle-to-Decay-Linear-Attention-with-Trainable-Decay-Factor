import torch

def sinusoidal_pos_emd(batch_size:int,nums_head:int,max_len:int,output_dim:int,device):
    trap = "trap"
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # position = [0,1,2....max_len] e.g [1,12,61,64]
    ids = torch.arange(0, output_dim//2, dtype=torch.float)
    # ids = [0,1,2...floor(\frac{d}{2})
    theta = torch.pow(10000, -2 * ids / output_dim)
    # 在\theta_i的选择上，我们同样沿用了Sinusoidal位置编码的方案，即
    # \theta_i = 10000^{\frac{-2i}{d}}
    # theta _size = [outdim//2]
    embeds = position*theta
    # broadcast detials: postision[max_len,1] got [[1],[2],..,[max_len]] and result is [[i*theta (head_dim//2)],...,] and finally produces [max_len,head_dim//2]
    # now embeds is $m\theta_0,...m\theta_{d/2-1}$ in 13 of https://kexue.fm/archives/8265  which need to be cosed

    embeds = torch.stack([torch.sin(embeds), torch.cos(embeds)], dim=-1)
    # in this, each element of sin(embeds) are along with cos(embeds) with same pos
    # element level manipulation do not change size
    # embeds now is [max_len,hidden_dim//2,2]
    embeds = embeds.repeat((batch_size, nums_head, *([1] * len(embeds.shape))))
    # now we got [batch_size,nums_head] 个 same [max_len,nums_head,2] embeds
    embeds = torch.reshape(embeds,(batch_size,nums_head,max_len,output_dim))
    # in this ,batch_szie x num_head same [max_len,output_dim],
    # and in the i-length , got [cos i theta_0,sin i theta_0,...cos i theta_{d/2-1}, sin i \theta_{d/2-1},]
    embeds = embeds.to(device)
    return embeds

def ROPE(query:torch.Tensor,k:torch.Tensor):
    # # query  and length [batch_size, num_heads, seq_len, head_size]
    batch_size = query.size(0)
    num_heads = query.size(1)
    max_len = query.size(2)
    head_size = query.size(3)
    pos_emb = sinusoidal_pos_emd(batch_size,num_heads,max_len,head_size,query.device)
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
    # Stack them to the one and then tear them apart,..., ridiculous
    q2 = torch.stack([-query[..., 1::2], query[..., ::2]], dim=-1)
    # one more time, to form row1,3 e.q.(13) of https://kexue.fm/archives/8265
    q2 = q2.reshape(query.shape)
    # last time of e.q.(13) 1 ele_mul 2 + 3 elemul 4
    query = query * cos_pos + q2 * sin_pos
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos
    # Rope
    return query,k