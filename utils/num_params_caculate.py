
def num_params_caculate(vocab_size, n_embd, block_size, n_block):
  num_token_table = vocab_size * n_embd
  num_pos_table = block_size * n_embd
  num_block = n_block * (n_embd * 2 + n_embd * n_embd * 3 + n_embd * n_embd + n_embd * 2 + n_embd * n_embd * 4 * 2)
  num_layernorm = n_embd * 2
  num_retoken_table = n_embd * vocab_size + vocab_size
  return num_token_table \
          + num_pos_table \
            + num_block \
              + num_layernorm \
                + num_retoken_table


# print(num_params_caculate(65, 384, 512, 6))

# GPT-3-175B
# 预训练数据集大小：300B， batch_size = 3.2M, num_heads = 96
# print(num_params_caculate(5e4, 12288, 2048, 96))
