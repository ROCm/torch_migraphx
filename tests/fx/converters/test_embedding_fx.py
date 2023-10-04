import pytest
import torch
from fx_test_utils import convert_to_mgx, verify_outputs


@pytest.mark.parametrize("num_embed, embed_dim", [(10, 5), (6, 20), (100, 64)])
def test_embedding(num_embed, embed_dim):
    inp = torch.tensor([[0, 5, 2], [1, 1, 3]])

    mod = torch.nn.Embedding(num_embeddings=num_embed,
                             embedding_dim=embed_dim)

    mgx_mod = convert_to_mgx(mod, [inp])
    verify_outputs(mod, mgx_mod, inp)