import pytest
import torch
from fx_test_utils import convert_to_mgx, verify_outputs


@pytest.mark.skip(reason="linalg.norm converter not implemented")
def test_linalg_norm():
    pass
