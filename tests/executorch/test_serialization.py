import struct

import pytest

from torch_migraphx.executorch.serialization import (
    HEADER_SIZE,
    MIGRAPHX_MAGIC,
    MIGraphXBlobMetadata,
    MIGraphXTensorSpec,
    deserialize_program,
    serialize_program,
)


def _metadata():
    return MIGraphXBlobMetadata(
        io_bindings=[
            MIGraphXTensorSpec(
                name="x",
                dtype="float_type",
                shape=[2, 3],
                strides=[3, 1],
                is_input=True,
            ),
            MIGraphXTensorSpec(
                name="output_0",
                dtype="float_type",
                shape=[2, 3],
                strides=[3, 1],
                is_input=False,
            ),
        ],
        target_arch="gfx942",
        device_id=2,
        migraphx_version="prototype",
    )


def test_mg01_round_trip():
    blob = serialize_program(b"compiled-migraphx-program", _metadata())

    assert blob[:4] == MIGRAPHX_MAGIC
    program, metadata = deserialize_program(blob)
    assert program == b"compiled-migraphx-program"
    assert metadata.target_arch == "gfx942"
    assert metadata.device_id == 2
    assert [binding.name for binding in metadata.io_bindings] == [
        "x",
        "output_0",
    ]
    assert [binding.is_input for binding in metadata.io_bindings] == [
        True,
        False,
    ]


def test_program_payload_is_16_byte_aligned():
    blob = serialize_program(b"program", _metadata())
    _, _, _, program_offset, _, _ = struct.unpack(
        "<4sIIIQ8s", blob[:HEADER_SIZE]
    )
    assert program_offset % 16 == 0


@pytest.mark.parametrize(
    "blob, message",
    [
        (b"", "smaller"),
        (b"NOPE" + b"\x00" * (HEADER_SIZE - 4), "magic"),
    ],
)
def test_invalid_blob_is_rejected(blob, message):
    with pytest.raises(ValueError, match=message):
        deserialize_program(blob)
