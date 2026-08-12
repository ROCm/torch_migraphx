"""Wire format shared by the ExecuTorch AOT and native MIGraphX backends."""

from __future__ import annotations

import dataclasses
import json
import struct
from dataclasses import dataclass, field
from typing import List, Tuple


MIGRAPHX_MAGIC = b"MG01"
HEADER_FORMAT = "<4sIIIQ8s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
PROGRAM_ALIGNMENT = 16


def _align(offset: int, alignment: int = PROGRAM_ALIGNMENT) -> int:
    return (offset + alignment - 1) & ~(alignment - 1)


@dataclass
class MIGraphXTensorSpec:
    """A positional tensor binding in a compiled MIGraphX program."""

    name: str
    dtype: str
    shape: List[int] = field(default_factory=list)
    strides: List[int] = field(default_factory=list)
    is_input: bool = True


@dataclass
class MIGraphXBlobMetadata:
    """Metadata required to bind and validate a serialized MIGraphX program."""

    io_bindings: List[MIGraphXTensorSpec] = field(default_factory=list)
    target_arch: str = ""
    device_id: int = 0
    migraphx_version: str = ""
    compiled: bool = True

    def to_json(self) -> bytes:
        data = {
            "io_bindings": [dataclasses.asdict(binding) for binding in self.io_bindings],
            "target_arch": self.target_arch,
            "device_id": self.device_id,
            "migraphx_version": self.migraphx_version,
            "compiled": self.compiled,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes) -> "MIGraphXBlobMetadata":
        parsed = json.loads(data.decode("utf-8"))
        binding_fields = {item.name for item in dataclasses.fields(MIGraphXTensorSpec)}
        bindings = [
            MIGraphXTensorSpec(
                **{key: value for key, value in binding.items() if key in binding_fields}
            )
            for binding in parsed.get("io_bindings", [])
        ]
        return cls(
            io_bindings=bindings,
            target_arch=str(parsed.get("target_arch", "")),
            device_id=int(parsed.get("device_id", 0)),
            migraphx_version=str(parsed.get("migraphx_version", "")),
            compiled=bool(parsed.get("compiled", True)),
        )


def serialize_program(
    program_bytes: bytes, metadata: MIGraphXBlobMetadata
) -> bytes:
    """Serialize a MIGraphX program and its binding metadata as an MG01 blob."""

    metadata_json = metadata.to_json()
    metadata_offset = HEADER_SIZE
    program_offset = _align(metadata_offset + len(metadata_json))
    reserved = b"\x01" + b"\x00" * 7
    header = struct.pack(
        HEADER_FORMAT,
        MIGRAPHX_MAGIC,
        metadata_offset,
        len(metadata_json),
        program_offset,
        len(program_bytes),
        reserved,
    )
    padding = b"\x00" * (
        program_offset - metadata_offset - len(metadata_json)
    )
    return header + metadata_json + padding + program_bytes


def deserialize_program(
    blob: bytes,
) -> Tuple[bytes, MIGraphXBlobMetadata]:
    """Validate and deserialize an MG01 blob."""

    if len(blob) < HEADER_SIZE:
        raise ValueError(f"Blob is smaller than the {HEADER_SIZE}-byte header")

    (
        magic,
        metadata_offset,
        metadata_size,
        program_offset,
        program_size,
        _,
    ) = struct.unpack(HEADER_FORMAT, blob[:HEADER_SIZE])

    if magic != MIGRAPHX_MAGIC:
        raise ValueError(f"Invalid MIGraphX blob magic: {magic!r}")
    if metadata_offset < HEADER_SIZE:
        raise ValueError("Metadata offset points inside the header")
    if program_offset % PROGRAM_ALIGNMENT != 0:
        raise ValueError(
            f"Program offset {program_offset} is not {PROGRAM_ALIGNMENT}-byte aligned"
        )
    if metadata_offset + metadata_size > len(blob):
        raise ValueError("Metadata extends beyond the blob")
    if program_offset + program_size > len(blob):
        raise ValueError("Program extends beyond the blob")
    if metadata_offset + metadata_size > program_offset:
        raise ValueError("Metadata overlaps the program payload")

    metadata = MIGraphXBlobMetadata.from_json(
        blob[metadata_offset : metadata_offset + metadata_size]
    )
    program = blob[program_offset : program_offset + program_size]
    return program, metadata
