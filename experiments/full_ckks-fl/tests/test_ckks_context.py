from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fl_simulation.crypto import ckks_context as ckks_module
from fl_simulation.crypto.ckks_context import CKKSConfig, SharedCKKSContext


def _key_paths(context: SharedCKKSContext) -> list:
    return [
        context._ctx_file,
        context._pk_file,
        context._sk_file,
        context._relin_file,
        context._rotate_file,
    ]


@pytest.fixture(autouse=True, scope="module")
def fake_pyfhel():
    monkeypatch = pytest.MonkeyPatch()
    class FakePyCtxt:
        def __init__(self, pyfhel=None, serialized=None, data=None):
            if serialized is not None:
                self.data = np.frombuffer(serialized, dtype=np.float64).copy()
            elif data is not None:
                self.data = np.array(data, dtype=np.float64)
            else:
                self.data = np.array([], dtype=np.float64)

        def to_bytes(self) -> bytes:
            return self.data.astype(np.float64).tobytes()

        def __add__(self, other: "FakePyCtxt") -> "FakePyCtxt":
            return FakePyCtxt(data=self.data + other.data)

        def __radd__(self, other: "FakePyCtxt") -> "FakePyCtxt":
            return self.__add__(other)

        def __mul__(self, scalar: float) -> "FakePyCtxt":
            return FakePyCtxt(data=self.data * scalar)

        __rmul__ = __mul__

    class FakePyfhel:
        def __init__(self):
            self.with_secret = False

        def contextGen(self, **kwargs):
            return None

        def keyGen(self):
            return None

        def relinKeyGen(self):
            return None

        def rotateKeyGen(self):
            return None

        def save_context(self, path: str):
            Path(path).write_bytes(b"context")

        def save_public_key(self, path: str):
            Path(path).write_bytes(b"public")

        def save_secret_key(self, path: str):
            Path(path).write_bytes(b"secret")

        def save_relin_key(self, path: str):
            Path(path).write_bytes(b"relin")

        def save_rotate_key(self, path: str):
            Path(path).write_bytes(b"rotate")

        def load_context(self, path: str):
            Path(path).read_bytes()

        def load_public_key(self, path: str):
            Path(path).read_bytes()

        def load_secret_key(self, path: str):
            Path(path).read_bytes()
            self.with_secret = True

        def load_relin_key(self, path: str):
            Path(path).read_bytes()

        def load_rotate_key(self, path: str):
            Path(path).read_bytes()

        def encryptFrac(self, values):
            return FakePyCtxt(data=np.array(values, dtype=np.float64))

        def decryptFrac(self, ctxt: FakePyCtxt):
            return ctxt.data.copy()

    monkeypatch.setattr(ckks_module, "Pyfhel", FakePyfhel)
    monkeypatch.setattr(ckks_module, "PyCtxt", FakePyCtxt)
    yield FakePyfhel, FakePyCtxt
    monkeypatch.undo()


@pytest.fixture(scope="module")
def ckks_context(tmp_path_factory: pytest.TempPathFactory) -> SharedCKKSContext:
    """Provision a CKKS context for the full_ckks-fl experiment tests."""
    base_dir = tmp_path_factory.mktemp("full_ckks_ctx")
    context = SharedCKKSContext(base_dir=base_dir, config=CKKSConfig())
    context.ensure_keys()
    return context


@pytest.fixture()
def he(ckks_context: SharedCKKSContext):
    """Return a Pyfhel instance with the secret key loaded."""
    return ckks_context.build_he(with_secret=True)


def test_ensure_keys_is_idempotent(tmp_path):
    context = SharedCKKSContext(base_dir=tmp_path, config=CKKSConfig())
    context.ensure_keys()
    first_snapshot = {path.name: path.stat().st_mtime for path in _key_paths(context)}
    context.ensure_keys()
    second_snapshot = {path.name: path.stat().st_mtime for path in _key_paths(context)}
    assert first_snapshot == second_snapshot


def test_encrypt_decrypt_roundtrip(ckks_context: SharedCKKSContext, he):
    vector = np.random.random(257).astype(np.float64)
    payload = ckks_context.encrypt_vector(he, vector)
    ciphertexts, original_len = ckks_context.deserialize_ciphertexts(payload, he)

    assert original_len == vector.size

    decrypted = ckks_context.decrypt_vector(he, ciphertexts, original_len)
    np.testing.assert_allclose(decrypted, vector, atol=1e-6)


def test_serialize_deserialize_symmetry(ckks_context: SharedCKKSContext, he):
    vector = np.random.random(1024).astype(np.float64)
    payload = ckks_context.encrypt_vector(he, vector)
    ciphertexts, original_len = ckks_context.deserialize_ciphertexts(payload, he)

    reserialized = ckks_context.serialize_ciphertexts(ciphertexts, original_len)

    assert len(reserialized) == len(payload)
    for lhs, rhs in zip(payload, reserialized):
        np.testing.assert_array_equal(lhs, rhs)


def test_ciphertext_addition_and_scaling(ckks_context: SharedCKKSContext, he):
    vec_a = np.random.random(128).astype(np.float64)
    vec_b = np.random.random(128).astype(np.float64)

    payload_a = ckks_context.encrypt_vector(he, vec_a)
    payload_b = ckks_context.encrypt_vector(he, vec_b)

    ctxt_a, length_a = ckks_context.deserialize_ciphertexts(payload_a, he)
    ctxt_b, length_b = ckks_context.deserialize_ciphertexts(payload_b, he)

    assert length_a == length_b == 128

    scaled_a = ckks_context.scale_ciphertexts(ctxt_a, 2)
    scaled_b = ckks_context.scale_ciphertexts(ctxt_b, 1)

    summed = ckks_context.add_ciphertext_lists(scaled_a, scaled_b)
    averaged = ckks_context.scale_ciphertexts(summed, 1 / 3)

    decrypted = ckks_context.decrypt_vector(he, averaged, length_a)
    expected = (2 * vec_a + vec_b) / 3

    np.testing.assert_allclose(decrypted, expected, atol=1e-4)
