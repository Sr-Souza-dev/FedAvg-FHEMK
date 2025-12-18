from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from Pyfhel import PyCtxt, Pyfhel


@dataclass
class CKKSConfig:
    poly_mod_degree: int = 2 ** 14
    scale: float = 2 ** 30
    qi_sizes: Sequence[int] | None = (60, 40, 40, 40, 60)


class SharedCKKSContext:
    """Utility responsible for generating/loading a shared CKKS context."""

    def __init__(self, base_dir: str | Path, config: CKKSConfig | None = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or CKKSConfig()
        self._ctx_file = self.base_dir / "context.ckks"
        self._pk_file = self.base_dir / "public.key"
        self._sk_file = self.base_dir / "private.key"
        self._relin_file = self.base_dir / "relin.key"
        self._rotate_file = self.base_dir / "rotate.key"

    # ------------------------------------------------------------------
    # Context bootstrap
    # ------------------------------------------------------------------
    def ensure_keys(self) -> None:
        if all(
            path.exists()
            for path in [
                self._ctx_file,
                self._pk_file,
                self._sk_file,
                self._relin_file,
                self._rotate_file,
            ]
        ):
            return
        he = Pyfhel()
        ctx_kwargs = {
            "scheme": "CKKS",
            "n": self.config.poly_mod_degree,
            "scale": self.config.scale,
        }
        if self.config.qi_sizes:
            ctx_kwargs["qi_sizes"] = list(self.config.qi_sizes)
        he.contextGen(**ctx_kwargs)
        he.keyGen()
        he.relinKeyGen()
        he.rotateKeyGen()
        he.save_context(str(self._ctx_file))
        he.save_public_key(str(self._pk_file))
        he.save_secret_key(str(self._sk_file))
        he.save_relin_key(str(self._relin_file))
        he.save_rotate_key(str(self._rotate_file))

    def build_he(self, with_secret: bool = True) -> Pyfhel:
        self.ensure_keys()
        he = Pyfhel()
        he.load_context(str(self._ctx_file))
        he.load_public_key(str(self._pk_file))
        he.load_relin_key(str(self._relin_file))
        he.load_rotate_key(str(self._rotate_file))
        if with_secret:
            he.load_secret_key(str(self._sk_file))
        return he

    @property
    def slot_count(self) -> int:
        return self.config.poly_mod_degree // 2

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def encrypt_vector(self, he: Pyfhel, vector: np.ndarray) -> List[np.ndarray]:
        """Encrypt an array and return a Flower-friendly payload."""
        vec = np.asarray(vector, dtype=np.float64)
        chunks = self._chunks(vec)
        payload: List[np.ndarray] = [np.array([len(vec)], dtype=np.int64)]
        for chunk in chunks:
            ctxt = he.encryptFrac(chunk)
            payload.append(self._serialize_ciphertext(ctxt))
        return payload

    def decrypt_vector(self, he: Pyfhel, ciphertexts: Iterable[PyCtxt], original_length: int) -> np.ndarray:
        plain: List[float] = []
        for ctxt in ciphertexts:
            plain.extend(he.decryptFrac(ctxt))
        return np.array(plain[:original_length], dtype=np.float64)

    def deserialize_ciphertexts(
        self, payload: List[np.ndarray], he: Pyfhel
    ) -> Tuple[List[PyCtxt], int]:
        if not payload:
            return [], 0
        original_len = int(np.asarray(payload[0], dtype=np.int64)[0])
        ctxts: List[PyCtxt] = []
        for blob in payload[1:]:
            ctxts.append(PyCtxt(pyfhel=he, bytestring=blob.tobytes()))
        return ctxts, original_len

    def serialize_ciphertexts(self, ciphertexts: Iterable[PyCtxt], original_length: int) -> List[np.ndarray]:
        payload: List[np.ndarray] = [np.array([original_length], dtype=np.int64)]
        payload.extend(self._serialize_ciphertext(ctxt) for ctxt in ciphertexts)
        return payload

    def add_ciphertext_lists(
        self, accumulators: List[PyCtxt], increments: List[PyCtxt]
    ) -> List[PyCtxt]:
        if not accumulators:
            return list(increments)
        return [a + b for a, b in zip(accumulators, increments)]

    def scale_ciphertexts(
        self, ciphertexts: List[PyCtxt], scalar: float
    ) -> List[PyCtxt]:
        return [ctxt * scalar for ctxt in ciphertexts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _chunks(self, vec: np.ndarray) -> List[np.ndarray]:
        slot = self.slot_count
        if vec.size == 0:
            return [np.zeros(slot, dtype=np.float64)]
        chunks: List[np.ndarray] = []
        for start in range(0, vec.size, slot):
            chunk = vec[start : start + slot]
            if chunk.size < slot:
                pad = np.zeros(slot - chunk.size, dtype=np.float64)
                chunk = np.concatenate([chunk, pad])
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _serialize_ciphertext(ctxt: PyCtxt) -> np.ndarray:
        data = ctxt.to_bytes()
        return np.frombuffer(data, dtype=np.uint8)


def build_shared_context() -> SharedCKKSContext:
    base_dir = Path(__file__).resolve().parents[2] / "keys"
    return SharedCKKSContext(base_dir=base_dir)
