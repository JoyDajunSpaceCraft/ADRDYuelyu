from abc import ABCMeta, abstractmethod

from cryptography.hazmat.primitives.serialization import Encoding, KeySerializationEncryption, PrivateFormat, PublicFormat

class Ed25519PrivateKey(metaclass=ABCMeta):
    @classmethod
    def generate(cls) -> Ed25519PrivateKey: ...
    @classmethod
    def from_private_bytes(cls, data: bytes) -> Ed25519PrivateKey: ...
    @abstractmethod
    def private_bytes(
        self, encoding: Encoding, format: PrivateFormat, encryption_algorithm: KeySerializationEncryption
    ) -> bytes: ...
    @abstractmethod
    def public_key(self) -> Ed25519PublicKey: ...
    @abstractmethod
    def sign(self, data: bytes) -> bytes: ...

class Ed25519PublicKey(metaclass=ABCMeta):
    @classmethod
    def from_public_bytes(cls, data: bytes) -> Ed25519PublicKey: ...
    @abstractmethod
    def public_bytes(self, encoding: Encoding, format: PublicFormat) -> bytes: ...
    @abstractmethod
    def verify(self, signature: bytes, data: bytes) -> None: ...
