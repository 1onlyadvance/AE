// CryptoPP ECDSA stub for CUDA build compatibility
#pragma once

namespace CryptoPP {

struct ECP {};
struct SHA256 {
    SHA256() {}
    typedef int HashTransformation;
};

namespace ASN1 {
    inline int secp256r1() { return 0; }
    struct OID {};
}

template <class EC, class HASH>
struct ECDSA {
    struct PrivateKey {
        __host__ __device__ PrivateKey() {}
        __host__ __device__ void Initialize(int, int) {}
        __host__ __device__ void MakePublicKey(struct PublicKey&) {}
    };
    struct PublicKey {
        __host__ __device__ PublicKey() {}
    };
    struct Signer {
        __host__ __device__ Signer(const PrivateKey&) {}
        __host__ __device__ size_t MaxSignatureLength() const { return 64; }
    };
    struct Verifier {
        __host__ __device__ Verifier(const PublicKey&) {}
    };
};

} // namespace CryptoPP
