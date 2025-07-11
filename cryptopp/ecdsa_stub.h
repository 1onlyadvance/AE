#pragma once

// Minimal CryptoPP ECDSA, ECP, SHA256 stub for build
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace cryptopp_stub {

namespace CryptoPP {

class ECP {};
class SHA256 {
public:
    SHA256() {}
    typedef int HashTransformation;
};

namespace ASN1 {
    inline int secp256r1() { return 0; }
    class OID {};
}

template <class EC, class HASH>
class ECDSA {
public:
    class PrivateKey {
    public:
        PrivateKey() {}
        void Initialize(int, int) {}
        void MakePublicKey(class PublicKey&) {}
    };
    class PublicKey {
    public:
        PublicKey() {}
    };
    class Signer {
    public:
        Signer(const PrivateKey&) {}
        size_t MaxSignatureLength() const { return 64; }
    };
    class Verifier {
    public:
        Verifier(const PublicKey&) {}
    };
};

} // namespace CryptoPP

} // namespace cryptopp_stub
