//! JSON Web Signatures, including JWT signing and headers
//!
//! Defined in [RFC 7515](https://tools.ietf.org/html/rfc7515). For most common use,
//! you will want to look at the  [`Compact`](enum.Compact.html) enum.
mod compact;
mod flattened;

pub use compact::Compact;
pub use flattened::{Signable, SignedData};

use crate::errors::Error;
use crate::jwa::SignatureAlgorithm;
use crate::jwk;
use crate::{CompactJson, Empty};

use num_bigint::BigUint;
use ring::signature;
use serde::{self, de::DeserializeOwned, Deserialize, Serialize};
use std::sync::Arc;

/// The secrets used to sign and/or encrypt tokens
#[derive(Clone)]
pub enum Secret {
    /// Used with the `None` algorithm variant.
    None,
    /// Bytes used for HMAC secret. Can be constructed from a string literal
    ///
    /// # Examples
    /// ```
    /// use biscuit::jws::Secret;
    ///
    /// let secret = Secret::bytes_from_str("secret");
    /// ```
    Bytes(Vec<u8>),
    /// An RSA Key pair constructed from a DER-encoded private key
    ///
    /// To generate a private key, use
    ///
    /// ```sh
    /// openssl genpkey -algorithm RSA \
    ///                 -pkeyopt rsa_keygen_bits:2048 \
    ///                 -outform der \
    ///                 -out private_key.der
    /// ```
    ///
    /// Often, keys generated for use in OpenSSL-based software are
    /// encoded in PEM format, which is not supported by *ring*. PEM-encoded
    /// keys that are in `RSAPrivateKey` format can be decoded into the using
    /// an OpenSSL command like this:
    ///
    /// ```sh
    /// openssl rsa -in private_key.pem -outform DER -out private_key.der
    /// ```
    ///
    /// # Examples
    /// ```
    /// use biscuit::jws::Secret;
    ///
    /// let secret = Secret::rsa_keypair_from_file("test/fixtures/rsa_private_key.der");
    /// ```
    RsaKeyPair(Arc<signature::RsaKeyPair>),
    /// An ECDSA Key pair constructed from a PKCS8 DER encoded private key
    ///
    /// To generate a private key, use
    ///
    /// ```sh
    /// openssl ecparam -genkey -name prime256v1 | \
    /// openssl pkcs8 -topk8 -nocrypt -outform DER > ecdsa_private_key.p8
    /// ```
    ///
    /// # Examples
    /// ```
    /// use biscuit::jws::Secret;
    ///
    /// let secret = Secret::ecdsa_keypair_from_file(biscuit::jwa::SignatureAlgorithm::ES256, "test/fixtures/ecdsa_private_key.p8");
    /// ```
    EcdsaKeyPair(Arc<signature::EcdsaKeyPair>),
    /// Bytes of a DER encoded RSA Public Key
    ///
    /// To generate the public key from your DER-encoded private key
    ///
    /// ```sh
    /// openssl rsa -in private_key.der \
    ///             -inform DER
    ///             -RSAPublicKey_out \
    ///             -outform DER \
    ///             -out public_key.der
    /// ```
    ///
    /// To convert a PEM formatted public key
    ///
    /// ```sh
    /// openssl rsa -RSAPublicKey_in \
    ///             -in public_key.pem \
    ///             -inform PEM \
    ///             -outform DER \
    ///             -RSAPublicKey_out \
    ///             -out public_key.der
    /// ```
    ///
    /// Note that the underlying crate (ring) does not support the format used
    /// by OpenSSL. You can check the format using
    ///
    /// ```sh
    /// openssl asn1parse -inform DER -in public_key.der
    /// ```
    ///
    /// It should output something like
    ///
    /// ```sh
    ///     0:d=0  hl=4 l= 290 cons: SEQUENCE
    ///     4:d=1  hl=2 l=  13 cons: SEQUENCE
    ///     6:d=2  hl=2 l=   9 prim: OBJECT            :rsaEncryption
    ///    17:d=2  hl=2 l=   0 prim: NULL
    ///    19:d=1  hl=4 l= 271 prim: BIT STRING
    /// ```
    ///
    /// There is a header here that indicates the content of the file
    /// (a public key for `rsaEncryption`). The actual key is contained
    /// within the BIT STRING at the end. The bare public key can be
    /// extracted with
    ///
    /// ```sh
    /// openssl asn1parse -inform DER \
    ///                   -in public_key.der \
    ///                   -offset 24 \
    ///                   -out public_key_extracted.der
    /// ```
    ///
    /// Run the following to verify that the key is in the right format
    ///
    /// ```sh
    /// openssl asn1parse -inform DER -in public_key_extracted.der
    /// ```
    ///
    /// The right format looks like this (the `<>` elements show the actual
    /// numbers)
    ///
    /// ```sh
    ///     0:d=0  hl=4 l= 266 cons: SEQUENCE
    ///     4:d=1  hl=4 l= 257 prim: INTEGER           :<public key modulus>
    ///   265:d=1  hl=2 l=   3 prim: INTEGER           :<public key exponent>
    /// ```
    ///
    /// Every other format will be rejected by ring with an unspecified error.
    /// Note that OpenSSL is no longer able to interpret this file as a public key,
    /// since it no longer contains the expected header.
    ///
    /// # Examples
    /// ```
    /// use biscuit::jws::Secret;
    ///
    /// let secret = Secret::public_key_from_file("test/fixtures/rsa_public_key.der");
    Ed25519KeyPair(Arc<signature::Ed25519KeyPair>),
    /// An Ed25519 Key pair constructed from a PKCS8 DER encoded private key
    ///
    /// To generate a private key, use
    ///
    /// ```sh
    /// openssl genpkey -algorithm ed25519 -outform DER -out test25519.der
    /// ```
    ///
    /// # Examples
    /// ```
    /// use biscuit::jws::Secret;
    ///
    /// let secret = Secret::ecdsa_keypair_from_file(biscuit::jwa::SignatureAlgorithm::ES256, "test/fixtures/ecdsa_private_key.p8");
    /// ```
    PublicKey(Vec<u8>),
    /// Use the modulus (`n`) and exponent (`e`) of an RSA key directly
    ///
    /// These parameters can be obtained from a JWK directly using
    /// [`jwk::RSAKeyParameters::jws_public_key_secret`]
    RSAModulusExponent {
        /// RSA modulus
        n: BigUint,
        /// RSA exponent
        e: BigUint,
    },
}

impl Secret {
    fn read_bytes(path: &str) -> Result<Vec<u8>, Error> {
        use std::fs::File;
        use std::io::prelude::*;

        let mut file = File::open(path)?;
        let metadata = file.metadata()?;
        let mut bytes: Vec<u8> = Vec::with_capacity(metadata.len() as usize);
        let _ = file.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// Convenience function to create a secret bytes array from a string
    /// See example in the [`Secret::Bytes`] variant documentation for usage.
    pub fn bytes_from_str(secret: &str) -> Self {
        Secret::Bytes(secret.to_string().into_bytes())
    }

    /// Convenience function to get the RSA Keypair from a DER encoded RSA private key.
    /// See example in the [`Secret::RsaKeyPair`] variant documentation for usage.
    pub fn rsa_keypair_from_file(path: &str) -> Result<Self, Error> {
        let der = Self::read_bytes(path)?;
        let key_pair = signature::RsaKeyPair::from_der(der.as_slice())?;
        Ok(Secret::RsaKeyPair(Arc::new(key_pair)))
    }

    /// Convenience function to get the ECDSA Keypair from a PKCS8-DER encoded EC private key.
    pub fn ecdsa_keypair_from_file(
        algorithm: SignatureAlgorithm,
        path: &str,
    ) -> Result<Self, Error> {
        let der = Self::read_bytes(path)?;
        let ring_algorithm = match algorithm {
            SignatureAlgorithm::ES256 => &signature::ECDSA_P256_SHA256_FIXED_SIGNING,
            SignatureAlgorithm::ES384 => &signature::ECDSA_P384_SHA384_FIXED_SIGNING,
            _ => return Err(Error::UnsupportedOperation),
        };
        let key_pair = signature::EcdsaKeyPair::from_pkcs8(ring_algorithm, der.as_slice())?;
        Ok(Secret::EcdsaKeyPair(Arc::new(key_pair)))
    }

    /// Convenience function to create a Public key from a DER encoded RSA or ECDSA public key
    /// See examples in the [`Secret::PublicKey`] variant documentation for usage.
    pub fn public_key_from_file(path: &str) -> Result<Self, Error> {
        let der = Self::read_bytes(path)?;
        Ok(Secret::PublicKey(der.to_vec()))
    }
}

impl From<jwk::RSAKeyParameters> for Secret {
    fn from(rsa: jwk::RSAKeyParameters) -> Self {
        rsa.jws_public_key_secret()
    }
}

/// JWS Header, consisting of the registered fields and other custom fields
#[derive(Debug, Eq, PartialEq, Clone, Default, Serialize, Deserialize)]
pub struct Header<T> {
    /// Registered header fields
    #[serde(flatten)]
    pub registered: RegisteredHeader,
    /// Private header fields
    #[serde(flatten)]
    pub private: T,
}

impl<T: Serialize + DeserializeOwned> CompactJson for Header<T> {}

impl Header<Empty> {
    /// Convenience function to create a header with only registered headers
    pub fn from_registered_header(registered: RegisteredHeader) -> Self {
        Self {
            registered,
            ..Default::default()
        }
    }
}

impl From<RegisteredHeader> for Header<Empty> {
    fn from(registered: RegisteredHeader) -> Self {
        Self::from_registered_header(registered)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// Registered JWS header fields.
/// The alg defaults to HS256 and typ is automatically
/// set to `JWT`. All the other fields are optional.
/// The fields are defined by [RFC7519#5](https://tools.ietf.org/html/rfc7519#section-5) and additionally in
/// [RFC7515#4.1](https://tools.ietf.org/html/rfc7515#section-4.1).
// TODO: Implement verification for registered headers and support custom headers
pub struct RegisteredHeader {
    /// Algorithms, as defined in [RFC 7518](https://tools.ietf.org/html/rfc7518), used to sign or encrypt the JWT
    /// Serialized to `alg`.
    /// Defined in [RFC7515#4.1.1](https://tools.ietf.org/html/rfc7515#section-4.1.1).
    #[serde(rename = "alg")]
    pub algorithm: SignatureAlgorithm,

    /// Media type of the complete JWS. Serialized to `typ`.
    /// Defined in [RFC7519#5.1](https://tools.ietf.org/html/rfc7519#section-5.1) and additionally
    /// [RFC7515#4.1.9](https://tools.ietf.org/html/rfc7515#section-4.1.9).
    /// The "typ" value "JOSE" can be used by applications to indicate that
    /// this object is a JWS or JWE using the JWS Compact Serialization or
    /// the JWE Compact Serialization.  The "typ" value "JOSE+JSON" can be
    /// used by applications to indicate that this object is a JWS or JWE
    /// using the JWS JSON Serialization or the JWE JSON Serialization.
    /// Other type values can also be used by applications.
    #[serde(rename = "typ", skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,

    /// Content Type of the secured payload.
    /// Typically used to indicate the presence of a nested JOSE object which is signed or encrypted.
    /// Serialized to `cty`.
    /// Defined in [RFC7519#5.2](https://tools.ietf.org/html/rfc7519#section-5.2) and additionally
    /// [RFC7515#4.1.10](https://tools.ietf.org/html/rfc7515#section-4.1.10).
    #[serde(rename = "cty", skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// The JSON Web Key Set URL. This is currently not implemented (correctly).
    /// Serialized to `jku`.
    /// Defined in [RFC7515#4.1.2](https://tools.ietf.org/html/rfc7515#section-4.1.2).
    #[serde(rename = "jku", skip_serializing_if = "Option::is_none")]
    pub web_key_url: Option<String>,

    /// The JSON Web Key.
    /// Serialized to `jwk`.
    /// Defined in [RFC7515#4.1.3](https://tools.ietf.org/html/rfc7515#section-4.1.3).
    #[serde(rename = "jwk", skip_serializing_if = "Option::is_none")]
    pub web_key: Option<jwk::JWK<Empty>>,

    /// The Key ID. This is currently not implemented (correctly).
    /// Serialized to `kid`.
    /// Defined in [RFC7515#4.1.3](https://tools.ietf.org/html/rfc7515#section-4.1.3).
    #[serde(rename = "kid", skip_serializing_if = "Option::is_none")]
    pub key_id: Option<String>,

    /// X.509 Public key cerfificate URL. This is currently not implemented (correctly).
    /// Serialized to `x5u`.
    /// Defined in [RFC7515#4.1.5](https://tools.ietf.org/html/rfc7515#section-4.1.5).
    #[serde(rename = "x5u", skip_serializing_if = "Option::is_none")]
    pub x509_url: Option<String>,

    /// X.509 public key certificate chain. This is currently not implemented (correctly).
    /// Serialized to `x5c`.
    /// Defined in [RFC7515#4.1.6](https://tools.ietf.org/html/rfc7515#section-4.1.6).
    #[serde(rename = "x5c", skip_serializing_if = "Option::is_none")]
    pub x509_chain: Option<Vec<String>>,

    /// X.509 Certificate thumbprint. This is currently not implemented (correctly).
    /// Also not implemented, is the SHA-256 thumbprint variant of this header.
    /// Serialized to `x5t`.
    /// Defined in [RFC7515#4.1.7](https://tools.ietf.org/html/rfc7515#section-4.1.7).
    // TODO: How to make sure the headers are mutually exclusive?
    #[serde(rename = "x5t", skip_serializing_if = "Option::is_none")]
    pub x509_fingerprint: Option<String>,

    /// List of critical extended headers.
    /// This is currently not implemented (correctly).
    /// Serialized to `crit`.
    /// Defined in [RFC7515#4.1.11](https://tools.ietf.org/html/rfc7515#section-4.1.11).
    #[serde(rename = "crit", skip_serializing_if = "Option::is_none")]
    pub critical: Option<Vec<String>>,
}

impl Default for RegisteredHeader {
    fn default() -> RegisteredHeader {
        RegisteredHeader {
            algorithm: SignatureAlgorithm::default(),
            media_type: Some("JWT".to_string()),
            content_type: None,
            web_key_url: None,
            web_key: None,
            key_id: None,
            x509_url: None,
            x509_chain: None,
            x509_fingerprint: None,
            critical: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RegisteredHeader;

    #[test]
    fn header_serialization_round_trip_no_optional() {
        let expected = RegisteredHeader::default();
        let expected_json = r#"{"alg":"HS256","typ":"JWT"}"#;

        let encoded = not_err!(serde_json::to_string(&expected));
        assert_eq!(expected_json, encoded);

        let decoded: RegisteredHeader = not_err!(serde_json::from_str(&encoded));
        assert_eq!(decoded, expected);
    }

    #[test]
    fn header_serialization_round_trip_with_optional() {
        let expected = RegisteredHeader {
            key_id: Some("kid".to_string()),
            ..Default::default()
        };

        let expected_json = r#"{"alg":"HS256","typ":"JWT","kid":"kid"}"#;

        let encoded = not_err!(serde_json::to_string(&expected));
        assert_eq!(expected_json, encoded);

        let decoded: RegisteredHeader = not_err!(serde_json::from_str(&encoded));
        assert_eq!(decoded, expected);
    }
}
