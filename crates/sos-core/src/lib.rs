//! Deterministic, dependency-light foundations for SOS Decompiler.
//!
//! This crate deliberately does not contain AI logic. It owns facts extracted from
//! an artifact so later semantic layers can attach evidence without changing the
//! underlying bytes.

use std::{fmt, fs, path::Path};

pub const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
pub const EM_BPF: u16 = 0x107;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfFacts {
    pub class: u8,
    pub data_encoding: u8,
    pub object_type: u16,
    pub machine: u16,
    pub entrypoint: u64,
    pub program_header_offset: u64,
    pub section_header_offset: u64,
}

impl ElfFacts {
    pub fn is_sbpf(&self) -> bool {
        self.class == 2 && self.data_encoding == 1 && self.machine == EM_BPF
    }
}

#[derive(Debug)]
pub enum ArtifactError {
    Io(std::io::Error),
    NotElf,
    TruncatedElf,
    UnsupportedClass(u8),
    UnsupportedEncoding(u8),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::NotElf => write!(f, "file does not begin with an ELF magic number"),
            Self::TruncatedElf => write!(f, "ELF header is truncated"),
            Self::UnsupportedClass(value) => write!(f, "unsupported ELF class: {value}"),
            Self::UnsupportedEncoding(value) => write!(f, "unsupported ELF encoding: {value}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

impl From<std::io::Error> for ArtifactError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn read_elf_facts(path: impl AsRef<Path>) -> Result<ElfFacts, ArtifactError> {
    let bytes = fs::read(path)?;
    parse_elf_facts(&bytes)
}

pub fn parse_elf_facts(bytes: &[u8]) -> Result<ElfFacts, ArtifactError> {
    if bytes.len() < 64 {
        return Err(ArtifactError::TruncatedElf);
    }
    if bytes[0..4] != ELF_MAGIC {
        return Err(ArtifactError::NotElf);
    }

    let class = bytes[4];
    if class != 2 {
        return Err(ArtifactError::UnsupportedClass(class));
    }
    let data_encoding = bytes[5];
    if data_encoding != 1 {
        return Err(ArtifactError::UnsupportedEncoding(data_encoding));
    }

    Ok(ElfFacts {
        class,
        data_encoding,
        object_type: read_u16(bytes, 16),
        machine: read_u16(bytes, 18),
        entrypoint: read_u64(bytes, 24),
        program_header_offset: read_u64(bytes, 32),
        section_header_offset: read_u64(bytes, 40),
    })
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_little_endian_elf_header() {
        let mut bytes = [0u8; 64];
        bytes[0..4].copy_from_slice(&ELF_MAGIC);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[16..18].copy_from_slice(&3u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&EM_BPF.to_le_bytes());
        bytes[24..32].copy_from_slice(&0x1000u64.to_le_bytes());

        let facts = parse_elf_facts(&bytes).expect("valid ELF header");
        assert!(facts.is_sbpf());
        assert_eq!(facts.entrypoint, 0x1000);
    }

    #[test]
    fn rejects_non_elf_input() {
        assert_eq!(
            parse_elf_facts(&[0u8; 64]).unwrap_err().to_string(),
            "file does not begin with an ELF magic number"
        );
    }
}
