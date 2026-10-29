//! Artifact hashing, manifest generation, and content-addressed storage.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use sha2::{Digest, Sha256};
use sos_types::{
    ArtifactManifest, LoaderKind, ProgramArtifact, ANALYSIS_SCHEMA_VERSION, ARTIFACT_SCHEMA_VERSION,
};

#[derive(Debug)]
pub enum ArtifactError {
    Io(std::io::Error),
    InvalidElfRange,
    InvalidElf(String),
    NotSbpf,
    ManifestMismatch(String),
    Serialization(serde_json::Error),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidElfRange => write!(f, "ELF offset and length exceed raw account data"),
            Self::InvalidElf(error) => write!(f, "invalid ELF artifact: {error}"),
            Self::NotSbpf => write!(f, "artifact is ELF but not an SBPF executable"),
            Self::ManifestMismatch(field) => write!(f, "artifact manifest mismatch: {field}"),
            Self::Serialization(error) => write!(f, "manifest serialization error: {error}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

impl From<std::io::Error> for ArtifactError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ArtifactError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn build_manifest(
    artifact: &ProgramArtifact,
    context: &ManifestContext,
) -> Result<ArtifactManifest, ArtifactError> {
    let start = usize::try_from(artifact.elf_offset).map_err(|_| ArtifactError::InvalidElfRange)?;
    let end = start
        .checked_add(artifact.elf_bytes.len())
        .ok_or(ArtifactError::InvalidElfRange)?;
    if end > artifact.raw_account_data.len() {
        return Err(ArtifactError::InvalidElfRange);
    }

    Ok(ArtifactManifest {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        program_address: artifact.program_address.clone(),
        cluster: context.cluster.clone(),
        rpc_endpoint_id: context.rpc_endpoint_id.clone(),
        commitment: context.commitment.clone(),
        context_slot: context.context_slot,
        requested_slot: context.requested_slot,
        loader: artifact.loader,
        programdata_address: artifact.programdata_address.clone(),
        deployment_slot: artifact.deployment_slot,
        upgrade_authority: artifact.upgrade_authority.clone(),
        raw_account_data_length: artifact.raw_account_data.len() as u64,
        elf_offset: artifact.elf_offset,
        elf_length: artifact.elf_bytes.len() as u64,
        raw_account_sha256: sha256_hex(&artifact.raw_account_data),
        elf_sha256: sha256_hex(&artifact.elf_bytes),
        sbpf_version: context.sbpf_version.clone(),
        sos_version: context.sos_version.clone(),
        analysis_schema_version: ANALYSIS_SCHEMA_VERSION,
    })
}

/// Provenance fields used to construct an immutable artifact manifest.
#[derive(Debug, Clone, Default)]
pub struct ManifestContext {
    pub cluster: Option<String>,
    pub rpc_endpoint_id: Option<String>,
    pub commitment: Option<String>,
    pub context_slot: Option<u64>,
    pub requested_slot: Option<u64>,
    pub sbpf_version: Option<String>,
    pub sos_version: String,
}

impl ManifestContext {
    pub fn new(sos_version: impl Into<String>) -> Self {
        Self {
            sos_version: sos_version.into(),
            ..Self::default()
        }
    }
}

pub struct FileArtifactStore {
    root: PathBuf,
}

impl FileArtifactStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn put(
        &self,
        artifact: &ProgramArtifact,
        manifest: &ArtifactManifest,
    ) -> Result<PathBuf, ArtifactError> {
        let facts = sos_core::parse_elf_facts(&artifact.elf_bytes)
            .map_err(|error| ArtifactError::InvalidElf(error.to_string()))?;
        if !facts.is_sbpf() {
            return Err(ArtifactError::NotSbpf);
        }
        if manifest.elf_sha256 != sha256_hex(&artifact.elf_bytes) {
            return Err(ArtifactError::ManifestMismatch("elf_sha256".into()));
        }
        if manifest.raw_account_sha256 != sha256_hex(&artifact.raw_account_data) {
            return Err(ArtifactError::ManifestMismatch("raw_account_sha256".into()));
        }
        if manifest.elf_offset != artifact.elf_offset
            || manifest.elf_length != artifact.elf_bytes.len() as u64
            || manifest.raw_account_data_length != artifact.raw_account_data.len() as u64
        {
            return Err(ArtifactError::ManifestMismatch(
                "artifact byte ranges".into(),
            ));
        }
        let directory = self.root.join("sha256");
        fs::create_dir_all(&directory)?;
        let hash = &manifest.elf_sha256;
        let artifact_path = directory.join(format!("{hash}.so"));
        let manifest_path = directory.join(format!("{hash}.manifest.json"));
        if !artifact_path.exists() {
            fs::write(&artifact_path, &artifact.elf_bytes)?;
        }
        if !manifest_path.exists() {
            let bytes = serde_json::to_vec_pretty(manifest)?;
            fs::write(&manifest_path, bytes)?;
        }
        Ok(artifact_path)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

pub fn local_artifact(
    path: impl AsRef<Path>,
    program_address: impl Into<String>,
) -> Result<(ProgramArtifact, ArtifactManifest), ArtifactError> {
    let elf_bytes = fs::read(path)?;
    let artifact = ProgramArtifact {
        program_address: program_address.into(),
        loader: LoaderKind::LegacyBpf,
        programdata_address: None,
        deployment_slot: None,
        upgrade_authority: None,
        raw_account_data: elf_bytes.clone(),
        elf_bytes,
        elf_offset: 0,
    };
    let manifest = build_manifest(&artifact, &ManifestContext::new(env!("CARGO_PKG_VERSION")))?;
    Ok((artifact, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn hashes_and_stores_an_artifact_by_elf_hash() {
        let bytes = minimal_sbpf_elf();
        let artifact = ProgramArtifact {
            program_address: "example".into(),
            loader: LoaderKind::LegacyBpf,
            programdata_address: None,
            deployment_slot: None,
            upgrade_authority: None,
            raw_account_data: bytes.clone(),
            elf_bytes: bytes,
            elf_offset: 0,
        };
        let manifest =
            build_manifest(&artifact, &ManifestContext::new("test")).expect("valid artifact");
        let root = std::env::temp_dir().join(format!(
            "sos-artifact-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = FileArtifactStore::new(&root)
            .put(&artifact, &manifest)
            .unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains(&manifest.elf_sha256));
    }

    #[test]
    fn refuses_non_elf_artifacts_before_storage() {
        let bytes = vec![1, 2, 3, 4];
        let artifact = ProgramArtifact {
            program_address: "example".into(),
            loader: LoaderKind::LegacyBpf,
            programdata_address: None,
            deployment_slot: None,
            upgrade_authority: None,
            raw_account_data: bytes.clone(),
            elf_bytes: bytes,
            elf_offset: 0,
        };
        let manifest = build_manifest(&artifact, &ManifestContext::new("test"))
            .expect("manifest range is valid");
        let root = std::env::temp_dir().join("sos-artifact-invalid-test");
        let error = FileArtifactStore::new(root)
            .put(&artifact, &manifest)
            .unwrap_err();
        assert!(matches!(error, ArtifactError::InvalidElf(_)));
    }

    #[test]
    fn refuses_a_manifest_for_different_bytes() {
        let bytes = minimal_sbpf_elf();
        let artifact = ProgramArtifact {
            program_address: "example".into(),
            loader: LoaderKind::LegacyBpf,
            programdata_address: None,
            deployment_slot: None,
            upgrade_authority: None,
            raw_account_data: bytes.clone(),
            elf_bytes: bytes,
            elf_offset: 0,
        };
        let mut manifest = build_manifest(&artifact, &ManifestContext::new("test"))
            .expect("manifest range is valid");
        manifest.elf_sha256 = "0".repeat(64);
        let error = FileArtifactStore::new(std::env::temp_dir())
            .put(&artifact, &manifest)
            .unwrap_err();
        assert!(matches!(error, ArtifactError::ManifestMismatch(field) if field == "elf_sha256"));
    }

    fn minimal_sbpf_elf() -> Vec<u8> {
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(&sos_core::ELF_MAGIC);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&sos_core::EM_BPF.to_le_bytes());
        bytes
    }
}
