//! Durable metadata for immutable SOS artifacts and derived analysis bundles.
//!
//! The filesystem remains the source of truth for bytes and large JSON outputs.
//! SQLite stores only indexed provenance and lifecycle metadata so the storage
//! layer can later be replaced by an API or object-store implementation.

use std::{fmt, path::Path};

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use sos_types::ArtifactManifest;

pub const STORAGE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub enum StorageError {
    Sqlite(rusqlite::Error),
    Serialization(serde_json::Error),
    InvalidManifest(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(error) => write!(f, "SQLite error: {error}"),
            Self::Serialization(error) => write!(f, "metadata serialization error: {error}"),
            Self::InvalidManifest(message) => write!(f, "invalid manifest: {message}"),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<rusqlite::Error> for StorageError {
    fn from(error: rusqlite::Error) -> Self {
        Self::Sqlite(error)
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactRecord {
    pub artifact_id: String,
    pub elf_sha256: String,
    pub program_address: String,
    pub artifact_path: String,
    pub manifest: ArtifactManifest,
}

pub struct MetadataStore {
    connection: Connection,
}

impl MetadataStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let connection = Connection::open(path)?;
        let store = Self { connection };
        store.migrate()?;
        Ok(store)
    }

    pub fn in_memory() -> Result<Self, StorageError> {
        let connection = Connection::open_in_memory()?;
        let store = Self { connection };
        store.migrate()?;
        Ok(store)
    }

    pub fn migrate(&self) -> Result<(), StorageError> {
        self.connection.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS sos_schema (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 version INTEGER NOT NULL
             );
             INSERT INTO sos_schema (id, version) VALUES (1, 1)
                 ON CONFLICT(id) DO UPDATE SET version = excluded.version;
             CREATE TABLE IF NOT EXISTS artifacts (
                 artifact_id TEXT PRIMARY KEY,
                 elf_sha256 TEXT NOT NULL,
                 program_address TEXT NOT NULL,
                 artifact_path TEXT NOT NULL,
                 manifest_json TEXT NOT NULL,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                 UNIQUE (elf_sha256, manifest_json)
             );
             CREATE INDEX IF NOT EXISTS artifacts_program_idx
                 ON artifacts(program_address, created_at DESC);
             CREATE TABLE IF NOT EXISTS bundle_runs (
                 run_id TEXT PRIMARY KEY,
                 artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                 kind TEXT NOT NULL,
                 output_path TEXT NOT NULL,
                 schema_version INTEGER NOT NULL,
                 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
             );
             CREATE INDEX IF NOT EXISTS bundle_runs_artifact_idx
                 ON bundle_runs(artifact_id, kind, created_at DESC);",
        )?;
        Ok(())
    }

    pub fn register_artifact(
        &self,
        manifest: &ArtifactManifest,
        artifact_path: impl AsRef<Path>,
    ) -> Result<ArtifactRecord, StorageError> {
        if manifest.elf_sha256.len() != 64
            || !manifest
                .elf_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(StorageError::InvalidManifest(
                "elf_sha256 must be a 64-character hexadecimal digest".into(),
            ));
        }

        let manifest_json = serde_json::to_string(manifest)?;
        let artifact_id = digest(manifest_json.as_bytes());
        let artifact_path = artifact_path.as_ref().to_string_lossy().into_owned();
        self.connection.execute(
            "INSERT INTO artifacts
                (artifact_id, elf_sha256, program_address, artifact_path, manifest_json)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(artifact_id) DO UPDATE SET
                artifact_path = excluded.artifact_path,
                manifest_json = excluded.manifest_json",
            params![
                artifact_id,
                manifest.elf_sha256,
                manifest.program_address,
                artifact_path,
                manifest_json,
            ],
        )?;

        Ok(ArtifactRecord {
            artifact_id,
            elf_sha256: manifest.elf_sha256.clone(),
            program_address: manifest.program_address.clone(),
            artifact_path,
            manifest: manifest.clone(),
        })
    }

    pub fn find_artifact(&self, artifact_id: &str) -> Result<Option<ArtifactRecord>, StorageError> {
        self.connection
            .query_row(
                "SELECT artifact_id, elf_sha256, program_address, artifact_path, manifest_json
                 FROM artifacts WHERE artifact_id = ?1",
                params![artifact_id],
                |row| {
                    let manifest_json: String = row.get(4)?;
                    let manifest = serde_json::from_str::<ArtifactManifest>(&manifest_json)
                        .map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                4,
                                rusqlite::types::Type::Text,
                                Box::new(error),
                            )
                        })?;
                    Ok(ArtifactRecord {
                        artifact_id: row.get(0)?,
                        elf_sha256: row.get(1)?,
                        program_address: row.get(2)?,
                        artifact_path: row.get(3)?,
                        manifest,
                    })
                },
            )
            .optional()
            .map_err(StorageError::from)
    }

    pub fn record_bundle(
        &self,
        run_id: &str,
        artifact_id: &str,
        kind: &str,
        output_path: impl AsRef<Path>,
        schema_version: u32,
    ) -> Result<(), StorageError> {
        self.connection.execute(
            "INSERT INTO bundle_runs
                (run_id, artifact_id, kind, output_path, schema_version)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(run_id) DO UPDATE SET
                artifact_id = excluded.artifact_id,
                kind = excluded.kind,
                output_path = excluded.output_path,
                schema_version = excluded.schema_version",
            params![
                run_id,
                artifact_id,
                kind,
                output_path.as_ref().to_string_lossy(),
                schema_version,
            ],
        )?;
        Ok(())
    }

    pub fn schema_version(&self) -> Result<u32, StorageError> {
        self.connection
            .query_row("SELECT version FROM sos_schema WHERE id = 1", [], |row| {
                row.get::<_, u32>(0)
            })
            .map_err(StorageError::from)
    }
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sos_types::{
        ArtifactManifest, LoaderKind, ANALYSIS_SCHEMA_VERSION, ARTIFACT_SCHEMA_VERSION,
    };

    fn manifest() -> ArtifactManifest {
        ArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            program_address: "Program111111111111111111111111111111111111".into(),
            cluster: Some("localnet".into()),
            rpc_endpoint_id: Some("local".into()),
            commitment: Some("finalized".into()),
            context_slot: Some(42),
            requested_slot: None,
            loader: LoaderKind::LegacyBpf,
            programdata_address: None,
            deployment_slot: None,
            upgrade_authority: None,
            raw_account_data_length: 4,
            elf_offset: 0,
            elf_length: 4,
            raw_account_sha256: "a".repeat(64),
            elf_sha256: "b".repeat(64),
            sbpf_version: Some("V0".into()),
            sos_version: "test".into(),
            analysis_schema_version: ANALYSIS_SCHEMA_VERSION,
        }
    }

    #[test]
    fn migration_is_idempotent_and_registers_provenance() {
        let store = MetadataStore::in_memory().unwrap();
        store.migrate().unwrap();
        assert_eq!(store.schema_version().unwrap(), STORAGE_SCHEMA_VERSION);

        let manifest = manifest();
        let record = store
            .register_artifact(&manifest, "artifacts/sha256/b.so")
            .unwrap();
        let loaded = store.find_artifact(&record.artifact_id).unwrap().unwrap();
        assert_eq!(loaded.manifest, manifest);
        assert_eq!(loaded.artifact_path, "artifacts/sha256/b.so");
    }

    #[test]
    fn artifact_identity_includes_provenance() {
        let store = MetadataStore::in_memory().unwrap();
        let mut first = manifest();
        let first_record = store.register_artifact(&first, "a.so").unwrap();
        first.context_slot = Some(43);
        let second_record = store.register_artifact(&first, "b.so").unwrap();
        assert_ne!(first_record.artifact_id, second_record.artifact_id);
        assert_eq!(first_record.elf_sha256, second_record.elf_sha256);
    }
}
