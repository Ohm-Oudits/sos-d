//! Versioned data contracts shared by the SOS pipeline.

use serde::{Deserialize, Serialize};

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const ANALYSIS_SCHEMA_VERSION: u32 = 1;
pub const SEMANTIC_SCHEMA_VERSION: u32 = 1;
pub const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LoaderKind {
    LegacyBpf,
    UpgradeableV3,
    LoaderV4,
    Native,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub address: String,
    pub owner: String,
    pub executable: bool,
    pub lamports: u64,
    pub data: Vec<u8>,
    pub context_slot: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignatureInfo {
    pub signature: String,
    pub slot: Option<u64>,
    pub err: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionEvidence {
    pub signature: String,
    pub slot: Option<u64>,
    pub raw_json: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramArtifact {
    pub program_address: String,
    pub loader: LoaderKind,
    pub programdata_address: Option<String>,
    pub deployment_slot: Option<u64>,
    pub upgrade_authority: Option<String>,
    pub raw_account_data: Vec<u8>,
    pub elf_bytes: Vec<u8>,
    pub elf_offset: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub program_address: String,
    pub cluster: Option<String>,
    pub rpc_endpoint_id: Option<String>,
    pub commitment: Option<String>,
    pub context_slot: Option<u64>,
    pub requested_slot: Option<u64>,
    pub loader: LoaderKind,
    pub programdata_address: Option<String>,
    pub deployment_slot: Option<u64>,
    pub upgrade_authority: Option<String>,
    pub raw_account_data_length: u64,
    pub elf_offset: u64,
    pub elf_length: u64,
    pub raw_account_sha256: String,
    pub elf_sha256: String,
    pub sbpf_version: Option<String>,
    pub sos_version: String,
    pub analysis_schema_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Fact,
    Recovered,
    Hypothesis,
    NotAvailable,
    Unverified,
    BehaviorallyValidated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub source: String,
    pub location: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub kind: String,
    pub value: serde_json::Value,
    pub confidence: Option<f32>,
    pub status: EvidenceStatus,
    pub evidence: Vec<EvidenceRef>,
}

/// A bounded, evidence-linked behavior claim emitted by an AI refinement
/// candidate. These observations are compared with a supplied replay trace;
/// they are not treated as facts until an external validator matches them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateObservation {
    pub kind: String,
    pub value: String,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serializes_with_a_stable_schema_version() {
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            program_address: "example".into(),
            cluster: None,
            rpc_endpoint_id: None,
            commitment: None,
            context_slot: None,
            requested_slot: None,
            loader: LoaderKind::Unknown,
            programdata_address: None,
            deployment_slot: None,
            upgrade_authority: None,
            raw_account_data_length: 0,
            elf_offset: 0,
            elf_length: 0,
            raw_account_sha256: String::new(),
            elf_sha256: String::new(),
            sbpf_version: None,
            sos_version: "test".into(),
            analysis_schema_version: ANALYSIS_SCHEMA_VERSION,
        };

        let json = serde_json::to_string(&manifest).expect("manifest is serializable");
        assert!(json.contains("\"schema_version\":1"));
    }

    #[test]
    fn checked_in_contract_schemas_are_parseable_and_versioned() {
        let schemas = [
            (
                include_str!("../../../schemas/v1/artifact-manifest.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v1/static-analysis.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v1/canonical-ir.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v1/evidence-pack.schema.json"),
                1,
            ),
            (include_str!("../../../schemas/v1/ai-bundle.schema.json"), 1),
            (
                include_str!("../../../schemas/v1/refinement.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v1/validation.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v2/execution-trace.schema.json"),
                2,
            ),
            (include_str!("../../../schemas/v1/report.schema.json"), 1),
            (
                include_str!("../../../schemas/v1/semantic-analysis.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v1/security-report.schema.json"),
                1,
            ),
            (
                include_str!("../../../schemas/v1/binary-judge.schema.json"),
                1,
            ),
        ];

        for (source, expected_version) in schemas {
            let value: serde_json::Value =
                serde_json::from_str(source).expect("checked-in schema is valid JSON");
            assert_eq!(
                value["$schema"],
                "https://json-schema.org/draft/2020-12/schema"
            );
            assert_eq!(value["type"], "object");
            assert_eq!(
                value["properties"]["schema_version"]["const"],
                expected_version
            );
            assert!(value["required"]
                .as_array()
                .is_some_and(|required| required.iter().any(|field| field == "schema_version")));
        }
    }

    #[test]
    fn hello_fixture_manifest_matches_the_checked_in_golden_contract() {
        let source = include_str!("../../../fixtures/golden/hello-manifest.json");
        let manifest: ArtifactManifest =
            serde_json::from_str(source).expect("golden manifest is a valid contract");
        assert_eq!(manifest.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(manifest.analysis_schema_version, ANALYSIS_SCHEMA_VERSION);
        assert_eq!(manifest.elf_sha256, manifest.raw_account_sha256);
        assert_eq!(manifest.elf_length, manifest.raw_account_data_length);
    }
}
