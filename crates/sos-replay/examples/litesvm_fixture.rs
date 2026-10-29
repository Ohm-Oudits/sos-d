//! Execute the checked-in hello-world SBF fixture through the real replay
//! engine. Run after building fixtures:
//!
//! cargo run -p sos-replay --features litesvm --example litesvm_fixture

use std::{fs, path::PathBuf};

use base64::Engine;
use litesvm::LiteSVM;
use solana_address::Address;
use solana_instruction::Instruction;
use solana_keypair::Keypair;
use solana_message::{Message, VersionedMessage};
use solana_signer::Signer;
use solana_transaction::versioned::VersionedTransaction;
use sos_replay::{compare_candidate_observations, LiteSvmReplayEngine, ReplayEngine, TraceStatus};
use sos_types::{
    AccountSnapshot, CandidateObservation, EvidenceRef, LoaderKind, ProgramArtifact,
    TransactionEvidence,
};

fn main() {
    let artifact_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/deploy/sos_hello_world.so"));
    let elf_bytes = fs::read(&artifact_path).unwrap_or_else(|error| {
        panic!("read {}: {error}", artifact_path.display());
    });

    let program_id = Address::new_unique();
    let payer = Keypair::new();
    let mut setup = LiteSVM::new().with_sigverify(false);
    setup
        .airdrop(&payer.pubkey(), 1_000_000_000)
        .expect("airdrop payer");
    let payer_account = setup.get_account(&payer.pubkey()).expect("payer account");
    let transaction = VersionedTransaction::try_new(
        VersionedMessage::Legacy(Message::new_with_blockhash(
            &[Instruction {
                program_id,
                accounts: Vec::new(),
                data: vec![7, 8, 9],
            }],
            Some(&payer.pubkey()),
            &setup.latest_blockhash(),
        )),
        &[&payer],
    )
    .expect("sign fixture transaction");
    let transaction_bytes = bincode::serialize(&transaction).expect("serialize transaction");

    let artifact = ProgramArtifact {
        program_address: program_id.to_string(),
        loader: LoaderKind::LegacyBpf,
        programdata_address: None,
        deployment_slot: None,
        upgrade_authority: None,
        raw_account_data: elf_bytes.clone(),
        elf_bytes,
        elf_offset: 0,
    };
    let payer_snapshot = AccountSnapshot {
        address: payer.pubkey().to_string(),
        owner: payer_account.owner.to_string(),
        executable: payer_account.executable,
        lamports: payer_account.lamports,
        data: payer_account.data,
        context_slot: Some(1),
    };
    let transaction_evidence = TransactionEvidence {
        signature: "sos-litesvm-fixture".into(),
        slot: Some(1),
        raw_json: serde_json::json!({
            "transaction": [
                base64::engine::general_purpose::STANDARD.encode(transaction_bytes),
                "base64"
            ],
            "meta": {"innerInstructions": []}
        }),
    };

    let trace = LiteSvmReplayEngine
        .replay(&artifact, &transaction_evidence, &[payer_snapshot])
        .expect("replay should return a trace");
    assert_eq!(trace.status, TraceStatus::Success, "trace: {trace:?}");
    assert!(trace.logs.iter().any(|log| log.contains("SOS hello world")));
    assert!(trace.compute_units.is_some());
    let comparison = compare_candidate_observations(
        &[CandidateObservation {
            kind: "log_contains".into(),
            value: "SOS hello world".into(),
            evidence: vec![EvidenceRef {
                source: "ir".into(),
                location: "function:fn_0".into(),
                detail: None,
            }],
        }],
        &trace,
    );
    assert_eq!(
        comparison.status,
        sos_types::EvidenceStatus::BehaviorallyValidated
    );
    assert!(comparison.mismatches.is_empty());
    println!("replay_status=success");
    println!("logs={}", trace.logs.len());
    println!("compute_units={:?}", trace.compute_units);
}
