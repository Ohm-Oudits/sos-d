#![no_main]

use libfuzzer_sys::fuzz_target;
use sos_loader::{
    LegacyBpfAdapter, LoaderAdapter, LoaderV4Adapter, UpgradeableV3Adapter, BPF_LOADER,
    BPF_LOADER_DEPRECATED, BPF_LOADER_UPGRADEABLE, LOADER_V4,
};
use sos_rpc::{ChainProvider, RpcError};
use sos_types::{AccountSnapshot, SignatureInfo, TransactionEvidence};

struct NoopProvider;

impl ChainProvider for NoopProvider {
    fn get_account(&self, address: &str, _slot: Option<u64>) -> Result<AccountSnapshot, RpcError> {
        Err(RpcError::NotFound(address.into()))
    }

    fn get_signatures(
        &self,
        _program: &str,
        _limit: usize,
    ) -> Result<Vec<SignatureInfo>, RpcError> {
        Err(RpcError::NotFound("fuzz history".into()))
    }

    fn get_transaction(
        &self,
        signature: &str,
        _slot: Option<u64>,
    ) -> Result<TransactionEvidence, RpcError> {
        Err(RpcError::NotFound(signature.into()))
    }
}

fuzz_target!(|data: &[u8]| {
    let provider = NoopProvider;
    for owner in [
        BPF_LOADER,
        BPF_LOADER_DEPRECATED,
        BPF_LOADER_UPGRADEABLE,
        LOADER_V4,
        "unknown-owner",
    ] {
        let account = AccountSnapshot {
            address: "11111111111111111111111111111111".into(),
            owner: owner.into(),
            executable: true,
            lamports: 1,
            data: data.to_vec(),
            context_slot: Some(1),
        };
        let _ = LegacyBpfAdapter.resolve_artifact(&account, &provider);
        let _ = UpgradeableV3Adapter.resolve_artifact(&account, &provider);
        let _ = LoaderV4Adapter.resolve_artifact(&account, &provider);
    }
});
