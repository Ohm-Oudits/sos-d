//! Loader adapters isolate Solana account-layout changes from analysis code.

use std::fmt;

use solana_loader_v3_interface::state::UpgradeableLoaderState;
use solana_loader_v4_interface::state::LoaderV4State;
use sos_rpc::{ChainProvider, RpcError};
use sos_types::{AccountSnapshot, LoaderKind, ProgramArtifact};

pub const BPF_LOADER_DEPRECATED: &str = "BPFLoader1111111111111111111111111111111111";
pub const BPF_LOADER: &str = "BPFLoader2111111111111111111111111111111111";
pub const BPF_LOADER_UPGRADEABLE: &str = "BPFLoaderUpgradeab1e11111111111111111111111";
pub const LOADER_V4: &str = "LoaderV411111111111111111111111111111111111";

pub trait LoaderAdapter {
    fn kind(&self) -> LoaderKind;
    fn identify(&self, account: &AccountSnapshot) -> bool;
    fn resolve_artifact(
        &self,
        program: &AccountSnapshot,
        provider: &dyn ChainProvider,
    ) -> Result<ProgramArtifact, LoaderError>;
}

#[derive(Debug)]
pub enum LoaderError {
    Rpc(RpcError),
    NotExecutable(String),
    UnsupportedOwner(String),
    InvalidState(String),
    MissingProgramData(String),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rpc(error) => write!(f, "{error}"),
            Self::NotExecutable(address) => write!(f, "account is not executable: {address}"),
            Self::UnsupportedOwner(owner) => write!(f, "unsupported program owner: {owner}"),
            Self::InvalidState(error) => write!(f, "invalid loader state: {error}"),
            Self::MissingProgramData(address) => {
                write!(f, "missing ProgramData account: {address}")
            }
        }
    }
}

impl std::error::Error for LoaderError {}

impl From<RpcError> for LoaderError {
    fn from(error: RpcError) -> Self {
        Self::Rpc(error)
    }
}

pub struct LoaderRegistry {
    adapters: Vec<Box<dyn LoaderAdapter>>,
}

impl LoaderRegistry {
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
        }
    }

    pub fn register(&mut self, adapter: Box<dyn LoaderAdapter>) {
        self.adapters.push(adapter);
    }

    pub fn identify(&self, account: &AccountSnapshot) -> Option<&dyn LoaderAdapter> {
        self.adapters
            .iter()
            .find(|adapter| adapter.identify(account))
            .map(|adapter| adapter.as_ref())
    }
}

impl Default for LoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn default_registry() -> LoaderRegistry {
    let mut registry = LoaderRegistry::new();
    registry.register(Box::new(UpgradeableV3Adapter));
    registry.register(Box::new(LegacyBpfAdapter));
    registry.register(Box::new(LoaderV4Adapter));
    registry
}

pub struct UpgradeableV3Adapter;

impl LoaderAdapter for UpgradeableV3Adapter {
    fn kind(&self) -> LoaderKind {
        LoaderKind::UpgradeableV3
    }

    fn identify(&self, account: &AccountSnapshot) -> bool {
        account.owner == BPF_LOADER_UPGRADEABLE
    }

    fn resolve_artifact(
        &self,
        program: &AccountSnapshot,
        provider: &dyn ChainProvider,
    ) -> Result<ProgramArtifact, LoaderError> {
        ensure_program_account(program)?;
        let state: UpgradeableLoaderState = bincode::deserialize(&program.data)
            .map_err(|error| LoaderError::InvalidState(error.to_string()))?;
        let UpgradeableLoaderState::Program {
            programdata_address,
        } = state
        else {
            return Err(LoaderError::InvalidState(
                "program account is not in Program state".into(),
            ));
        };
        let programdata_address = programdata_address.to_string();
        let programdata = provider.get_account(&programdata_address, program.context_slot)?;
        if programdata.data.len() < UpgradeableLoaderState::size_of_programdata_metadata() {
            return Err(LoaderError::InvalidState(
                "ProgramData metadata is truncated".into(),
            ));
        }
        let programdata_state: UpgradeableLoaderState = bincode::deserialize(
            &programdata.data[..UpgradeableLoaderState::size_of_programdata_metadata()],
        )
        .map_err(|error| LoaderError::InvalidState(error.to_string()))?;
        let UpgradeableLoaderState::ProgramData {
            slot,
            upgrade_authority_address,
        } = programdata_state
        else {
            return Err(LoaderError::InvalidState(
                "ProgramData account is not in ProgramData state".into(),
            ));
        };
        let offset = UpgradeableLoaderState::size_of_programdata_metadata();
        let elf_bytes = programdata.data[offset..].to_vec();
        if elf_bytes.len() < 4 || elf_bytes[..4] != [0x7f, b'E', b'L', b'F'] {
            return Err(LoaderError::InvalidState(
                "ProgramData does not contain an ELF payload".into(),
            ));
        }
        Ok(ProgramArtifact {
            program_address: program.address.clone(),
            loader: LoaderKind::UpgradeableV3,
            programdata_address: Some(programdata_address),
            deployment_slot: Some(slot),
            upgrade_authority: upgrade_authority_address.map(|address| address.to_string()),
            raw_account_data: programdata.data,
            elf_bytes,
            elf_offset: offset as u64,
        })
    }
}

pub struct LegacyBpfAdapter;

impl LoaderAdapter for LegacyBpfAdapter {
    fn kind(&self) -> LoaderKind {
        LoaderKind::LegacyBpf
    }

    fn identify(&self, account: &AccountSnapshot) -> bool {
        account.owner == BPF_LOADER || account.owner == BPF_LOADER_DEPRECATED
    }

    fn resolve_artifact(
        &self,
        program: &AccountSnapshot,
        _provider: &dyn ChainProvider,
    ) -> Result<ProgramArtifact, LoaderError> {
        ensure_program_account(program)?;
        ensure_elf(&program.data)?;
        Ok(ProgramArtifact {
            program_address: program.address.clone(),
            loader: LoaderKind::LegacyBpf,
            programdata_address: None,
            deployment_slot: program.context_slot,
            upgrade_authority: None,
            raw_account_data: program.data.clone(),
            elf_bytes: program.data.clone(),
            elf_offset: 0,
        })
    }
}

pub struct LoaderV4Adapter;

impl LoaderAdapter for LoaderV4Adapter {
    fn kind(&self) -> LoaderKind {
        LoaderKind::LoaderV4
    }

    fn identify(&self, account: &AccountSnapshot) -> bool {
        account.owner == LOADER_V4
    }

    fn resolve_artifact(
        &self,
        program: &AccountSnapshot,
        _provider: &dyn ChainProvider,
    ) -> Result<ProgramArtifact, LoaderError> {
        ensure_program_account(program)?;
        let offset = LoaderV4State::program_data_offset();
        if program.data.len() <= offset {
            return Err(LoaderError::InvalidState(
                "loader-v4 account has no ELF payload".into(),
            ));
        }
        let elf_bytes = program.data[offset..].to_vec();
        ensure_elf(&elf_bytes)?;
        Ok(ProgramArtifact {
            program_address: program.address.clone(),
            loader: LoaderKind::LoaderV4,
            programdata_address: None,
            deployment_slot: program.context_slot,
            upgrade_authority: None,
            raw_account_data: program.data.clone(),
            elf_bytes,
            elf_offset: offset as u64,
        })
    }
}

fn ensure_elf(bytes: &[u8]) -> Result<(), LoaderError> {
    if bytes.len() < 4 || bytes[..4] != [0x7f, b'E', b'L', b'F'] {
        return Err(LoaderError::InvalidState(
            "account does not contain an ELF payload".into(),
        ));
    }
    Ok(())
}

fn ensure_program_account(account: &AccountSnapshot) -> Result<(), LoaderError> {
    if !account.executable {
        return Err(LoaderError::NotExecutable(account.address.clone()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_executable_program_accounts() {
        let account = AccountSnapshot {
            address: "program".into(),
            owner: BPF_LOADER.to_owned(),
            executable: false,
            lamports: 1,
            data: vec![0x7f, b'E', b'L', b'F'],
            context_slot: None,
        };
        let error = LegacyBpfAdapter
            .resolve_artifact(&account, &NoopProvider)
            .unwrap_err();
        assert!(matches!(error, LoaderError::NotExecutable(address) if address == "program"));
    }

    struct NoopProvider;

    impl ChainProvider for NoopProvider {
        fn get_account(
            &self,
            _address: &str,
            _slot: Option<u64>,
        ) -> Result<AccountSnapshot, RpcError> {
            unreachable!("test does not request account data")
        }

        fn get_signatures(
            &self,
            _program: &str,
            _limit: usize,
        ) -> Result<Vec<sos_types::SignatureInfo>, RpcError> {
            unreachable!("test does not request signatures")
        }

        fn get_transaction(
            &self,
            _signature: &str,
            _slot: Option<u64>,
        ) -> Result<sos_types::TransactionEvidence, RpcError> {
            unreachable!("test does not request transactions")
        }
    }
}
