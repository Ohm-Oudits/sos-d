//! Minimal native Solana program used as a deterministic SOS Decompiler fixture.
//!
//! The program intentionally stays small and explicit so the generated SBF ELF is
//! easy to inspect, disassemble, and compare with this source.

// The current Solana entrypoint macro emits a target-specific `cfg(solana)` check
// that stable Rust reports as unexpected on the host build. The warning originates
// in the dependency macro, not in this fixture's source.
#![allow(unexpected_cfgs)]

use solana_program::{
    account_info::AccountInfo, entrypoint, entrypoint::ProgramResult, msg, pubkey::Pubkey,
};

entrypoint!(process_instruction);

/// Logs a deterministic greeting and echoes the instruction payload.
pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo<'_>],
    instruction_data: &[u8],
) -> ProgramResult {
    msg!("SOS hello world");
    msg!("instruction_data_len={}", instruction_data.len());

    if !instruction_data.is_empty() {
        msg!("first_byte={}", instruction_data[0]);
    }

    Ok(())
}
