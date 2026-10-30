//! Native fixture that forces a real SBPF indirect function call.

#![allow(unexpected_cfgs)]

use solana_program::{
    account_info::AccountInfo, entrypoint, entrypoint::ProgramResult, program_error::ProgramError,
    pubkey::Pubkey,
};

entrypoint!(process_instruction);

#[inline(never)]
fn add_one(value: u64) -> u64 {
    value.wrapping_add(1)
}

#[inline(never)]
fn subtract_one(value: u64) -> u64 {
    value.wrapping_sub(1)
}

#[inline(never)]
fn invoke_selected(operation: fn(u64) -> u64, value: u64) -> u64 {
    operation(value)
}

pub fn process_instruction(
    _program_id: &Pubkey,
    _accounts: &[AccountInfo<'_>],
    instruction_data: &[u8],
) -> ProgramResult {
    let (&selector, rest) = instruction_data
        .split_first()
        .ok_or(ProgramError::InvalidInstructionData)?;
    let operation: fn(u64) -> u64 = if selector & 1 == 0 {
        add_one
    } else {
        subtract_one
    };
    let value = rest.first().copied().unwrap_or_default() as u64;
    Err(ProgramError::Custom(
        invoke_selected(operation, value) as u32
    ))
}
