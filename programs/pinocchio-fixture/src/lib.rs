//! Pinocchio fixture for framework-diverse decompiler evaluation.

#![no_std]
#![allow(unexpected_cfgs)]

extern crate alloc;

#[allow(unused_imports)]
use alloc::format;
use pinocchio::{
    account_info::AccountInfo, entrypoint, msg, program_error::ProgramError, pubkey::Pubkey,
    ProgramResult,
};

entrypoint!(process_instruction);

#[inline(never)]
pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    instruction_data: &[u8],
) -> ProgramResult {
    if instruction_data.len() < 4 {
        msg!("pinocchio:invalid_instruction");
        return Err(ProgramError::InvalidInstructionData);
    }

    let tag = u32::from_le_bytes([
        instruction_data[0],
        instruction_data[1],
        instruction_data[2],
        instruction_data[3],
    ]);
    if tag == 0x5049_4e47 {
        msg!("pinocchio:ping");
    } else {
        msg!("pinocchio:unknown");
    }

    for account in accounts {
        if !account.is_signer() {
            return Err(ProgramError::MissingRequiredSignature);
        }
        if !account.is_owned_by(program_id) {
            return Err(ProgramError::IncorrectProgramId);
        }
    }
    Ok(())
}
