//! Native Rust ABI fixture with explicit dispatch, account checks, and raw data access.

#![allow(unexpected_cfgs)]

use solana_program::{
    account_info::AccountInfo, entrypoint, entrypoint::ProgramResult, msg,
    program_error::ProgramError, pubkey::Pubkey,
};

entrypoint!(process_instruction);

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo<'_>],
    instruction_data: &[u8],
) -> ProgramResult {
    if instruction_data.len() < 4 {
        return Err(ProgramError::InvalidInstructionData);
    }

    let tag = u32::from_le_bytes([
        instruction_data[0],
        instruction_data[1],
        instruction_data[2],
        instruction_data[3],
    ]);
    if tag == 0x5357_5350 {
        msg!("native_dispatch:swap");
    } else {
        msg!("native_dispatch:unknown");
    }

    for account in accounts {
        if !account.is_signer {
            return Err(ProgramError::MissingRequiredSignature);
        }
        if account.owner != program_id {
            return Err(ProgramError::IncorrectProgramId);
        }
    }
    Ok(())
}
