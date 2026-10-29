//! Small Anchor fixture with a discriminator, account initialization, and an
//! explicit authorization check for semantic-recovery benchmarks.

#![allow(deprecated, unexpected_cfgs)]

use anchor_lang::prelude::*;

declare_id!("4wBqpZM9xaSheZzJSMawUKKwhdpChKbZ5eu5ky4Vigw");

#[program]
pub mod sos_anchor_fixture {
    use super::*;

    pub fn initialize(ctx: Context<Initialize>, value: u64) -> Result<()> {
        require!(value > 0, FixtureError::InvalidValue);
        ctx.accounts.state.authority = ctx.accounts.payer.key();
        ctx.accounts.state.value = value;
        msg!("anchor:initialize");
        Ok(())
    }

    pub fn set_value(ctx: Context<SetValue>, value: u64) -> Result<()> {
        ctx.accounts.state.value = value;
        msg!("anchor:set_value");
        Ok(())
    }
}

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(init, payer = payer, space = 8 + 32 + 8)]
    pub state: Account<'info, State>,
    #[account(mut)]
    pub payer: Signer<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SetValue<'info> {
    #[account(mut, has_one = authority)]
    pub state: Account<'info, State>,
    pub authority: Signer<'info>,
}

#[account]
pub struct State {
    pub authority: Pubkey,
    pub value: u64,
}

#[error_code]
pub enum FixtureError {
    #[msg("value must be non-zero")]
    InvalidValue,
    #[msg("authority must sign")]
    MissingAuthority,
}
