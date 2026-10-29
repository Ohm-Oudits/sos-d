// SOS deterministic syntax normalization; not original source.
// Types and pointer helpers are unverified readability aids; inspect source-line evidence.
pub unsafe fn process_account(ctx: i64, instruction_data: *mut u8, instruction_len: u64) -> u64 // source line 1
{ // source line 2
  let mut account: i64; // source line 3
  let mut amount: u64; // source line 4

  if instruction_len < 9 { // source line 6
    return 0x100000002; // source line 7
  } // source line 8
  account = read_i64(ctx + 0x20); // source line 9
  if read_u8(account + 0x08) == 0 { // source line 10
    return 5; // source line 11
  } // source line 12
  if read_u8(account + 0x09) == 0 { // source line 13
    return 7; // source line 14
  } // source line 15
  if read_value(instruction_data) != 0x2a { // source line 16
    return 0x100000003; // source line 17
  } // source line 18
  amount = read_u64(instruction_data + 1); // source line 19
  write_u64(account + 0x50, read_u64(account + 0x50) + amount); // source line 20
  sol_log("account updated", 15); // source line 21
  return 0; // source line 22
} // source line 23
