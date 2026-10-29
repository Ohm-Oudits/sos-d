#include <solana_sdk.h>

/* Small C/SBF corpus member with serialized input, account-role checks, and
 * observable branches for decompiler evaluation. */
extern uint64_t entrypoint(const uint8_t *input) {
  SolAccountInfo accounts[2] = {0};
  SolParameters params = {.ka = accounts};

  if (!sol_deserialize(input, &params, SOL_ARRAY_SIZE(accounts))) {
    return ERROR_INVALID_ARGUMENT;
  }

  sol_log("SOS C fixture");
  if (params.data_len > 0 && params.data[0] == 0x43) {
    sol_log("c:ping");
  } else {
    sol_log("c:other");
  }

  if (params.ka_num > 0 && !params.ka[0].is_signer) {
    return ERROR_MISSING_REQUIRED_SIGNATURES;
  }
  if (params.ka_num > 1 && !params.ka[1].is_writable) {
    return ERROR_INVALID_ARGUMENT;
  }
  return SUCCESS;
}
