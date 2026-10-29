#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(response) = serde_json::from_slice::<sos_ai::ModelResponse>(data) {
        let _ = sos_ai::validate_response(&response);
    }
});
