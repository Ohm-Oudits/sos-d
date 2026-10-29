#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = sos_core::parse_elf_facts(data);
    let _ = sos_analysis::analyze_elf(data);
});
