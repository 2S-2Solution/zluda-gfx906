fn main() {
    if cfg!(windows) {
        // 2 Solution (gfx906): every HIP call goes to the Marconi shim (amdhip64_7_shim.dll), which
        // provides the HIP 7 ABI of these bindings on top of the HIP 5.7 runtime of the Windows
        // driver. The shim is also installed as amdhip64_7.dll, but that name is avoided on purpose:
        // a real amdhip64_7.dll in System32 (newer drivers) would be found before the PATH.
        println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7_shim.dll");
    } else {
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
}
