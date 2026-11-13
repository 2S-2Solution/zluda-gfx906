#[cfg(windows)]
fn main() {
    extern crate embed_resource;
    embed_resource::compile("../zluda_windows/manifest.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
    // 2 Solution (gfx906): our code also calls hipblaslt.dll directly (raw-dylib), and some of those
    // functions do not exist in the gfx906 build (e.g. the _64 variants of rocBLAS 5.7): the DLL
    // stays delay-loaded as before the merge, so a missing function fails only when it is called.
    println!("cargo:rustc-link-arg=delayimp.lib");
    println!("cargo:rustc-link-arg=/DELAYLOAD:hipblaslt.dll");
    println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7_shim.dll");
}

#[cfg(not(windows))]
fn main() {}
