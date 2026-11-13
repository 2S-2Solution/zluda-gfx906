#[cfg(windows)]
fn main() {
    extern crate embed_resource;
    embed_resource::compile("../zluda_windows/manifest.rc", embed_resource::NONE)
        .manifest_optional()
        .unwrap();
    println!("cargo:rustc-link-arg=delayimp.lib");
    // 2 Solution (gfx906): HIP comes from the Marconi shim, MIOpen is imported directly
    println!("cargo:rustc-link-arg=/DELAYLOAD:amdhip64_7_shim.dll");
    println!("cargo:rustc-link-arg=/DELAYLOAD:MIOpen.dll");
}

#[cfg(not(windows))]
fn main() {}
