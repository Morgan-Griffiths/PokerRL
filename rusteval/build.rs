// rusteval/build.rs

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let library_name = "librusteval"; // Replace with your library name

    #[cfg(target_os = "macos")]
    let library_extension = ".dylib";
    #[cfg(target_os = "linux")]
    let library_extension = ".so";
    #[cfg(target_os = "windows")]
    let library_extension = ".dll";

    let library_filename = format!("{}{}", library_name, library_extension);
    let library_path = Path::new(&out_dir)
        .join("../../../../../PokerRL")
        .join(&library_filename);

    // Remove existing library if it exists
    if library_path.exists() {
        fs::remove_file(&library_path).unwrap();
    }

    // Copy the built library to the desired location
    fs::copy(
        Path::new(&out_dir)
            .join("../../../release")
            .join(&library_filename),
        &library_path,
    )
    .unwrap();
}
