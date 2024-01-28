use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let library_name = "librusteval"; // Replace with your library name

    #[cfg(target_os = "macos")]
    let library_extension = ".dylib";
    #[cfg(target_os = "linux")]
    let library_extension = ".so";
    #[cfg(target_os = "windows")]
    let library_extension = ".dll";
    let library_filename = format!("{}{}", library_name, library_extension);
    
    let project_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let library_path = Path::new(&project_dir)
        .join(&library_filename);

    let source_path = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("target")
        .join("release")
        .join(&library_filename);

    // Remove existing library if it exists
    if library_path.exists() {
        fs::remove_file(&library_path).expect("Failed to remove existing library");
    }

    // Copy the built library to the desired location if it exists
    if source_path.exists() {
        fs::copy(&source_path, &library_path)
            .expect("Failed to copy library to desired location");
    }
}
