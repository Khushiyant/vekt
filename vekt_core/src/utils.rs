use std::path::{Path, PathBuf};
use std::fs::{self};
use std::io;

/// Ensures .vekt directory exists with proper .gitignore file
pub fn ensure_vekt_dir(vekt_path: &Path) -> io::Result<()> {
    if !vekt_path.exists() {
        fs::create_dir_all(vekt_path)?;
        // Create .gitignore to ignore everything in .vekt
        let gitignore_path = vekt_path.join(".gitignore");
        if !gitignore_path.exists() {
            fs::write(gitignore_path, "*\n")?;
        }
    }
    Ok(())
}

/// Locates the root .vekt directory by traversing up from the current directory.
/// Returns the path containing .vekt (e.g., /path/to/repo).
pub fn find_vekt_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        let vekt_path = current.join(".vekt");
        if vekt_path.exists() && vekt_path.is_dir() {
            return Some(current);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

/// Returns the path to the blobs directory.
/// Uses the local repository's .vekt/blobs if found, otherwise defaults to ./.vekt/blobs
/// Also ensures .vekt has a .gitignore file
pub fn get_store_path() -> PathBuf {
    let vekt_dir = match find_vekt_root() {
        Some(root) => root.join(".vekt"),
        None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(".vekt"),
    };
    
    // Ensure .vekt has .gitignore (ignore errors as this is best-effort)
    let _ = ensure_vekt_dir(&vekt_dir);
    
    vekt_dir.join("blobs")
}

pub fn get_dtype_size(dtype: &str) -> usize {
    match dtype {
        "F32" => 4,
        "F16" => 2,
        "BF16" => 2,
        "I64" => 8,
        "I32" => 4,
        "I16" => 2,
        "I8" => 1,
        "U8" => 1,
        "BOOL" => 1,
        _ => 1, // Fallback
    }
}

pub struct LockFile {
    path: PathBuf,
}

impl LockFile {
    pub fn lock() -> Result<Self, io::Error> {
        // Use the found root or current dir for locking
        let root = find_vekt_root().unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        let vekt_dir = root.join(".vekt");
        let path = vekt_dir.join("lock");
        
        // Ensure .vekt exists with .gitignore
        ensure_vekt_dir(&vekt_dir)?;

        // Try to create the file atomically. fails if exists.
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|_| io::Error::new(io::ErrorKind::AlreadyExists, "vekt is currently locked by another process."))?;

        Ok(LockFile { path })
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}
