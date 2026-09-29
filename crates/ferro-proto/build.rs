fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../proto/ferrogrid.proto");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // The portable build's bullseye protoc (3.12) rejects proto3 `optional`
        // without this; 3.15+ enables it by default and ignores the flag.
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["../../proto/ferrogrid.proto"], &["../../proto"])?;
    Ok(())
}
