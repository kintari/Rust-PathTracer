{ pkgs }: {
	deps = [
		pkgs.rustfmt
		pkgs.rustup
		pkgs.cargo
		pkgs.cargo-edit
    pkgs.rust-analyzer
	];
}