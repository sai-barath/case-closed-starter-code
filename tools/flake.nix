{
  description = "Development environment with opencode";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    opencode-flake.url = "github:aodhanhayter/opencode-flake";
  };

  outputs = { nixpkgs, opencode-flake, ... }: {
    devShells.x86_64-linux.default = nixpkgs.legacyPackages.x86_64-linux.mkShell {
      buildInputs = [
        opencode-flake.packages.x86_64-linux.default
        nixpkgs.legacyPackages.x86_64-linux.go
      ];
    };
  };
}
