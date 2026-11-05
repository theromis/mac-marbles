# Glass Marbles for Windows

Rust and Vulkan port of the Mac demo in the parent folder: path traced glass marbles on a wooden
track, with a newspaper in the dish that shows the refraction.

**Status: this version has never been run on a GPU.** It was written on a Mac, which cannot run
Vulkan ray queries. Type checking against the Windows target, SPIR-V validation and the unit tests
pass, but expect the first run to find problems in the renderer, and the look may differ from the Mac version.

## Requirements

- Windows 10 or 11, with a GPU and a current driver that support Vulkan ray queries
  (`VK_KHR_ray_query`). Examples are NVIDIA RTX, AMD RX 6000 or newer, and Intel Arc.
- Rust, from <https://rustup.rs>. Building needs neither the Vulkan SDK nor a shader compiler.
- Optional: the Vulkan SDK, for `--validate`.

## Build and run

    cd rust
    cargo run --release

Run it from a terminal, since errors are printed there. If no suitable GPU is found, the message
lists each GPU and what it is missing.

Keys: `[` `]` marble count, `1`-`5` quality, `A` automatic quality, `D` denoiser, `C` newspaper close-up,
`Space` camera, `F` fullscreen, `Esc` quit. The title bar shows FPS. All options are in `--help`.

## First run checklist

Run these in order, and keep the output of the first failing one:

    cargo run --release -- --validate --png first.png
    cargo run --release -- --benchmark 30
    cargo run --release -- --validate --log --quit-after 30

## How it differs from the Mac version

| | Mac | Windows |
|---|---|---|
| Ray tracing | Metal, intersection functions | Vulkan ray queries; spheres are confirmed in a loop in the shader |
| Denoiser | MetalFX | Own: temporal accumulation, then an edge-aware a-trous filter |
| Upscaling | MetalFX | Plain stretching of the final image |
| Antialiasing | MetalFX | Own temporal antialiasing |
| Newspaper | Drawn at startup with Core Text | `assets/newspaper.png`, exported from the Mac version |
| Statistics | On-screen text | Title bar and `--log` |
| Heat state | Logged | Not available |

The glass approach is the same: no randomness in the glass paths, and those pixels skip the
denoiser. Physics and geometry are direct ports; the scene has the same 44272 triangles.

## Changing shaders

The compiled shaders in `shaders/spv/` are committed. After editing a shader, run this
(it needs `glslangValidator` and `spirv-val`, both in the Vulkan SDK):

    sh shaders/compile.sh

## Layout

- `src/physics.rs`, `src/scene.rs`: simulation and geometry, tested with `cargo test` on any OS
- `src/renderer.rs`: Vulkan setup, acceleration structures and the frame's passes
- `src/main.rs`: window, automatic quality and benchmark modes
- `shaders/trace.comp`: the path tracer
- `shaders/temporal.comp`, `atrous.comp`, `resolve.comp`, `tonemap.comp`: denoising and final image
