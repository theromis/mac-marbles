//! Glass marbles: window, quality control and benchmark modes.

mod renderer;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use renderer::{Renderer, RendererConfig, Result, WindowHandles, TIERS};
use std::fmt::Write as _;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

const HELP: &str = "\
Glass Marbles: real-time path traced glass, using Vulkan ray queries.

Without options a window opens, with automatic quality. Keys:
  [ ]    fewer or more marbles       1-5    quality (turns automatic off)
  A      automatic quality           D      denoiser
  C      newspaper close-up          Space  pause camera
  F      fullscreen                  Esc    quit

Options:
  --marbles N       number of marbles (default 150)
  --tier NAME       cinematic, high, medium, balanced or low (default medium)
  --size WxH        window or benchmark size (default 1600x900)
  --no-denoise      show the raw traced image
  --no-adaptive     keep the chosen quality
  --no-vsync        do not wait for the display in the window
  --closeup         start on the newspaper
  --pause-camera    keep the camera still
  --benchmark S     render S seconds with no window, at full speed, and report
  --warmup S        seconds to skip before measuring (default 3)
  --csv FILE        also write benchmark results to a file
  --png FILE        render frames with no window and save the last one
  --png-after N     frames to render before saving (default 240)
  --log             print window statistics once per second
  --quit-after S    close the window after S seconds
  --validate        enable Vulkan validation (needs the Vulkan SDK)
";

struct Options {
    marbles: usize,
    tier_index: usize,
    width: u32,
    height: u32,
    denoise: bool,
    adaptive: bool,
    vsync: bool,
    close_up: bool,
    paused_camera: bool,
    benchmark_seconds: Option<f64>,
    warmup_seconds: f64,
    csv_path: Option<String>,
    png_path: Option<String>,
    png_after: u32,
    log: bool,
    quit_after: Option<f64>,
    validate: bool,
}

impl Options {
    fn parse() -> std::result::Result<Self, String> {
        let mut o = Self {
            marbles: 150,
            tier_index: 2,
            width: 1600,
            height: 900,
            denoise: true,
            adaptive: true,
            vsync: true,
            close_up: false,
            paused_camera: false,
            benchmark_seconds: None,
            warmup_seconds: 3.0,
            csv_path: None,
            png_path: None,
            png_after: 240,
            log: false,
            quit_after: None,
            validate: false,
        };
        let mut args = std::env::args().skip(1);
        fn value(args: &mut impl Iterator<Item = String>, name: &str) -> std::result::Result<String, String> {
            args.next().ok_or_else(|| format!("{name} needs a value"))
        }
        fn number<T: std::str::FromStr>(text: String, name: &str) -> std::result::Result<T, String> {
            text.parse().map_err(|_| format!("{name}: '{text}' is not a valid number"))
        }
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print!("{HELP}");
                    std::process::exit(0);
                }
                "--marbles" => o.marbles = number(value(&mut args, &arg)?, &arg)?,
                "--tier" => {
                    let name = value(&mut args, &arg)?;
                    o.tier_index = TIERS
                        .iter()
                        .position(|t| t.name == name)
                        .ok_or_else(|| format!("Unknown tier '{name}'"))?;
                }
                "--size" => {
                    let text = value(&mut args, &arg)?;
                    let (w, h) = text.split_once('x').ok_or("--size needs WxH, such as 1920x1080")?;
                    o.width = number(w.to_string(), &arg)?;
                    o.height = number(h.to_string(), &arg)?;
                }
                "--no-denoise" => o.denoise = false,
                "--no-adaptive" => o.adaptive = false,
                "--no-vsync" => o.vsync = false,
                "--closeup" => o.close_up = true,
                "--pause-camera" => o.paused_camera = true,
                "--benchmark" => o.benchmark_seconds = Some(number(value(&mut args, &arg)?, &arg)?),
                "--warmup" => o.warmup_seconds = number(value(&mut args, &arg)?, &arg)?,
                "--csv" => o.csv_path = Some(value(&mut args, &arg)?),
                "--png" => o.png_path = Some(value(&mut args, &arg)?),
                "--png-after" => o.png_after = number(value(&mut args, &arg)?, &arg)?,
                "--log" => o.log = true,
                "--quit-after" => o.quit_after = Some(number(value(&mut args, &arg)?, &arg)?),
                "--validate" => o.validate = true,
                other => return Err(format!("Unknown option '{other}'. Use --help for the list.")),
            }
        }
        if o.marbles == 0 {
            return Err("--marbles must be at least 1".into());
        }
        Ok(o)
    }

    fn renderer_config(&self) -> RendererConfig {
        RendererConfig {
            marbles: self.marbles,
            tier: TIERS[self.tier_index],
            denoise: self.denoise,
            validate: self.validate,
            vsync: self.vsync,
            width: self.width,
            height: self.height,
        }
    }
}

// ---------------------------------------------------------------------- no-window modes

fn average(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile_95(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted.get(sorted.len() * 95 / 100).or(sorted.last()).copied().unwrap_or(0.0)
}

/// Renders as fast as the GPU allows, with no window and no vsync. Results therefore show the
/// real capacity of the machine, and the display cannot cap them or hide a slowdown.
fn run_benchmark(renderer: &mut Renderer, options: &Options, seconds: f64) -> Result<()> {
    let dt = 1.0 / 60.0;
    let warmup = Instant::now();
    while warmup.elapsed().as_secs_f64() < options.warmup_seconds {
        renderer.render_headless(dt)?;
    }

    println!("elapsed     fps   gpu_avg  gpu_p95    build    trace     post");
    let mut csv = String::from("elapsed_s,fps,gpu_ms_avg,gpu_ms_p95,build_ms,trace_ms,post_ms\n");
    let start = Instant::now();
    let mut window_start = start;
    let mut window: Vec<renderer::FrameTimes> = Vec::new();
    let mut all_totals = Vec::new();
    let mut total_frames = 0u64;
    while start.elapsed().as_secs_f64() < seconds {
        renderer.render_headless(dt)?;
        window.push(renderer.last_times());
        total_frames += 1;
        if window_start.elapsed().as_secs_f64() >= 10.0 {
            let totals: Vec<f32> = window.iter().map(|t| t.total()).collect();
            let part = |f: fn(&renderer::FrameTimes) -> f32| average(&window.iter().map(f).collect::<Vec<_>>());
            let (fps, avg, p95) =
                (window.len() as f64 / window_start.elapsed().as_secs_f64(), average(&totals), percentile_95(&totals));
            let (build, trace, post) = (part(|t| t.build_ms), part(|t| t.trace_ms), part(|t| t.post_ms));
            let elapsed = start.elapsed().as_secs_f64();
            println!("{elapsed:6.0}s {fps:7.1} {avg:9.2} {p95:8.2} {build:8.2} {trace:8.2} {post:8.2}");
            writeln!(csv, "{elapsed:.1},{fps:.2},{avg:.3},{p95:.3},{build:.3},{trace:.3},{post:.3}")?;
            all_totals.extend(totals);
            window.clear();
            window_start = Instant::now();
        }
    }
    all_totals.extend(window.iter().map(|t| t.total()));
    let (w, h) = renderer.render_size();
    println!(
        "RESULT gpu=\"{}\" marbles={} tier={} denoise={} traced={w}x{h} triangles={} fps={:.1} gpu_avg={:.2}ms gpu_p95={:.2}ms",
        renderer.device_name,
        renderer.simulation.count(),
        renderer.tier.name,
        if renderer.denoise { "on" } else { "off" },
        renderer.triangle_count(),
        total_frames as f64 / start.elapsed().as_secs_f64(),
        average(&all_totals),
        percentile_95(&all_totals),
    );
    if let Some(path) = &options.csv_path {
        std::fs::write(path, csv)?;
    }
    Ok(())
}

// -------------------------------------------------------------------------------- window

struct App {
    options: Options,
    window: Option<Window>,
    renderer: Option<Renderer>,
    error: Option<Box<dyn std::error::Error>>,
    adaptive: bool,
    tier_index: usize,
    occluded: bool,
    started: Instant,
    last_draw: Instant,
    stats_start: Instant,
    stats_frames: u32,
    stats_gpu: f32,
    slow_seconds: u32,
    comfortable_seconds: u32,
    /// Doubles after each drop, so the ladder does not bounce between two neighbouring tiers.
    climb_wait_seconds: u32,
    settle_seconds: u32,
}

/// Cinematic is for stills, so automatic quality never tries it.
const HIGHEST_AUTO_TIER: usize = 1;

impl App {
    fn new(options: Options) -> Self {
        let now = Instant::now();
        Self {
            adaptive: options.adaptive,
            tier_index: options.tier_index,
            options,
            window: None,
            renderer: None,
            error: None,
            occluded: false,
            started: now,
            last_draw: now,
            stats_start: now,
            stats_frames: 0,
            stats_gpu: 0.0,
            slow_seconds: 0,
            comfortable_seconds: 0,
            climb_wait_seconds: 20,
            settle_seconds: 3,
        }
    }

    fn start(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = event_loop.create_window(
            Window::default_attributes()
                .with_title("Glass Marbles")
                .with_inner_size(PhysicalSize::new(self.options.width, self.options.height)),
        )?;
        let size = window.inner_size();
        let mut config = self.options.renderer_config();
        (config.width, config.height) = (size.width.max(1), size.height.max(1));
        let handles = WindowHandles {
            display: window.display_handle()?.as_raw(),
            window: window.window_handle()?.as_raw(),
        };
        let mut renderer = Renderer::new(&config, Some(handles))?;
        renderer.close_up = self.options.close_up;
        renderer.camera_paused = self.options.paused_camera;
        println!("GPU: {}", renderer.device_name);
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.started = Instant::now();
        self.last_draw = self.started;
        self.stats_start = self.started;
        Ok(())
    }

    fn set_tier(&mut self, index: usize) -> Result<()> {
        let Some(renderer) = self.renderer.as_mut() else { return Ok(()) };
        if self.options.log {
            println!("tier change: {} -> {}", renderer.tier.name, TIERS[index].name);
        }
        self.tier_index = index;
        self.settle_seconds = 2;
        renderer.set_tier(TIERS[index])
    }

    /// Steers by frames actually delivered. GPU time is no guide to headroom: under vsync many
    /// GPUs lower their clock, so the same work takes longer when there is time to spare. The only
    /// way to learn whether a higher tier fits is to try it and step back if it does not.
    fn adapt_quality(&mut self, fps: f64) -> Result<()> {
        // The system draws a covered window less often, which says nothing about GPU capacity.
        if self.occluded {
            self.slow_seconds = 0;
            self.comfortable_seconds = 0;
            self.settle_seconds = self.settle_seconds.max(2);
            return Ok(());
        }
        if self.settle_seconds > 0 {
            self.settle_seconds -= 1; // startup and tier changes stall briefly; ignore those seconds
            return Ok(());
        }
        if fps < 57.0 {
            self.slow_seconds += 1;
            self.comfortable_seconds = 0;
            if self.slow_seconds >= 2 && self.tier_index < TIERS.len() - 1 {
                self.slow_seconds = 0;
                self.climb_wait_seconds = (self.climb_wait_seconds * 2).min(300);
                self.set_tier(self.tier_index + 1)?;
            }
            return Ok(());
        }
        self.slow_seconds = 0;
        self.comfortable_seconds += 1;
        if self.comfortable_seconds >= self.climb_wait_seconds && self.tier_index > HIGHEST_AUTO_TIER {
            self.comfortable_seconds = 0;
            self.set_tier(self.tier_index - 1)?;
        }
        Ok(())
    }

    fn once_per_second(&mut self) -> Result<()> {
        let elapsed = self.stats_start.elapsed().as_secs_f64();
        if elapsed < 1.0 {
            return Ok(());
        }
        let fps = self.stats_frames as f64 / elapsed;
        let gpu = self.stats_gpu / self.stats_frames.max(1) as f32;
        (self.stats_start, self.stats_frames, self.stats_gpu) = (Instant::now(), 0, 0.0);

        if self.adaptive {
            self.adapt_quality(fps)?;
        }
        let (Some(renderer), Some(window)) = (self.renderer.as_ref(), self.window.as_ref()) else { return Ok(()) };
        let (w, h) = renderer.render_size();
        let times = renderer.last_times();
        window.set_title(&format!(
            "Glass Marbles - {fps:.0} FPS - GPU {gpu:.1} ms - {}{} - {w}x{h} - {} marbles",
            renderer.tier.name,
            if self.adaptive { " (auto)" } else { "" },
            renderer.simulation.count(),
        ));
        if self.options.log {
            println!(
                "window fps={fps:.1} gpu={gpu:.1}ms build={:.1}ms trace={:.1}ms post={:.1}ms visible={} tier={} traced={w}x{h}",
                times.build_ms,
                times.trace_ms,
                times.post_ms,
                u8::from(!self.occluded),
                renderer.tier.name,
            );
        }
        Ok(())
    }

    fn handle_key(&mut self, key: &Key, event_loop: &ActiveEventLoop) -> Result<()> {
        match key {
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Named(NamedKey::Space) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.camera_paused = !r.camera_paused;
                }
            }
            Key::Character(text) => match text.to_lowercase().as_str() {
                digit @ ("1" | "2" | "3" | "4" | "5") => {
                    self.adaptive = false;
                    self.set_tier(digit.parse::<usize>()? - 1)?;
                }
                "a" => self.adaptive = !self.adaptive,
                "c" => {
                    if let Some(r) = self.renderer.as_mut() {
                        r.close_up = !r.close_up;
                    }
                }
                "d" => {
                    if let Some(r) = self.renderer.as_mut() {
                        r.denoise = !r.denoise;
                    }
                }
                "f" => {
                    if let Some(w) = self.window.as_ref() {
                        let next = if w.fullscreen().is_some() { None } else { Some(Fullscreen::Borderless(None)) };
                        w.set_fullscreen(next);
                    }
                }
                "]" => {
                    if let Some(r) = self.renderer.as_mut() {
                        r.set_marble_count((r.simulation.count() * 2).min(3200))?;
                    }
                }
                "[" => {
                    if let Some(r) = self.renderer.as_mut() {
                        r.set_marble_count((r.simulation.count() / 2).max(10))?;
                    }
                }
                _ => {}
            },
            _ => {}
        }
        Ok(())
    }

    fn fail(&mut self, error: Box<dyn std::error::Error>, event_loop: &ActiveEventLoop) {
        self.error = Some(error);
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            if let Err(error) = self.start(event_loop) {
                self.fail(error, event_loop);
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let result: Result<()> = (|| {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Occluded(occluded) => self.occluded = occluded,
                WindowEvent::Resized(size) => {
                    if let Some(renderer) = self.renderer.as_mut() {
                        renderer.resize(size.width, size.height)?;
                        self.settle_seconds = self.settle_seconds.max(2);
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat => {
                        self.handle_key(&event.logical_key, event_loop)?;
                    }
                WindowEvent::RedrawRequested => {
                    if self.options.quit_after.is_some_and(|s| self.started.elapsed().as_secs_f64() >= s) {
                        event_loop.exit();
                        return Ok(());
                    }
                    let Some(renderer) = self.renderer.as_mut() else { return Ok(()) };
                    let minimised = self.window.as_ref().is_some_and(|w| {
                        let size = w.inner_size();
                        size.width == 0 || size.height == 0
                    });
                    if minimised {
                        return Ok(());
                    }
                    let dt = self.last_draw.elapsed().as_secs_f32();
                    self.last_draw = Instant::now();
                    renderer.render_to_window(dt)?;
                    self.stats_frames += 1;
                    self.stats_gpu += renderer.last_times().total();
                    self.once_per_second()?;
                }
                _ => {}
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.fail(error, event_loop);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn run() -> Result<()> {
    let options = Options::parse()?;

    if options.benchmark_seconds.is_some() || options.png_path.is_some() {
        let mut renderer = Renderer::new(&options.renderer_config(), None)?;
        renderer.close_up = options.close_up;
        renderer.camera_paused = options.paused_camera;
        println!("GPU: {}", renderer.device_name);
        if let Some(path) = &options.png_path {
            for _ in 0..options.png_after {
                renderer.render_headless(1.0 / 60.0)?;
            }
            renderer.save_png(path)?;
            println!("Saved {path}");
        }
        if let Some(seconds) = options.benchmark_seconds {
            run_benchmark(&mut renderer, &options, seconds)?;
        }
        return Ok(());
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(options);
    event_loop.run_app(&mut app)?;
    // The renderer owns a Vulkan surface for this window, so it has to go first.
    app.renderer = None;
    app.window = None;
    match app.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}
