import AppKit
import CoreGraphics
import Foundation
import ImageIO
import Metal
import MetalKit
import UniformTypeIdentifiers

// MARK: - Command line

struct Options {
    var marbles = 150
    var tierIndex = 2
    var width = 2560
    var height = 1664
    var denoise = true
    var adaptive = true
    var benchmarkSeconds: Double?
    var warmupSeconds = 3.0
    var csvPath: String?
    var pngPath: String?
    var pngAfterFrames = 90
    var pausedCamera = false
    var logWindowStats = false
    var closeUp = false
    var quitAfterSeconds: Double?

    init(arguments: [String]) {
        var i = 1
        func value() -> String {
            i += 1
            guard i < arguments.count else { fatalError("Missing value for \(arguments[i - 1])") }
            return arguments[i]
        }
        while i < arguments.count {
            switch arguments[i] {
            case "--marbles": marbles = Int(value())!
            case "--tier": tierIndex = QualityTier.ladder.firstIndex { $0.name == arguments[i + 1] } ?? 2; i += 1
            case "--size":
                let parts = value().split(separator: "x")
                width = Int(parts[0])!
                height = Int(parts[1])!
            case "--no-denoise": denoise = false
            case "--no-adaptive": adaptive = false
            case "--benchmark": benchmarkSeconds = Double(value())!
            case "--warmup": warmupSeconds = Double(value())!
            case "--csv": csvPath = value()
            case "--png": pngPath = value()
            case "--png-after": pngAfterFrames = Int(value())!
            case "--pause-camera": pausedCamera = true
            case "--log": logWindowStats = true
            case "--closeup": closeUp = true
            case "--quit-after": quitAfterSeconds = Double(value())!
            default: fatalError("Unknown argument \(arguments[i])")
            }
            i += 1
        }
    }
}

func loadShaderSource() -> String {
    let candidates = [
        Bundle.main.resourceURL?.appendingPathComponent("PathTracer.metal"),
        URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
            .appendingPathComponent("../Resources/PathTracer.metal"),
    ]
    for case let url? in candidates {
        if let source = try? String(contentsOf: url, encoding: .utf8) { return source }
    }
    fatalError("PathTracer.metal not found next to the app")
}

func thermalName(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func writePNG(_ texture: MTLTexture, to path: String) {
    let width = texture.width
    let height = texture.height
    var bytes = [UInt8](repeating: 0, count: width * height * 4)
    texture.getBytes(&bytes, bytesPerRow: width * 4,
                     from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
    let info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
    let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: info)!
    let image = context.makeImage()!
    let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL,
                                                      UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, image, nil)
    CGImageDestinationFinalize(destination)
}

// MARK: - Headless benchmark

/// Renders as fast as the GPU allows, with no window and no vsync. Results therefore show
/// the real capacity of the machine, and the display cannot cap them or hide a slowdown.
func runBenchmark(renderer: Renderer, options: Options, seconds: Double) {
    final class Log: @unchecked Sendable {
        let lock = NSLock()
        var gpu: [Double] = []
        var cpu: [Double] = []
    }
    let log = Log()
    let dt: Float = 1.0 / 60.0
    var csv = "elapsed_s,fps,gpu_ms_avg,gpu_ms_p95,cpu_ms_avg,thermal\n"

    func summarize(_ values: [Double]) -> (avg: Double, p95: Double) {
        guard !values.isEmpty else { return (0, 0) }
        let sorted = values.sorted()
        return (values.reduce(0, +) / Double(values.count), sorted[min(sorted.count * 95 / 100, sorted.count - 1)])
    }

    let warmupEnd = CACurrentMediaTimeCompat() + options.warmupSeconds
    while CACurrentMediaTimeCompat() < warmupEnd {
        autoreleasepool {
            let commands = renderer.queue.makeCommandBuffer()!
            _ = renderer.encodeFrame(into: commands, dt: dt) { _ in }
            commands.commit()
        }
    }

    let start = CACurrentMediaTimeCompat()
    var windowStart = start
    var windowFrames = 0
    var totalFrames = 0
    var allGpu: [Double] = []
    let reportEvery = 10.0
    let end = start + seconds
    print("elapsed   fps   gpu_avg  gpu_p95  cpu_ms  thermal")
    while CACurrentMediaTimeCompat() < end {
        autoreleasepool {
            let commands = renderer.queue.makeCommandBuffer()!
            _ = renderer.encodeFrame(into: commands, dt: dt) { stats in
                log.lock.lock()
                log.gpu.append(stats.gpuMilliseconds)
                log.cpu.append(stats.cpuMilliseconds)
                log.lock.unlock()
            }
            commands.commit()
        }
        windowFrames += 1
        totalFrames += 1
        let now = CACurrentMediaTimeCompat()
        if now - windowStart >= reportEvery {
            log.lock.lock()
            let gpu = log.gpu
            let cpu = log.cpu
            log.gpu.removeAll(keepingCapacity: true)
            log.cpu.removeAll(keepingCapacity: true)
            log.lock.unlock()
            allGpu += gpu
            let g = summarize(gpu)
            let c = summarize(cpu)
            let fps = Double(windowFrames) / (now - windowStart)
            let thermal = thermalName(ProcessInfo.processInfo.thermalState)
            print(String(format: "%6.0fs %6.1f %7.2f %8.2f %7.2f  %@", now - start, fps, g.avg, g.p95, c.avg, thermal))
            csv += String(format: "%.1f,%.2f,%.3f,%.3f,%.3f,%@\n", now - start, fps, g.avg, g.p95, c.avg, thermal)
            windowStart = now
            windowFrames = 0
        }
    }
    let elapsed = CACurrentMediaTimeCompat() - start
    let g = summarize(allGpu)
    print(String(format: "RESULT marbles=%d tier=%@ denoise=%@ input=%dx%d output=%dx%d triangles=%d fps=%.1f gpu_avg=%.2fms gpu_p95=%.2fms thermal=%@",
                 renderer.marbleCount, renderer.tier.name, renderer.denoise ? "on" : "off",
                 renderer.inputWidth, renderer.inputHeight, renderer.outputWidth, renderer.outputHeight,
                 renderer.triangleCount, Double(totalFrames) / elapsed, g.avg, g.p95,
                 thermalName(ProcessInfo.processInfo.thermalState)))
    if let path = options.csvPath { try? csv.write(toFile: path, atomically: true, encoding: .utf8) }
}

func runScreenshot(renderer: Renderer, options: Options, path: String) {
    for frame in 0..<options.pngAfterFrames {
        autoreleasepool {
            let commands = renderer.queue.makeCommandBuffer()!
            let image = renderer.encodeFrame(into: commands, dt: 1.0 / 60.0) { _ in }
            if frame == options.pngAfterFrames - 1 {
                let readback = renderer.encodeReadback(into: commands, image: image)
                commands.commit()
                commands.waitUntilCompleted()
                writePNG(readback, to: path)
            } else {
                commands.commit()
            }
        }
    }
}

// MARK: - Interactive window

final class MarbleView: MTKView {
    var onKey: ((String) -> Void)?
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 {  // Escape
            NSApp.terminate(nil)
            return
        }
        onKey?(event.charactersIgnoringModifiers?.lowercased() ?? "")
    }
}

/// Draws frames, shows the on-screen stats, and steps the quality ladder to hold 60 FPS.
final class ViewController: NSObject, MTKViewDelegate {
    private let renderer: Renderer
    private let label: NSTextField
    private var adaptive: Bool
    private var tierIndex: Int
    private var lastDraw = CACurrentMediaTimeCompat()
    private var hudStart = CACurrentMediaTimeCompat()
    private var hudFrames = 0
    private let lock = NSLock()
    private var gpuTotal = 0.0
    private var gpuSamples = 0
    private var physicsTotal = 0.0
    private var encodeTotal = 0.0
    private var drawableTotal = 0.0
    private var comfortableSeconds = 0
    /// Doubles after each drop, so the ladder does not bounce between two neighbouring tiers.
    private var climbWaitSeconds = 20
    private var slowSeconds = 0
    private var settleSeconds = 3
    /// Cinematic cannot reach 60 FPS on this class of machine, so auto mode never tries it.
    private let highestAutoTier = 1
    private var hudVisible = true
    private let logStats: Bool

    init(renderer: Renderer, label: NSTextField, options: Options) {
        self.renderer = renderer
        self.label = label
        adaptive = options.adaptive
        tierIndex = options.tierIndex
        logStats = options.logWindowStats
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        do {
            try renderer.resize(width: Int(size.width), height: Int(size.height))
        } catch {
            fatalError("\(error)")
        }
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTimeCompat()
        let dt = Float(now - lastDraw)
        lastDraw = now
        guard renderer.outputWidth > 0 else { return }
        let commands = renderer.queue.makeCommandBuffer()!
        let encodeStart = CACurrentMediaTimeCompat()
        let image = renderer.encodeFrame(into: commands, dt: dt) { [weak self] stats in
            guard let self else { return }
            self.lock.lock()
            self.gpuTotal += stats.gpuMilliseconds
            self.physicsTotal += stats.cpuMilliseconds
            self.gpuSamples += 1
            self.lock.unlock()
        }
        let drawableStart = CACurrentMediaTimeCompat()
        encodeTotal += (drawableStart - encodeStart) * 1000
        let pass = view.currentRenderPassDescriptor
        let drawable = view.currentDrawable
        drawableTotal += (CACurrentMediaTimeCompat() - drawableStart) * 1000
        if let pass, let drawable {
            renderer.encodeDisplay(into: commands, image: image, target: pass)
            commands.present(drawable)
        }
        commands.commit()
        hudFrames += 1
        if now - hudStart >= 1 { updateOncePerSecond(now) }
    }

    private func updateOncePerSecond(_ now: Double) {
        lock.lock()
        let gpu = gpuSamples > 0 ? gpuTotal / Double(gpuSamples) : 0
        let physics = gpuSamples > 0 ? physicsTotal / Double(gpuSamples) : 0
        physicsTotal = 0
        gpuTotal = 0
        gpuSamples = 0
        lock.unlock()
        let fps = Double(hudFrames) / (now - hudStart)
        let frameCount = Double(max(hudFrames, 1))
        let encodeWait = encodeTotal / frameCount
        let drawableWait = drawableTotal / frameCount
        encodeTotal = 0
        drawableTotal = 0
        hudFrames = 0
        hudStart = now

        if adaptive {
            adaptQuality(fps: fps)
        }
        if logStats {
            let visible = label.window?.occlusionState.contains(.visible) ?? false
            print(String(format: "window fps=%.1f gpu=%.1fms cpu_setup=%.1fms physics=%.1fms drawable_wait=%.1fms active=%d visible=%d tier=%@ heat=%@",
                         fps, gpu, encodeWait, physics, drawableWait, NSApp.isActive ? 1 : 0, visible ? 1 : 0,
                         renderer.tier.name, thermalName(ProcessInfo.processInfo.thermalState)))
            fflush(stdout)
        }
        label.isHidden = !hudVisible
        label.stringValue = String(
            format: "%.0f FPS   GPU %.1f ms   %@%@   %dx%d → %dx%d   %d marbles   heat: %@\n"
                + "[ ] marbles  1-5 quality  A auto  D denoiser  Space camera  C newspaper  H hide  F fullscreen  Esc quit",
            fps, gpu, renderer.tier.name, adaptive ? " (auto)" : "",
            renderer.inputWidth, renderer.inputHeight, renderer.outputWidth, renderer.outputHeight,
            renderer.marbleCount, thermalName(ProcessInfo.processInfo.thermalState))
    }

    /// Steers by frames actually delivered. GPU time is no guide to headroom here: under
    /// vsync the GPU lowers its clock, so the same work takes longer when there is time to spare.
    /// The only way to learn whether a higher tier fits is to try it and step back if it does not.
    private func adaptQuality(fps: Double) {
        // macOS draws a covered window less often, which says nothing about GPU capacity.
        guard label.window?.occlusionState.contains(.visible) == true else {
            slowSeconds = 0
            comfortableSeconds = 0
            settleSeconds = max(settleSeconds, 2)
            return
        }
        if settleSeconds > 0 {
            settleSeconds -= 1  // startup and tier changes stall briefly; ignore those seconds
            return
        }
        if fps < 57 {
            slowSeconds += 1
            comfortableSeconds = 0
            if slowSeconds >= 2 && tierIndex < QualityTier.ladder.count - 1 {
                setTier(tierIndex + 1)
                slowSeconds = 0
                climbWaitSeconds = min(climbWaitSeconds * 2, 300)
            }
            return
        }
        slowSeconds = 0
        comfortableSeconds += 1
        if comfortableSeconds >= climbWaitSeconds && tierIndex > highestAutoTier {
            setTier(tierIndex - 1)
            comfortableSeconds = 0
        }
    }

    private func setTier(_ index: Int) {
        settleSeconds = 2
        if logStats { print("tier change: \(renderer.tier.name) -> \(QualityTier.ladder[index].name)") }
        tierIndex = index
        try? renderer.setTier(QualityTier.ladder[index])
    }

    func handleKey(_ key: String, window: NSWindow?) {
        switch key {
        case "1", "2", "3", "4", "5":
            adaptive = false
            setTier(Int(key)! - 1)
        case "a": adaptive.toggle()
        case "c": renderer.closeUp.toggle()
        case "d": renderer.setDenoise(!renderer.denoise)
        case " ": renderer.cameraPaused.toggle()
        case "h":
            hudVisible.toggle()
            label.isHidden = !hudVisible
        case "f": window?.toggleFullScreen(nil)
        case "]": try? renderer.setMarbleCount(min(renderer.marbleCount * 2, 3200))
        case "[": try? renderer.setMarbleCount(max(renderer.marbleCount / 2, 10))
        default: break
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    let options: Options
    let renderer: Renderer
    var window: NSWindow!
    var controller: ViewController!
    private var activity: NSObjectProtocol?

    init(options: Options, renderer: Renderer) {
        self.options = options
        self.renderer = renderer
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Real-time rendering: opt out of App Nap, which slows the draw loop in the background.
        activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Real-time rendering")
        let frame = NSRect(x: 0, y: 0, width: 1280, height: 832)
        window = NSWindow(contentRect: frame, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Glass Marbles"
        window.center()

        let view = MarbleView(frame: frame, device: renderer.device)
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.framebufferOnly = true
        view.preferredFramesPerSecond = 60
        view.autoresizingMask = [.width, .height]

        let label = NSTextField(labelWithString: "")
        label.font = .monospacedSystemFont(ofSize: 12, weight: .medium)
        label.textColor = .white
        label.maximumNumberOfLines = 2
        label.wantsLayer = true
        label.layer?.shadowColor = NSColor.black.cgColor
        label.layer?.shadowOpacity = 1
        label.layer?.shadowRadius = 2
        label.layer?.shadowOffset = .zero
        label.frame = NSRect(x: 12, y: 8, width: 1200, height: 34)
        label.autoresizingMask = [.maxYMargin, .width]
        view.addSubview(label)

        controller = ViewController(renderer: renderer, label: label, options: options)
        view.delegate = controller
        view.onKey = { [weak self] key in self?.controller.handleKey(key, window: self?.window) }

        window.contentView = view
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(view)
        NSApp.activate(ignoringOtherApps: true)
        // A timer, not the draw loop: drawing stops while the display is asleep.
        if let seconds = options.quitAfterSeconds {
            Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { _ in NSApp.terminate(nil) }
        }
        controller.mtkView(view, drawableSizeWillChange: view.drawableSize)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

// MARK: - Entry point

let options = Options(arguments: CommandLine.arguments)
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
guard MTLFXSupport.denoiserAvailable(on: device) else {
    fatalError("MetalFX denoising is not supported here (needs macOS 26 and Apple silicon M1 or newer)")
}

let renderer: Renderer
do {
    renderer = try Renderer(device: device, shaderSource: loadShaderSource(), marbles: options.marbles,
                            tier: QualityTier.ladder[options.tierIndex], denoise: options.denoise)
    renderer.cameraPaused = options.pausedCamera
    renderer.closeUp = options.closeUp
} catch {
    fatalError("Could not start the renderer: \(error)")
}

if options.benchmarkSeconds != nil || options.pngPath != nil {
    try! renderer.resize(width: options.width, height: options.height)
    if let path = options.pngPath { runScreenshot(renderer: renderer, options: options, path: path) }
    if let seconds = options.benchmarkSeconds { runBenchmark(renderer: renderer, options: options, seconds: seconds) }
    exit(0)
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate(options: options, renderer: renderer)
app.delegate = delegate
app.run()
