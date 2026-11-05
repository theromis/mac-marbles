import Foundation
import Metal
import MetalFX
import simd

/// Layout matches `Uniforms` in PathTracer.metal.
struct Uniforms {
    var viewProj = matrix_identity_float4x4
    var prevViewProj = matrix_identity_float4x4
    var camPos = SIMD4<Float>()
    var camRight = SIMD4<Float>()
    var camUp = SIMD4<Float>()
    var camForward = SIMD4<Float>()
    var paperOrigin = SIMD4<Float>()
    var paperAxes = SIMD4<Float>()
    var jitter = SIMD2<Float>()
    var size = SIMD2<UInt32>()
    var frame: UInt32 = 0
    var maxBounces: UInt32 = 0
    var diffuseBounces: UInt32 = 0
    var sphereCount: UInt32 = 0
    var time: Float = 0
    var pad0: UInt32 = 0
    var pad1: UInt32 = 0
}

/// One step of the quality ladder. Index 0 is the highest quality.
struct QualityTier {
    var name: String
    /// Denoiser output size relative to the window; the display pass stretches the rest.
    var outputScale: Float
    /// Traced size relative to the denoiser output.
    var inputScale: Float
    var bounces: UInt32
    var diffuseBounces: UInt32

    static let ladder: [QualityTier] = [
        QualityTier(name: "cinematic", outputScale: 1, inputScale: 0.5, bounces: 14, diffuseBounces: 2),
        QualityTier(name: "high", outputScale: 0.75, inputScale: 0.5, bounces: 12, diffuseBounces: 2),
        QualityTier(name: "medium", outputScale: 0.75, inputScale: 0.5, bounces: 8, diffuseBounces: 1),
        QualityTier(name: "balanced", outputScale: 0.65625, inputScale: 0.5, bounces: 6, diffuseBounces: 1),
        QualityTier(name: "low", outputScale: 0.5625, inputScale: 0.5, bounces: 6, diffuseBounces: 1),
    ]
}

struct RenderStats {
    var gpuMilliseconds: Double = 0
    var cpuMilliseconds: Double = 0
}

final class Renderer {
    static let framesInFlight = 3

    let device: MTLDevice
    let queue: MTLCommandQueue
    private let tracePipeline: MTLComputePipelineState
    private let displayPipeline: MTLRenderPipelineState
    private let intersectionHandle: MTLFunctionHandle

    private let path = TrackPath()
    private var simulation: MarbleSimulation
    private let vertexBuffer: MTLBuffer
    let triangleCount: Int
    private let triangleAccel: MTLAccelerationStructure

    /// All per-frame state that the GPU may still be reading while the CPU prepares the next.
    private struct FrameResources {
        var spheres: MTLBuffer
        var boxes: MTLBuffer
        var uniforms: MTLBuffer
        var instances: MTLBuffer
        var table: MTLIntersectionFunctionTable
        var sphereAccel: MTLAccelerationStructure
        var topAccel: MTLAccelerationStructure
        var sphereDescriptor: MTLPrimitiveAccelerationStructureDescriptor
        var topDescriptor: MTLInstanceAccelerationStructureDescriptor
        var scratch: MTLBuffer
    }
    private var frames: [FrameResources] = []
    private let inFlight = DispatchSemaphore(value: Renderer.framesInFlight)

    // Render targets.
    private var windowWidth = 0
    private var windowHeight = 0
    private var lastGpuEnd: Double = 0
    private(set) var outputWidth = 0
    private(set) var outputHeight = 0
    private(set) var inputWidth = 0
    private(set) var inputHeight = 0
    private var color: MTLTexture!
    private var depth: MTLTexture!
    private var motion: MTLTexture!
    private var normal: MTLTexture!
    private var diffuse: MTLTexture!
    private var specular: MTLTexture!
    private var roughness: MTLTexture!
    private var reactive: MTLTexture!
    private var denoiseMask: MTLTexture!
    private var paperTexture: MTLTexture!
    private var output: MTLTexture!
    private var scaler: MTLFXTemporalDenoisedScaler?
    private var resetHistory = true

    private(set) var tier: QualityTier
    var denoise: Bool
    var cameraPaused = false
    var closeUp = false

    private var frameIndex: UInt32 = 0
    private var cameraPhase: Float = 0
    private var previousViewProj: simd_float4x4?
    private var lastPhysicsTime: Double = 0

    var marbleCount: Int { simulation.count }

    init(device: MTLDevice, shaderSource: String, marbles: Int, tier: QualityTier, denoise: Bool) throws {
        self.device = device
        self.tier = tier
        self.denoise = denoise
        guard device.supportsRaytracing else { throw RendererError.message("This GPU has no ray tracing support") }
        queue = device.makeCommandQueue()!

        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        let library = try device.makeLibrary(source: shaderSource, options: options)

        let intersection = library.makeFunction(name: "sphereIntersection")!
        let linked = MTLLinkedFunctions()
        linked.functions = [intersection]
        let traceDescriptor = MTLComputePipelineDescriptor()
        traceDescriptor.computeFunction = library.makeFunction(name: "pathTrace")!
        traceDescriptor.linkedFunctions = linked
        traceDescriptor.label = "pathTrace"
        tracePipeline = try device.makeComputePipelineState(descriptor: traceDescriptor, options: [], reflection: nil)
        intersectionHandle = tracePipeline.functionHandle(function: intersection)!

        let displayDescriptor = MTLRenderPipelineDescriptor()
        displayDescriptor.vertexFunction = library.makeFunction(name: "fullscreenVertex")
        displayDescriptor.fragmentFunction = library.makeFunction(name: "displayFragment")
        displayDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        displayPipeline = try device.makeRenderPipelineState(descriptor: displayDescriptor)

        simulation = MarbleSimulation(count: marbles, path: path)

        let mesh = buildSceneMesh(path: path)
        triangleCount = mesh.count / 3
        vertexBuffer = mesh.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
        }

        // Static wood geometry: built once, optimised for trace speed, then never touched.
        let geometry = MTLAccelerationStructureTriangleGeometryDescriptor()
        geometry.vertexBuffer = vertexBuffer
        geometry.vertexStride = MemoryLayout<Vertex>.stride
        geometry.triangleCount = triangleCount
        geometry.opaque = true
        let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
        descriptor.geometryDescriptors = [geometry]
        triangleAccel = try Self.buildOnce(device: device, queue: queue, descriptor: descriptor)

        paperTexture = makeNewspaperTexture(device: device, queue: queue)
        try allocateFrames()
    }

    enum RendererError: Error {
        case message(String)
    }

    private static func buildOnce(device: MTLDevice, queue: MTLCommandQueue,
                                  descriptor: MTLPrimitiveAccelerationStructureDescriptor) throws -> MTLAccelerationStructure {
        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard let accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate),
              let commands = queue.makeCommandBuffer(),
              let encoder = commands.makeAccelerationStructureCommandEncoder() else {
            throw RendererError.message("Could not allocate the scene's acceleration structure")
        }
        encoder.build(accelerationStructure: accel, descriptor: descriptor, scratchBuffer: scratch, scratchBufferOffset: 0)
        encoder.endEncoding()
        commands.commit()
        commands.waitUntilCompleted()
        return accel
    }

    private func allocateFrames() throws {
        frames.removeAll()
        let count = simulation.count
        for _ in 0..<Self.framesInFlight {
            let spheres = device.makeBuffer(length: count * MemoryLayout<GPUSphere>.stride, options: .storageModeShared)!
            let boxes = device.makeBuffer(length: count * MemoryLayout<MTLBox>.stride, options: .storageModeShared)!
            let uniforms = device.makeBuffer(length: MemoryLayout<Uniforms>.stride, options: .storageModeShared)!

            let tableDescriptor = MTLIntersectionFunctionTableDescriptor()
            tableDescriptor.functionCount = 1
            let table = tracePipeline.makeIntersectionFunctionTable(descriptor: tableDescriptor)!
            table.setFunction(intersectionHandle, index: 0)
            table.setBuffer(spheres, offset: 0, index: 0)

            let geometry = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
            geometry.boundingBoxBuffer = boxes
            geometry.boundingBoxStride = MemoryLayout<MTLBox>.stride
            geometry.boundingBoxCount = count
            geometry.intersectionFunctionTableOffset = 0
            let sphereDescriptor = MTLPrimitiveAccelerationStructureDescriptor()
            sphereDescriptor.geometryDescriptors = [geometry]
            // Rebuilt every frame, so build time matters more than a perfect tree.
            sphereDescriptor.usage = .preferFastBuild
            let sphereSizes = device.accelerationStructureSizes(descriptor: sphereDescriptor)
            let sphereAccel = device.makeAccelerationStructure(size: sphereSizes.accelerationStructureSize)!

            let instances = device.makeBuffer(length: 2 * MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride,
                                              options: .storageModeShared)!
            let items = instances.contents().bindMemory(to: MTLAccelerationStructureInstanceDescriptor.self, capacity: 2)
            var identity = MTLPackedFloat4x3()
            identity.columns.0 = MTLPackedFloat3Make(1, 0, 0)
            identity.columns.1 = MTLPackedFloat3Make(0, 1, 0)
            identity.columns.2 = MTLPackedFloat3Make(0, 0, 1)
            identity.columns.3 = MTLPackedFloat3Make(0, 0, 0)
            for i in 0..<2 {
                items[i].transformationMatrix = identity
                items[i].options = []
                items[i].mask = 0xFF
                items[i].intersectionFunctionTableOffset = 0
                items[i].accelerationStructureIndex = UInt32(i)
            }

            let topDescriptor = MTLInstanceAccelerationStructureDescriptor()
            topDescriptor.instanceCount = 2
            topDescriptor.instanceDescriptorBuffer = instances
            topDescriptor.instancedAccelerationStructures = [triangleAccel, sphereAccel]
            topDescriptor.usage = .preferFastBuild
            let topSizes = device.accelerationStructureSizes(descriptor: topDescriptor)
            let topAccel = device.makeAccelerationStructure(size: topSizes.accelerationStructureSize)!

            let scratchSize = max(sphereSizes.buildScratchBufferSize, topSizes.buildScratchBufferSize)
            let scratch = device.makeBuffer(length: scratchSize * 2 + 256, options: .storageModePrivate)!

            frames.append(FrameResources(spheres: spheres, boxes: boxes, uniforms: uniforms, instances: instances,
                                         table: table, sphereAccel: sphereAccel, topAccel: topAccel,
                                         sphereDescriptor: sphereDescriptor, topDescriptor: topDescriptor,
                                         scratch: scratch))
        }
    }

    // MARK: - Configuration

    func setMarbleCount(_ count: Int) throws {
        waitForIdle()
        simulation = MarbleSimulation(count: count, path: path)
        try allocateFrames()
        resetHistory = true
    }

    func setTier(_ newTier: QualityTier) throws {
        let resolutionChanged = newTier.outputScale != tier.outputScale || newTier.inputScale != tier.inputScale
        tier = newTier
        if resolutionChanged { try resize(width: windowWidth, height: windowHeight) }
    }

    func setDenoise(_ enabled: Bool) {
        denoise = enabled
        resetHistory = true
    }

    private func waitForIdle() {
        for _ in 0..<Self.framesInFlight { inFlight.wait() }
        for _ in 0..<Self.framesInFlight { inFlight.signal() }
    }

    func resize(width: Int, height: Int) throws {
        guard width > 0 && height > 0 else { return }
        waitForIdle()
        windowWidth = width
        windowHeight = height
        outputWidth = max(Int(Float(width) * tier.outputScale), 16)
        outputHeight = max(Int(Float(height) * tier.outputScale), 16)
        inputWidth = max(Int(Float(outputWidth) * tier.inputScale), 16)
        inputHeight = max(Int(Float(outputHeight) * tier.inputScale), 16)

        let scalerDescriptor = MTLFXTemporalDenoisedScalerDescriptor()
        scalerDescriptor.colorTextureFormat = .rgba16Float
        scalerDescriptor.depthTextureFormat = .r32Float
        scalerDescriptor.motionTextureFormat = .rg16Float
        scalerDescriptor.normalTextureFormat = .rgba16Float
        scalerDescriptor.diffuseAlbedoTextureFormat = .rgba8Unorm
        scalerDescriptor.specularAlbedoTextureFormat = .rgba8Unorm
        scalerDescriptor.roughnessTextureFormat = .r8Unorm
        scalerDescriptor.isReactiveMaskTextureEnabled = true
        scalerDescriptor.reactiveMaskTextureFormat = .r8Unorm
        scalerDescriptor.isDenoiseStrengthMaskTextureEnabled = true
        scalerDescriptor.denoiseStrengthMaskTextureFormat = .r8Unorm
        scalerDescriptor.outputTextureFormat = .rgba16Float
        scalerDescriptor.inputWidth = inputWidth
        scalerDescriptor.inputHeight = inputHeight
        scalerDescriptor.outputWidth = outputWidth
        scalerDescriptor.outputHeight = outputHeight
        // Compile in the background: compiling while drawing froze the picture for about 2 seconds.
        scalerDescriptor.requiresSynchronousInitialization = false
        guard let newScaler = scalerDescriptor.makeTemporalDenoisedScaler(device: device) else {
            throw RendererError.message("Could not create the MetalFX denoiser for \(inputWidth)x\(inputHeight) -> \(width)x\(height)")
        }
        scaler = newScaler

        func target(_ format: MTLPixelFormat, _ w: Int, _ h: Int, _ usage: MTLTextureUsage) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: w, height: h, mipmapped: false)
            d.storageMode = .private
            d.usage = usage.union(.shaderWrite)
            return device.makeTexture(descriptor: d)!
        }
        color = target(.rgba16Float, inputWidth, inputHeight, newScaler.colorTextureUsage.union(.shaderRead))
        depth = target(.r32Float, inputWidth, inputHeight, newScaler.depthTextureUsage)
        motion = target(.rg16Float, inputWidth, inputHeight, newScaler.motionTextureUsage)
        normal = target(.rgba16Float, inputWidth, inputHeight, newScaler.normalTextureUsage)
        diffuse = target(.rgba8Unorm, inputWidth, inputHeight, newScaler.diffuseAlbedoTextureUsage)
        specular = target(.rgba8Unorm, inputWidth, inputHeight, newScaler.specularAlbedoTextureUsage)
        roughness = target(.r8Unorm, inputWidth, inputHeight, newScaler.roughnessTextureUsage)
        reactive = target(.r8Unorm, inputWidth, inputHeight, newScaler.reactiveMaskTextureUsage)
        denoiseMask = target(.r8Unorm, inputWidth, inputHeight, newScaler.denoiseStrengthMaskTextureUsage)
        output = target(.rgba16Float, outputWidth, outputHeight, newScaler.outputTextureUsage.union(.shaderRead))
        resetHistory = true
    }

    // MARK: - Frame

    /// Encodes simulation upload, acceleration structure builds, tracing and denoising.
    /// Returns the HDR image to display. `completed` fires when the GPU has finished.
    func encodeFrame(into commands: MTLCommandBuffer, dt: Float,
                     completed: @escaping (RenderStats) -> Void) -> MTLTexture {
        inFlight.wait()
        let cpuStart = CACurrentMediaTimeCompat()
        let frame = frames[Int(frameIndex) % Self.framesInFlight]

        simulation.beginFrame()
        simulation.step(dt: dt)
        simulation.writeSpheres(to: frame.spheres.contents().bindMemory(to: GPUSphere.self, capacity: simulation.count),
                                boxes: frame.boxes.contents().bindMemory(to: MTLBox.self, capacity: simulation.count))

        // ------------------------------------------------- camera
        // Swing around the lit side. Behind the track the key light is out of sight,
        // the marbles lose their highlights and stop reading as glass.
        if !cameraPaused { cameraPhase += 0.12 * dt }
        let cameraAngle = 0.6 + 0.6 * sin(cameraPhase)
        let target: SIMD3<Float>
        let eye: SIMD3<Float>
        if closeUp {
            // Low over the newspaper, to see the print through the marbles rolling across it.
            let sway = 0.12 * sin(cameraPhase)
            target = SIMD3(Layout.paperCenter.x, 0.12, Layout.paperCenter.y)
            eye = target + SIMD3(-1.45 + sway, 0.62, -0.75 - sway)
        } else {
            target = SIMD3(0, 0.95, 0)
            eye = SIMD3(cos(cameraAngle) * 5.8, 3.8, sin(cameraAngle) * 5.8)
        }
        let forward = simd_normalize(target - eye)
        let right = simd_normalize(simd_cross(forward, SIMD3(0, 1, 0)))
        let up = simd_cross(right, forward)
        let aspect = Float(inputWidth) / Float(inputHeight)
        let fovY: Float = 38 * .pi / 180
        let tanHalf = tan(fovY / 2)
        let view = lookAt(eye: eye, right: right, up: up, forward: forward)
        let projection = perspective(tanHalfFov: tanHalf, aspect: aspect, near: 0.1, far: 200)
        let viewProj = projection * view

        var uniforms = Uniforms()
        uniforms.viewProj = viewProj
        uniforms.prevViewProj = previousViewProj ?? viewProj
        uniforms.camPos = SIMD4(eye, 0)
        uniforms.camRight = SIMD4(right, aspect)
        uniforms.camUp = SIMD4(up, tanHalf)
        uniforms.camForward = SIMD4(forward, 0)
        uniforms.paperOrigin = SIMD4(Layout.paperCenter.x, Layout.paperCenter.y, 0, 0)
        let paperAcross = Layout.paperAcross / Layout.paperHalfWidth
        let paperUp = Layout.paperUp / Layout.paperHalfHeight
        uniforms.paperAxes = SIMD4(paperAcross.x, paperAcross.y, paperUp.x, paperUp.y)
        let jitter = SIMD2(halton(frameIndex % 32 + 1, base: 2) - 0.5, halton(frameIndex % 32 + 1, base: 3) - 0.5)
        uniforms.jitter = denoise ? jitter : .zero
        uniforms.size = SIMD2(UInt32(inputWidth), UInt32(inputHeight))
        uniforms.frame = frameIndex
        uniforms.maxBounces = tier.bounces
        uniforms.diffuseBounces = tier.diffuseBounces
        uniforms.sphereCount = UInt32(simulation.count)
        frame.uniforms.contents().storeBytes(of: uniforms, as: Uniforms.self)
        previousViewProj = viewProj
        let cpuEnd = CACurrentMediaTimeCompat()

        // ------------------------------------------------- acceleration structures
        // Separate encoders: the top level reads the bottom level the first one writes.
        let bottomEncoder = commands.makeAccelerationStructureCommandEncoder()!
        bottomEncoder.build(accelerationStructure: frame.sphereAccel, descriptor: frame.sphereDescriptor,
                            scratchBuffer: frame.scratch, scratchBufferOffset: 0)
        bottomEncoder.endEncoding()
        let topEncoder = commands.makeAccelerationStructureCommandEncoder()!
        topEncoder.build(accelerationStructure: frame.topAccel, descriptor: frame.topDescriptor,
                         scratchBuffer: frame.scratch, scratchBufferOffset: frame.scratch.length / 2)
        topEncoder.endEncoding()

        // ------------------------------------------------- tracing
        let trace = commands.makeComputeCommandEncoder()!
        trace.setComputePipelineState(tracePipeline)
        let targets: [MTLTexture] = [color, depth, motion, normal, diffuse, specular, roughness, reactive, denoiseMask]
        trace.setTextures(targets, range: 0..<targets.count)
        trace.setTexture(paperTexture, index: 9)
        trace.setBuffer(frame.uniforms, offset: 0, index: 0)
        trace.setAccelerationStructure(frame.topAccel, bufferIndex: 1)
        trace.setIntersectionFunctionTable(frame.table, bufferIndex: 2)
        trace.setBuffer(vertexBuffer, offset: 0, index: 3)
        trace.setBuffer(frame.spheres, offset: 0, index: 4)
        trace.useResource(triangleAccel, usage: .read)
        trace.useResource(frame.sphereAccel, usage: .read)
        trace.useResource(frame.boxes, usage: .read)
        trace.dispatchThreads(MTLSize(width: inputWidth, height: inputHeight, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        trace.endEncoding()

        // ------------------------------------------------- denoise and upscale
        var image: MTLTexture = color
        if denoise, let scaler {
            scaler.colorTexture = color
            scaler.depthTexture = depth
            scaler.motionTexture = motion
            scaler.normalTexture = normal
            scaler.diffuseAlbedoTexture = diffuse
            scaler.specularAlbedoTexture = specular
            scaler.roughnessTexture = roughness
            scaler.reactiveMaskTexture = reactive
            scaler.denoiseStrengthMaskTexture = denoiseMask
            scaler.outputTexture = output
            scaler.jitterOffsetX = jitter.x
            scaler.jitterOffsetY = jitter.y
            scaler.motionVectorScaleX = 1
            scaler.motionVectorScaleY = 1
            scaler.worldToViewMatrix = view
            scaler.viewToClipMatrix = projection
            scaler.shouldResetHistory = resetHistory
            resetHistory = false
            scaler.encode(commandBuffer: commands)
            image = output
        }

        let semaphore = inFlight
        commands.addCompletedHandler { buffer in
            // Frames queue up, so a buffer's start can precede the end of the previous one.
            // Counting only the non-overlapping part gives the time the GPU really spent.
            let gpu = (buffer.gpuEndTime - max(buffer.gpuStartTime, self.lastGpuEnd)) * 1000
            self.lastGpuEnd = buffer.gpuEndTime
            semaphore.signal()
            completed(RenderStats(gpuMilliseconds: gpu, cpuMilliseconds: (cpuEnd - cpuStart) * 1000))
        }
        frameIndex &+= 1
        return image
    }

    func encodeDisplay(into commands: MTLCommandBuffer, image: MTLTexture, target pass: MTLRenderPassDescriptor) {
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.setRenderPipelineState(displayPipeline)
        encoder.setFragmentTexture(image, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
    }

    /// Renders `image` through the display pass into a CPU-readable texture.
    func encodeReadback(into commands: MTLCommandBuffer, image: MTLTexture) -> MTLTexture {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: image.width,
                                                         height: image.height, mipmapped: false)
        d.usage = [.renderTarget]
        d.storageMode = .shared
        let texture = device.makeTexture(descriptor: d)!
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = texture
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        encodeDisplay(into: commands, image: image, target: pass)
        return texture
    }
}

// MARK: - Maths

func CACurrentMediaTimeCompat() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

func halton(_ index: UInt32, base: UInt32) -> Float {
    var f: Float = 1
    var result: Float = 0
    var i = index
    while i > 0 {
        f /= Float(base)
        result += f * Float(i % base)
        i /= base
    }
    return result
}

func lookAt(eye: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>, forward: SIMD3<Float>) -> simd_float4x4 {
    simd_float4x4(rows: [
        SIMD4(right, -simd_dot(right, eye)),
        SIMD4(up, -simd_dot(up, eye)),
        SIMD4(-forward, simd_dot(forward, eye)),
        SIMD4(0, 0, 0, 1),
    ])
}

/// Right-handed perspective projection with clip-space depth in [0, 1].
func perspective(tanHalfFov: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1 / tanHalfFov
    let x = y / aspect
    let z = far / (near - far)
    return simd_float4x4(rows: [
        SIMD4(x, 0, 0, 0),
        SIMD4(0, y, 0, 0),
        SIMD4(0, 0, z, z * near),
        SIMD4(0, 0, -1, 0),
    ])
}
