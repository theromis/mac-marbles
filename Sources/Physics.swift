import Foundation
import simd

/// Layout matches `Sphere` in PathTracer.metal.
struct GPUSphere {
    var centerRadius: SIMD4<Float>
    var prevCenter: SIMD4<Float>
    var absorption: SIMD4<Float>
}

/// CPU marble simulation: spheres rolling through a channel, over a dish and down a drain.
/// Drained marbles queue up and drop back in at the top of the run.
final class MarbleSimulation {
    let count: Int
    private let path: [SIMD3<Float>]
    private let segmentDirections: [SIMD3<Float>]
    private let segmentLengths: [Float]

    private(set) var positions: [SIMD3<Float>]
    private var previousFramePositions: [SIMD3<Float>]
    private var velocities: [SIMD3<Float>]
    private(set) var radii: [Float]
    private var absorption: [SIMD4<Float>]
    /// Current channel segment, or -1 when the marble is free.
    private var segments: [Int32]
    private var active: [Bool]
    private var justSpawned: [Bool]
    private var waiting: [Int] = []
    private var spawnCooldown: Float = 0

    // Uniform grid for neighbour search.
    private let cellSize: Float = 0.3
    private let bucketCount: Int
    private var bucketStart: [Int32]
    private var bucketItems: [Int32]

    private static let parkedPosition = SIMD3<Float>(0, -60, 0)

    init(count: Int, path: TrackPath) {
        self.count = count
        self.path = path.points
        var directions: [SIMD3<Float>] = []
        var lengths: [Float] = []
        for i in 0..<(path.points.count - 1) {
            let d = path.points[i + 1] - path.points[i]
            let l = simd_length(d)
            directions.append(d / l)
            lengths.append(l)
        }
        segmentDirections = directions
        segmentLengths = lengths

        positions = Array(repeating: Self.parkedPosition, count: count)
        previousFramePositions = positions
        velocities = Array(repeating: .zero, count: count)
        radii = Array(repeating: 0.09, count: count)
        absorption = Array(repeating: .zero, count: count)
        segments = Array(repeating: -1, count: count)
        active = Array(repeating: false, count: count)
        justSpawned = Array(repeating: false, count: count)
        bucketCount = max(4096, count * 4)
        bucketStart = Array(repeating: 0, count: bucketCount + 1)
        bucketItems = Array(repeating: 0, count: count)

        // Colour of light after passing through the width of one marble.
        let palette: [SIMD3<Float>] = [
            SIMD3(0.99, 0.995, 0.995),  // clear
            SIMD3(0.99, 0.86, 0.62),  // amber
            SIMD3(0.98, 0.62, 0.62),  // ruby
            SIMD3(0.62, 0.95, 0.72),  // emerald
            SIMD3(0.62, 0.74, 0.99),  // cobalt
            SIMD3(0.99, 0.995, 0.995),  // clear again, because clear glass sells the effect
            SIMD3(0.84, 0.66, 0.98),  // violet
            SIMD3(0.62, 0.93, 0.95),  // teal
        ]
        var random = SystemRandomNumberGenerator()
        for i in 0..<count {
            let radius = Float.random(in: 0.1...0.14, using: &random)
            radii[i] = radius
            let color = palette[i % palette.count]
            let perUnit = SIMD3<Float>(-log(color.x), -log(color.y), -log(color.z)) / (2 * 0.12)
            absorption[i] = SIMD4(perUnit, 1.5)
        }

        // Start with some marbles in the channel and the rest spread out over the dish,
        // so the first frames are representative. A few seconds of settling follow.
        let inChannel = min(count * 2 / 5, 100)
        let firstHelix = path.firstVisible + 30
        let usable = self.path.count - firstHelix - 2
        for i in 0..<inChannel {
            let segment = firstHelix + usable * i / max(inChannel, 1)
            positions[i] = self.path[segment]
            segments[i] = Int32(segment)
            active[i] = true
        }
        var ringRadius: Float = 1.0
        var angle: Float = 0
        var layerHeight: Float = 0.25
        for i in inChannel..<max(count, inChannel) {
            positions[i] = SIMD3(cos(angle) * ringRadius, layerHeight, sin(angle) * ringRadius)
            active[i] = true
            angle += 0.3 / ringRadius
            if angle > 2 * .pi {
                angle = 0
                ringRadius += 0.3
                if ringRadius > Layout.dishRadius - 0.15 {
                    ringRadius = 1.0
                    layerHeight += 0.3
                }
            }
        }
        previousFramePositions = positions
        for _ in 0..<(3 * 120) { substep(1.0 / 120.0) }
        beginFrame()
    }

    /// Call before stepping a frame; motion vectors are measured from this point.
    func beginFrame() {
        previousFramePositions = positions
        for i in 0..<count { justSpawned[i] = false }
    }

    func step(dt: Float) {
        let substeps = 2
        let clamped = min(dt, 1.0 / 30.0)
        for _ in 0..<substeps { substep(clamped / Float(substeps)) }
    }

    func writeSpheres(to spheres: UnsafeMutablePointer<GPUSphere>,
                      boxes: UnsafeMutablePointer<MTLBox>) {
        for i in 0..<count {
            let p = positions[i]
            let r = radii[i]
            // A teleported marble has no meaningful previous position.
            let prev = justSpawned[i] ? p : previousFramePositions[i]
            spheres[i] = GPUSphere(centerRadius: SIMD4(p, r), prevCenter: SIMD4(prev, 0), absorption: absorption[i])
            boxes[i] = MTLBox(minX: p.x - r, minY: p.y - r, minZ: p.z - r,
                              maxX: p.x + r, maxY: p.y + r, maxZ: p.z + r)
        }
    }

    // MARK: - Simulation

    private func substep(_ dt: Float) {
        let gravity = SIMD3<Float>(0, -9.8, 0)
        for i in 0..<count where active[i] {
            // 5/7: a rolling sphere accelerates slower than a sliding one.
            velocities[i] += gravity * dt * (segments[i] >= 0 ? 0.71 : 1)
            let drag: Float = segments[i] >= 0 ? 0.55 : 0.28
            velocities[i] *= max(0, 1 - drag * dt)
            positions[i] += velocities[i] * dt
        }
        solveMarbleContacts()
        for i in 0..<count where active[i] {
            if segments[i] >= 0 { solveChannel(i) } else { solveDish(i) }
        }
        respawnWaiting(dt)
    }

    /// Keeps a marble inside the channel, treated as a closed tube for robustness.
    private func solveChannel(_ i: Int) {
        var segment = Int(segments[i])
        let p = positions[i]
        // Marbles move slowly relative to segment length, so a local search is enough.
        var bestDistance = Float.greatestFiniteMagnitude
        var bestSegment = segment
        var bestPoint = p
        var bestAlong: Float = 0
        let lower = max(segment - 3, 0)
        let upper = min(segment + 6, segmentDirections.count - 1)
        for s in lower...upper {
            let along = simd_clamp(simd_dot(p - path[s], segmentDirections[s]), 0, segmentLengths[s])
            let q = path[s] + segmentDirections[s] * along
            let distance = simd_length_squared(p - q)
            if distance < bestDistance {
                bestDistance = distance
                bestSegment = s
                bestPoint = q
                bestAlong = along
            }
        }
        segment = bestSegment
        segments[i] = Int32(segment)
        if segment >= segmentDirections.count - 1 && bestAlong >= segmentLengths[segment] - 1e-3 {
            segments[i] = -1  // rolled off the end of the channel
            return
        }
        let offset = p - bestPoint
        let distance = simd_length(offset)
        let limit = Layout.channelRadius - radii[i]
        if distance > limit && distance > 1e-5 {
            let outward = offset / distance
            positions[i] = bestPoint + outward * limit
            let outwardSpeed = simd_dot(velocities[i], outward)
            if outwardSpeed > 0 { velocities[i] -= outward * outwardSpeed * 1.08 }
        }
    }

    private func solveDish(_ i: Int) {
        var p = positions[i]
        var v = velocities[i]
        let r = radii[i]
        let radial = SIMD2<Float>(p.x, p.z)
        var distance = simd_length(radial)
        if distance < 1e-4 { distance = 1e-4 }
        let direction = radial / distance
        var newDistance = distance

        if distance > Layout.holeRadius {
            // Over the funnel.
            let height = Layout.dishHeight(atRadius: distance) + r
            if p.y < height {
                p.y = height
                let normal = simd_normalize(SIMD3<Float>(-Layout.dishSlope * direction.x, 1, -Layout.dishSlope * direction.y))
                let into = simd_dot(v, normal)
                if into < 0 { v -= normal * into * 1.2 }
                v *= 0.9994  // rolling resistance
            }
            let wall = Layout.dishRadius - r
            if newDistance > wall {
                newDistance = wall
                let out = simd_dot(SIMD2(v.x, v.z), direction)
                if out > 0 {
                    v.x -= direction.x * out * 1.4
                    v.z -= direction.y * out * 1.4
                }
            }
        } else if p.y < 0 {
            // In the drain shaft: ease towards the shaft so the marble drops in smoothly.
            let outerLimit = Layout.holeRadius - r
            if newDistance > outerLimit { newDistance += (outerLimit - newDistance) * 0.2 }
        }

        // Central column.
        let inner = Layout.columnRadius + r
        if newDistance < inner && p.y < Layout.columnHeight {
            newDistance = inner
            let out = simd_dot(SIMD2(v.x, v.z), direction)
            if out < 0 {
                v.x -= direction.x * out * 1.3
                v.z -= direction.y * out * 1.3
            }
        }
        if newDistance != distance {
            p.x = direction.x * newDistance
            p.z = direction.y * newDistance
        }

        if p.y < -0.6 {
            park(i)
            return
        }
        positions[i] = p
        velocities[i] = v
    }

    private func park(_ i: Int) {
        active[i] = false
        positions[i] = Self.parkedPosition
        velocities[i] = .zero
        segments[i] = -1
        waiting.append(i)
    }

    private func respawnWaiting(_ dt: Float) {
        spawnCooldown -= dt
        guard spawnCooldown <= 0, let i = waiting.first else { return }
        waiting.removeFirst()
        positions[i] = path[0]
        velocities[i] = SIMD3(0, -1.5, 0)
        segments[i] = 0
        active[i] = true
        justSpawned[i] = true
        // Steady stream: limit how fast the channel is fed so marbles stay spaced out.
        spawnCooldown = 0.09
    }

    private func bucket(_ x: Int, _ y: Int, _ z: Int) -> Int {
        let h = (x &* 73856093) ^ (y &* 19349663) ^ (z &* 83492791)
        return Int(UInt(bitPattern: h) % UInt(bucketCount))
    }

    private func solveMarbleContacts() {
        let inverseCell = 1 / cellSize
        positions.withUnsafeMutableBufferPointer { positions in
        velocities.withUnsafeMutableBufferPointer { velocities in
        bucketStart.withUnsafeMutableBufferPointer { starts in
        bucketItems.withUnsafeMutableBufferPointer { items in
            // Counting sort of the marbles into grid buckets.
            for b in 0...bucketCount { starts[b] = 0 }
            for i in 0..<count where active[i] {
                let p = positions[i] * inverseCell
                starts[bucket(Int(floor(p.x)), Int(floor(p.y)), Int(floor(p.z))) + 1] += 1
            }
            for b in 0..<bucketCount { starts[b + 1] += starts[b] }
            for i in 0..<count where active[i] {
                let p = positions[i] * inverseCell
                let b = bucket(Int(floor(p.x)), Int(floor(p.y)), Int(floor(p.z)))
                items[Int(starts[b])] = Int32(i)
                starts[b] += 1
            }
            // The fill above shifted each start to the next bucket's start.
            for b in stride(from: bucketCount, to: 0, by: -1) { starts[b] = starts[b - 1] }
            starts[0] = 0

            for i in 0..<count where active[i] {
                let cell = positions[i] * inverseCell
                let cx = Int(floor(cell.x)), cy = Int(floor(cell.y)), cz = Int(floor(cell.z))
                for dz in -1...1 { for dy in -1...1 { for dx in -1...1 {
                    let b = bucket(cx + dx, cy + dy, cz + dz)
                    var k = Int(starts[b])
                    let end = Int(starts[b + 1])
                    while k < end {
                        let j = Int(items[k])
                        k += 1
                        if j <= i { continue }
                        let delta = positions[j] - positions[i]
                        let minDistance = radii[i] + radii[j]
                        let distance2 = simd_length_squared(delta)
                        if distance2 >= minDistance * minDistance || distance2 < 1e-10 { continue }
                        let distance = sqrt(distance2)
                        let normal = delta / distance
                        let correction = (minDistance - distance) * 0.5
                        positions[i] -= normal * correction
                        positions[j] += normal * correction
                        let closing = simd_dot(velocities[j] - velocities[i], normal)
                        if closing < 0 {
                            let impulse = normal * closing * 0.65  // restitution 0.3, equal masses
                            velocities[i] += impulse
                            velocities[j] -= impulse
                        }
                    }
                }}}
            }
        }}}}
    }
}

/// Bounding box with the same layout as MTLAxisAlignedBoundingBox.
struct MTLBox {
    var minX: Float, minY: Float, minZ: Float
    var maxX: Float, maxY: Float, maxZ: Float
}
