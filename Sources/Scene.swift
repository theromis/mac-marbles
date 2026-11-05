import Foundation
import simd

/// Interleaved triangle soup. Layout matches `Vertex` in PathTracer.metal.
struct Vertex {
    var position: SIMD4<Float>  // w = material id
    var normal: SIMD4<Float>
}

enum Material: Float {
    case floor = 0, dish, track, post, dark, paper
}

/// Shared dimensions of the marble run, used by both the mesh and the physics.
enum Layout {
    static let helixRadius: Float = 1.35
    static let helixTurns: Float = 3.2
    static let helixTop: Float = 2.75
    static let helixBottom: Float = 0.55
    static let channelRadius: Float = 0.2
    static let channelWall: Float = 0.025
    static let columnRadius: Float = 0.28
    static let columnHeight: Float = 3.05
    static let holeRadius: Float = 0.68
    static let dishRadius: Float = 3.0
    static let dishSlope: Float = 0.05
    static let exitRadius: Float = 2.3

    // Newspaper lying in the dish, on the path marbles take from the channel exit to the drain.
    static let paperCenter = SIMD2<Float>(-1.19, 1.12)
    static let paperHalfWidth: Float = 0.92
    static let paperHalfHeight: Float = 0.64
    /// Reading direction of the sheet, chosen so the text faces the camera's side of the scene.
    static let paperAcross = SIMD2<Float>(0.5646, -0.8253)
    static var paperUp: SIMD2<Float> { SIMD2(-paperAcross.y, paperAcross.x) * -1 }

    /// Height of the shallow funnel that rolls marbles towards the drain hole.
    static func dishHeight(atRadius r: Float) -> Float { dishSlope * (r - holeRadius) }
}

/// Centre line of the marble channel, from off-screen down to the dish.
struct TrackPath {
    var points: [SIMD3<Float>] = []
    /// Index of the first point that has visible channel geometry.
    var firstVisible = 0

    init() {
        let segmentCount = 520
        let totalAngle = Layout.helixTurns * 2 * .pi
        func helix(_ t: Float) -> SIMD3<Float> {
            let a = t * totalAngle
            return SIMD3(Layout.helixRadius * cos(a),
                         Layout.helixTop + (Layout.helixBottom - Layout.helixTop) * t,
                         Layout.helixRadius * sin(a))
        }
        let start = helix(0)
        let startTangent = simd_normalize(helix(0.001) - start)
        let up = SIMD3<Float>(0, 1, 0)
        let lipRadius: Float = 0.45

        // Invisible vertical guide: marbles appear above the frame and drop in.
        let leadTop = start - startTangent * lipRadius + up * (lipRadius + 3.5)
        let leadBottom = start - startTangent * lipRadius + up * lipRadius
        for i in 0..<30 {
            points.append(simd_mix(leadTop, leadBottom, SIMD3(repeating: Float(i) / 30)))
        }
        // Quarter circle from falling straight down to following the helix.
        for i in 0..<24 {
            let a = Float(i) / 24 * .pi / 2
            points.append(start - startTangent * lipRadius * cos(a) + up * lipRadius * (1 - sin(a)))
        }
        for i in 0...segmentCount {
            points.append(helix(Float(i) / Float(segmentCount)))
        }
        // Straight run out over the dish.
        let end = helix(1)
        let endTangent = simd_normalize(SIMD3(helix(1).x - helix(0.999).x, 0, helix(1).z - helix(0.999).z))
        let alongToExit = sqrt(Layout.exitRadius * Layout.exitRadius - Layout.helixRadius * Layout.helixRadius)
        let exitHeight = Layout.dishHeight(atRadius: Layout.exitRadius) + Layout.channelRadius + 0.03
        for i in 1...40 {
            let t = Float(i) / 40
            var p = end + endTangent * (alongToExit * t)
            p.y = Layout.helixBottom + (exitHeight - Layout.helixBottom) * t
            points.append(p)
        }
        firstVisible = 30 + 6
    }

    /// Angle (radians) at which the helix meets the straight exit, for avoiding supports.
    var exitAngle: Float { Layout.helixTurns * 2 * .pi }

    func helixHeight(atAngle angle: Float, turn: Int) -> Float {
        // Finds the height of the helix for a given polar angle on a given turn.
        let total = Layout.helixTurns * 2 * Float.pi
        let unwrapped = angle + Float(turn) * 2 * .pi
        let t = unwrapped / total
        return Layout.helixTop + (Layout.helixBottom - Layout.helixTop) * t
    }
}

struct MeshBuilder {
    private(set) var vertices: [Vertex] = []

    mutating func triangle(_ a: SIMD3<Float>, _ na: SIMD3<Float>,
                           _ b: SIMD3<Float>, _ nb: SIMD3<Float>,
                           _ c: SIMD3<Float>, _ nc: SIMD3<Float>, _ material: Material) {
        let m = material.rawValue
        vertices.append(Vertex(position: SIMD4(a, m), normal: SIMD4(na, 0)))
        vertices.append(Vertex(position: SIMD4(b, m), normal: SIMD4(nb, 0)))
        vertices.append(Vertex(position: SIMD4(c, m), normal: SIMD4(nc, 0)))
    }

    mutating func quad(_ a: SIMD3<Float>, _ na: SIMD3<Float>, _ b: SIMD3<Float>, _ nb: SIMD3<Float>,
                       _ c: SIMD3<Float>, _ nc: SIMD3<Float>, _ d: SIMD3<Float>, _ nd: SIMD3<Float>,
                       _ material: Material) {
        triangle(a, na, b, nb, c, nc, material)
        triangle(a, na, c, nc, d, nd, material)
    }

    /// Solid-of-revolution band around the Y axis between two (radius, height) points.
    mutating func revolve(from a: (r: Float, y: Float), to b: (r: Float, y: Float),
                          normal: (r: Float, y: Float), material: Material, steps: Int = 128) {
        for i in 0..<steps {
            let a0 = Float(i) / Float(steps) * 2 * .pi
            let a1 = Float(i + 1) / Float(steps) * 2 * .pi
            let d0 = SIMD3<Float>(cos(a0), 0, sin(a0))
            let d1 = SIMD3<Float>(cos(a1), 0, sin(a1))
            func point(_ d: SIMD3<Float>, _ s: (r: Float, y: Float)) -> SIMD3<Float> {
                SIMD3(d.x * s.r, s.y, d.z * s.r)
            }
            let n0 = simd_normalize(SIMD3(d0.x * normal.r, normal.y, d0.z * normal.r))
            let n1 = simd_normalize(SIMD3(d1.x * normal.r, normal.y, d1.z * normal.r))
            quad(point(d0, a), n0, point(d1, a), n1, point(d1, b), n1, point(d0, b), n0, material)
        }
    }

    mutating func cylinder(from a: SIMD3<Float>, to b: SIMD3<Float>, radius: Float,
                           material: Material, sides: Int = 14) {
        let axis = simd_normalize(b - a)
        let helper = abs(axis.y) > 0.9 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0)
        let u = simd_normalize(simd_cross(axis, helper))
        let v = simd_cross(axis, u)
        for i in 0..<sides {
            let a0 = Float(i) / Float(sides) * 2 * .pi
            let a1 = Float(i + 1) / Float(sides) * 2 * .pi
            let n0 = u * cos(a0) + v * sin(a0)
            let n1 = u * cos(a1) + v * sin(a1)
            quad(a + n0 * radius, n0, a + n1 * radius, n1, b + n1 * radius, n1, b + n0 * radius, n0, material)
            triangle(b, axis, b + n0 * radius, axis, b + n1 * radius, axis, material)
        }
    }

    /// Open U-shaped wooden channel swept along the track's centre line.
    mutating func channel(along path: TrackPath) {
        let pieces = 16
        let halfArc: Float = 100 * .pi / 180
        let up = SIMD3<Float>(0, 1, 0)
        let inner = Layout.channelRadius
        let outer = inner + Layout.channelWall

        struct Ring {
            var inner: [SIMD3<Float>]
            var outer: [SIMD3<Float>]
            var normals: [SIMD3<Float>]
        }
        var rings: [Ring] = []
        let points = path.points
        for i in path.firstVisible..<points.count {
            let tangent = simd_normalize(points[min(i + 1, points.count - 1)] - points[max(i - 1, 0)])
            let side = simd_normalize(simd_cross(tangent, up))
            let localUp = simd_cross(side, tangent)
            var ring = Ring(inner: [], outer: [], normals: [])
            for p in 0...pieces {
                let angle = -halfArc + 2 * halfArc * Float(p) / Float(pieces)
                let radial = side * sin(angle) - localUp * cos(angle)
                ring.inner.append(points[i] + radial * inner)
                ring.outer.append(points[i] + radial * outer)
                ring.normals.append(radial)
            }
            rings.append(ring)
        }
        for i in 0..<(rings.count - 1) {
            let a = rings[i]
            let b = rings[i + 1]
            for p in 0..<pieces {
                quad(a.inner[p], -a.normals[p], b.inner[p], -b.normals[p],
                     b.inner[p + 1], -b.normals[p + 1], a.inner[p + 1], -a.normals[p + 1], .track)
                quad(a.outer[p], a.normals[p], a.outer[p + 1], a.normals[p + 1],
                     b.outer[p + 1], b.normals[p + 1], b.outer[p], b.normals[p], .track)
            }
            // Flat tops of the channel walls.
            for edge in [0, pieces] {
                let n = simd_normalize(simd_cross(a.outer[edge] - a.inner[edge], b.inner[edge] - a.inner[edge]))
                let flat = simd_dot(n, up) < 0 ? -n : n
                quad(a.inner[edge], flat, a.outer[edge], flat, b.outer[edge], flat, b.inner[edge], flat, .track)
            }
        }
    }
}

/// Builds the static wooden scene: table, dish with a drain hole, column, supports, channel.
func buildSceneMesh(path: TrackPath) -> [Vertex] {
    var mesh = MeshBuilder()
    let up: (r: Float, y: Float) = (0, 1)
    let dish = Layout.dishHeight

    // Table.
    mesh.revolve(from: (Layout.dishRadius + 0.1, -0.02), to: (42, -0.02), normal: up, material: .floor)

    // Dish: funnel, inner rim wall, top of the rim, outer wall.
    let slopeNormal: (r: Float, y: Float) = (-Layout.dishSlope, 1)
    mesh.revolve(from: (Layout.holeRadius, dish(Layout.holeRadius)),
                 to: (Layout.dishRadius, dish(Layout.dishRadius)), normal: slopeNormal, material: .dish)
    let rimTop = dish(Layout.dishRadius) + 0.24
    mesh.revolve(from: (Layout.dishRadius, dish(Layout.dishRadius)), to: (Layout.dishRadius, rimTop),
                 normal: (-1, 0), material: .dish)
    mesh.revolve(from: (Layout.dishRadius, rimTop), to: (Layout.dishRadius + 0.1, rimTop),
                 normal: up, material: .dish)
    mesh.revolve(from: (Layout.dishRadius + 0.1, rimTop), to: (Layout.dishRadius + 0.1, -0.02),
                 normal: (1, 0), material: .dish)

    // Drain: dark shaft below the hole.
    mesh.revolve(from: (Layout.holeRadius, 0), to: (Layout.holeRadius, -0.9), normal: (-1, 0), material: .dark)
    mesh.revolve(from: (0, -0.9), to: (Layout.holeRadius, -0.9), normal: up, material: .dark)

    // Central column passes through the middle of the drain.
    mesh.cylinder(from: SIMD3(0, -0.9, 0), to: SIMD3(0, Layout.columnHeight, 0),
                  radius: Layout.columnRadius, material: .post, sides: 32)

    // Outer posts and the spokes that carry each turn of the helix.
    let postRadius = Layout.helixRadius + Layout.channelRadius + Layout.channelWall + 0.035
    let postCount = 8
    let exit = path.exitAngle.truncatingRemainder(dividingBy: 2 * .pi)
    let wholeTurns = Int(Layout.helixTurns.rounded(.up))
    for i in 0..<postCount {
        let angle = Float(i) / Float(postCount) * 2 * .pi
        // The straight exit leaves the helix here; a post would block it.
        var offset = (angle - exit).truncatingRemainder(dividingBy: 2 * .pi)
        if offset < 0 { offset += 2 * .pi }
        let blocksExit = offset < 0.95
        var highest = Float(0)
        for turn in 0..<wholeTurns {
            let height = path.helixHeight(atAngle: angle, turn: turn)
            guard height >= Layout.helixBottom && height <= Layout.helixTop else { continue }
            highest = max(highest, height)
            let y = height - Layout.channelRadius - Layout.channelWall - 0.03
            let direction = SIMD3<Float>(cos(angle), 0, sin(angle))
            mesh.cylinder(from: SIMD3(0, y, 0) + direction * Layout.columnRadius,
                          to: SIMD3(0, y, 0) + direction * postRadius,
                          radius: 0.026, material: .post, sides: 10)
        }
        if !blocksExit {
            let base = SIMD3<Float>(cos(angle) * postRadius, dish(postRadius), sin(angle) * postRadius)
            mesh.cylinder(from: base, to: SIMD3(base.x, highest + 0.03, base.z),
                          radius: 0.034, material: .post, sides: 12)
        }
    }

    // Newspaper: a fine grid, because it has to follow the funnel's slope.
    let cells = 32
    func paperPoint(_ i: Int, _ j: Int) -> (SIMD3<Float>, SIMD3<Float>) {
        let across = (Float(i) / Float(cells) * 2 - 1) * Layout.paperHalfWidth
        let along = (Float(j) / Float(cells) * 2 - 1) * Layout.paperHalfHeight
        let xz = Layout.paperCenter + Layout.paperAcross * across + Layout.paperUp * along
        let r = simd_length(xz)
        let direction = xz / r
        let normal = simd_normalize(SIMD3(-Layout.dishSlope * direction.x, 1, -Layout.dishSlope * direction.y))
        return (SIMD3(xz.x, dish(r) + 0.003, xz.y), normal)
    }
    for i in 0..<cells {
        for j in 0..<cells {
            let a = paperPoint(i, j), b = paperPoint(i + 1, j), c = paperPoint(i + 1, j + 1), d = paperPoint(i, j + 1)
            mesh.quad(a.0, a.1, b.0, b.1, c.0, c.1, d.0, d.1, .paper)
        }
    }

    mesh.channel(along: path)
    return mesh.vertices
}
