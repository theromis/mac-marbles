//! Static geometry of the marble run. Shared dimensions live in `layout`, because the mesh
//! and the physics have to agree on them.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::f32::consts::{PI, TAU};

/// Interleaved triangle soup. Layout matches `Vertex` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    /// w = material id
    pub position: [f32; 4],
    pub normal: [f32; 4],
}

#[derive(Clone, Copy)]
pub enum Material {
    Floor = 0,
    Dish,
    Track,
    Post,
    Dark,
    Paper,
}

pub mod layout {
    use glam::Vec2;

    pub const HELIX_RADIUS: f32 = 1.35;
    pub const HELIX_TURNS: f32 = 3.2;
    pub const HELIX_TOP: f32 = 2.75;
    pub const HELIX_BOTTOM: f32 = 0.55;
    pub const CHANNEL_RADIUS: f32 = 0.2;
    pub const CHANNEL_WALL: f32 = 0.025;
    pub const COLUMN_RADIUS: f32 = 0.28;
    pub const COLUMN_HEIGHT: f32 = 3.05;
    pub const HOLE_RADIUS: f32 = 0.68;
    pub const DISH_RADIUS: f32 = 3.0;
    pub const DISH_SLOPE: f32 = 0.05;
    pub const EXIT_RADIUS: f32 = 2.3;

    // Newspaper in the dish, on the path marbles take from the channel exit to the drain.
    pub const PAPER_CENTER: Vec2 = Vec2::new(-1.19, 1.12);
    pub const PAPER_HALF_WIDTH: f32 = 0.92;
    pub const PAPER_HALF_HEIGHT: f32 = 0.64;
    /// Reading direction of the sheet, chosen so the text faces the camera's side of the scene.
    pub const PAPER_ACROSS: Vec2 = Vec2::new(0.5646, -0.8253);
    pub const PAPER_UP: Vec2 = Vec2::new(PAPER_ACROSS.y, -PAPER_ACROSS.x);

    /// Height of the shallow funnel that rolls marbles towards the drain hole.
    pub fn dish_height(radius: f32) -> f32 {
        DISH_SLOPE * (radius - HOLE_RADIUS)
    }
}

/// Centre line of the marble channel, from off-screen down to the dish.
pub struct TrackPath {
    pub points: Vec<Vec3>,
    /// Index of the first point that has visible channel geometry.
    pub first_visible: usize,
}

impl Default for TrackPath {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackPath {
    pub fn new() -> Self {
        use layout::*;
        let segment_count = 520;
        let total_angle = HELIX_TURNS * TAU;
        let helix = |t: f32| {
            let a = t * total_angle;
            Vec3::new(HELIX_RADIUS * a.cos(), HELIX_TOP + (HELIX_BOTTOM - HELIX_TOP) * t, HELIX_RADIUS * a.sin())
        };
        let start = helix(0.0);
        let start_tangent = (helix(0.001) - start).normalize();
        let lip_radius = 0.45;
        let mut points = Vec::new();

        // Invisible vertical guide: marbles appear above the frame and drop in.
        let lead_top = start - start_tangent * lip_radius + Vec3::Y * (lip_radius + 3.5);
        let lead_bottom = start - start_tangent * lip_radius + Vec3::Y * lip_radius;
        for i in 0..30 {
            points.push(lead_top.lerp(lead_bottom, i as f32 / 30.0));
        }
        // Quarter circle from falling straight down to following the helix.
        for i in 0..24 {
            let a = i as f32 / 24.0 * PI / 2.0;
            points.push(start - start_tangent * lip_radius * a.cos() + Vec3::Y * lip_radius * (1.0 - a.sin()));
        }
        for i in 0..=segment_count {
            points.push(helix(i as f32 / segment_count as f32));
        }
        // Straight run out over the dish.
        let end = helix(1.0);
        let before = helix(0.999);
        let end_tangent = Vec3::new(end.x - before.x, 0.0, end.z - before.z).normalize();
        let along_to_exit = (EXIT_RADIUS * EXIT_RADIUS - HELIX_RADIUS * HELIX_RADIUS).sqrt();
        let exit_height = dish_height(EXIT_RADIUS) + CHANNEL_RADIUS + 0.03;
        for i in 1..=40 {
            let t = i as f32 / 40.0;
            let mut p = end + end_tangent * (along_to_exit * t);
            p.y = HELIX_BOTTOM + (exit_height - HELIX_BOTTOM) * t;
            points.push(p);
        }
        Self { points, first_visible: 36 }
    }

    /// Angle at which the helix meets the straight exit, for keeping supports out of the way.
    fn exit_angle(&self) -> f32 {
        layout::HELIX_TURNS * TAU
    }

    fn helix_height(&self, angle: f32, turn: i32) -> f32 {
        use layout::*;
        let t = (angle + turn as f32 * TAU) / (HELIX_TURNS * TAU);
        HELIX_TOP + (HELIX_BOTTOM - HELIX_TOP) * t
    }
}

#[derive(Default)]
struct MeshBuilder {
    vertices: Vec<Vertex>,
}

impl MeshBuilder {
    fn vertex(&mut self, p: Vec3, n: Vec3, m: Material) {
        self.vertices.push(Vertex { position: [p.x, p.y, p.z, m as i32 as f32], normal: [n.x, n.y, n.z, 0.0] });
    }

    fn triangle(&mut self, a: (Vec3, Vec3), b: (Vec3, Vec3), c: (Vec3, Vec3), m: Material) {
        self.vertex(a.0, a.1, m);
        self.vertex(b.0, b.1, m);
        self.vertex(c.0, c.1, m);
    }

    fn quad(&mut self, a: (Vec3, Vec3), b: (Vec3, Vec3), c: (Vec3, Vec3), d: (Vec3, Vec3), m: Material) {
        self.triangle(a, b, c, m);
        self.triangle(a, c, d, m);
    }

    /// Solid-of-revolution band around the Y axis between two (radius, height) points.
    fn revolve(&mut self, a: (f32, f32), b: (f32, f32), normal: (f32, f32), m: Material) {
        let steps = 128;
        for i in 0..steps {
            let a0 = i as f32 / steps as f32 * TAU;
            let a1 = (i + 1) as f32 / steps as f32 * TAU;
            let d0 = Vec3::new(a0.cos(), 0.0, a0.sin());
            let d1 = Vec3::new(a1.cos(), 0.0, a1.sin());
            let point = |d: Vec3, s: (f32, f32)| Vec3::new(d.x * s.0, s.1, d.z * s.0);
            let n0 = Vec3::new(d0.x * normal.0, normal.1, d0.z * normal.0).normalize();
            let n1 = Vec3::new(d1.x * normal.0, normal.1, d1.z * normal.0).normalize();
            self.quad((point(d0, a), n0), (point(d1, a), n1), (point(d1, b), n1), (point(d0, b), n0), m);
        }
    }

    fn cylinder(&mut self, a: Vec3, b: Vec3, radius: f32, m: Material, sides: usize) {
        let axis = (b - a).normalize();
        let helper = if axis.y.abs() > 0.9 { Vec3::X } else { Vec3::Y };
        let u = axis.cross(helper).normalize();
        let v = axis.cross(u);
        for i in 0..sides {
            let a0 = i as f32 / sides as f32 * TAU;
            let a1 = (i + 1) as f32 / sides as f32 * TAU;
            let n0 = u * a0.cos() + v * a0.sin();
            let n1 = u * a1.cos() + v * a1.sin();
            self.quad(
                (a + n0 * radius, n0),
                (a + n1 * radius, n1),
                (b + n1 * radius, n1),
                (b + n0 * radius, n0),
                m,
            );
            self.triangle((b, axis), (b + n0 * radius, axis), (b + n1 * radius, axis), m);
        }
    }

    /// Open U-shaped wooden channel swept along the track's centre line.
    fn channel(&mut self, path: &TrackPath) {
        use layout::*;
        struct Ring {
            inner: Vec<Vec3>,
            outer: Vec<Vec3>,
            normals: Vec<Vec3>,
        }
        let pieces = 16;
        let half_arc = 100.0_f32.to_radians();
        let inner = CHANNEL_RADIUS;
        let outer = inner + CHANNEL_WALL;
        let points = &path.points;
        let mut rings = Vec::new();
        for i in path.first_visible..points.len() {
            let tangent = (points[(i + 1).min(points.len() - 1)] - points[i.saturating_sub(1)]).normalize();
            let side = tangent.cross(Vec3::Y).normalize();
            let local_up = side.cross(tangent);
            let mut ring = Ring { inner: vec![], outer: vec![], normals: vec![] };
            for p in 0..=pieces {
                let angle = -half_arc + 2.0 * half_arc * p as f32 / pieces as f32;
                let radial = side * angle.sin() - local_up * angle.cos();
                ring.inner.push(points[i] + radial * inner);
                ring.outer.push(points[i] + radial * outer);
                ring.normals.push(radial);
            }
            rings.push(ring);
        }
        for pair in rings.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            for p in 0..pieces {
                self.quad(
                    (a.inner[p], -a.normals[p]),
                    (b.inner[p], -b.normals[p]),
                    (b.inner[p + 1], -b.normals[p + 1]),
                    (a.inner[p + 1], -a.normals[p + 1]),
                    Material::Track,
                );
                self.quad(
                    (a.outer[p], a.normals[p]),
                    (a.outer[p + 1], a.normals[p + 1]),
                    (b.outer[p + 1], b.normals[p + 1]),
                    (b.outer[p], b.normals[p]),
                    Material::Track,
                );
            }
            // Flat tops of the channel walls.
            for edge in [0, pieces] {
                let n = (a.outer[edge] - a.inner[edge]).cross(b.inner[edge] - a.inner[edge]).normalize();
                let flat = if n.dot(Vec3::Y) < 0.0 { -n } else { n };
                self.quad(
                    (a.inner[edge], flat),
                    (a.outer[edge], flat),
                    (b.outer[edge], flat),
                    (b.inner[edge], flat),
                    Material::Track,
                );
            }
        }
    }
}

/// Builds the static scene: table, dish with a drain hole, column, supports, newspaper, channel.
pub fn build_scene_mesh(path: &TrackPath) -> Vec<Vertex> {
    use layout::*;
    let mut mesh = MeshBuilder::default();
    let up = (0.0, 1.0);
    let dish = dish_height;

    // Table.
    mesh.revolve((DISH_RADIUS + 0.1, -0.02), (42.0, -0.02), up, Material::Floor);

    // Dish: funnel, inner rim wall, top of the rim, outer wall.
    mesh.revolve((HOLE_RADIUS, dish(HOLE_RADIUS)), (DISH_RADIUS, dish(DISH_RADIUS)), (-DISH_SLOPE, 1.0), Material::Dish);
    let rim_top = dish(DISH_RADIUS) + 0.24;
    mesh.revolve((DISH_RADIUS, dish(DISH_RADIUS)), (DISH_RADIUS, rim_top), (-1.0, 0.0), Material::Dish);
    mesh.revolve((DISH_RADIUS, rim_top), (DISH_RADIUS + 0.1, rim_top), up, Material::Dish);
    mesh.revolve((DISH_RADIUS + 0.1, rim_top), (DISH_RADIUS + 0.1, -0.02), (1.0, 0.0), Material::Dish);

    // Drain: dark shaft below the hole.
    mesh.revolve((HOLE_RADIUS, 0.0), (HOLE_RADIUS, -0.9), (-1.0, 0.0), Material::Dark);
    mesh.revolve((0.0, -0.9), (HOLE_RADIUS, -0.9), up, Material::Dark);

    // Central column passes through the middle of the drain.
    mesh.cylinder(Vec3::new(0.0, -0.9, 0.0), Vec3::new(0.0, COLUMN_HEIGHT, 0.0), COLUMN_RADIUS, Material::Post, 32);

    // Outer posts and the spokes that carry each turn of the helix.
    let post_radius = HELIX_RADIUS + CHANNEL_RADIUS + CHANNEL_WALL + 0.035;
    let post_count = 8;
    let exit = path.exit_angle().rem_euclid(TAU);
    let whole_turns = HELIX_TURNS.ceil() as i32;
    for i in 0..post_count {
        let angle = i as f32 / post_count as f32 * TAU;
        // The straight exit leaves the helix here; a post would block it.
        let blocks_exit = (angle - exit).rem_euclid(TAU) < 0.95;
        let mut highest = 0.0_f32;
        let direction = Vec3::new(angle.cos(), 0.0, angle.sin());
        for turn in 0..whole_turns {
            let height = path.helix_height(angle, turn);
            if !(HELIX_BOTTOM..=HELIX_TOP).contains(&height) {
                continue;
            }
            highest = highest.max(height);
            let y = Vec3::new(0.0, height - CHANNEL_RADIUS - CHANNEL_WALL - 0.03, 0.0);
            mesh.cylinder(y + direction * COLUMN_RADIUS, y + direction * post_radius, 0.026, Material::Post, 10);
        }
        if !blocks_exit {
            let base = Vec3::new(direction.x * post_radius, dish(post_radius), direction.z * post_radius);
            mesh.cylinder(base, Vec3::new(base.x, highest + 0.03, base.z), 0.034, Material::Post, 12);
        }
    }

    // Newspaper: a fine grid, because it has to follow the funnel's slope.
    let cells = 32;
    let paper_point = |i: i32, j: i32| {
        let across = (i as f32 / cells as f32 * 2.0 - 1.0) * PAPER_HALF_WIDTH;
        let along = (j as f32 / cells as f32 * 2.0 - 1.0) * PAPER_HALF_HEIGHT;
        let xz: Vec2 = PAPER_CENTER + PAPER_ACROSS * across + PAPER_UP * along;
        let r = xz.length();
        let d = xz / r;
        let normal = Vec3::new(-DISH_SLOPE * d.x, 1.0, -DISH_SLOPE * d.y).normalize();
        (Vec3::new(xz.x, dish(r) + 0.003, xz.y), normal)
    };
    for i in 0..cells {
        for j in 0..cells {
            mesh.quad(
                paper_point(i, j),
                paper_point(i + 1, j),
                paper_point(i + 1, j + 1),
                paper_point(i, j + 1),
                Material::Paper,
            );
        }
    }

    mesh.channel(path);
    mesh.vertices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_is_well_formed() {
        let path = TrackPath::new();
        let mesh = build_scene_mesh(&path);
        assert_eq!(mesh.len() % 3, 0);
        // The Mac version of the same scene has 44272 triangles.
        assert_eq!(mesh.len() / 3, 44272);
        for v in &mesh {
            let n = Vec3::new(v.normal[0], v.normal[1], v.normal[2]);
            assert!((n.length() - 1.0).abs() < 1e-3, "normal is not unit length: {n}");
            assert!(v.position.iter().all(|c| c.is_finite()));
            assert!((0.0..=5.0).contains(&v.position[3]));
        }
    }

    #[test]
    fn newspaper_stays_inside_the_dish() {
        use layout::*;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let corner: Vec2 = PAPER_CENTER + PAPER_ACROSS * (a * PAPER_HALF_WIDTH) + PAPER_UP * (b * PAPER_HALF_HEIGHT);
            assert!(corner.length() > HOLE_RADIUS && corner.length() < DISH_RADIUS, "corner {corner}");
        }
    }
}
