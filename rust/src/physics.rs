//! CPU marble simulation: spheres rolling through a channel, over a dish and down a drain.
//! Drained marbles queue up and drop back in at the top of the run.

use crate::scene::{layout, TrackPath};
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};
use std::collections::VecDeque;
use std::f32::consts::TAU;

/// Layout matches `Sphere` in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSphere {
    pub center_radius: [f32; 4],
    pub prev_center: [f32; 4],
    /// xyz = absorption per unit length, w = index of refraction
    pub absorption: [f32; 4],
}

/// Same layout as `VkAabbPositionsKHR`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

const PARKED: Vec3 = Vec3::new(0.0, -60.0, 0.0);
const CELL_SIZE: f32 = 0.3;
/// Not free-standing: `FREE` marks a marble that has left the channel.
const FREE: i32 = -1;

pub struct MarbleSimulation {
    path: Vec<Vec3>,
    directions: Vec<Vec3>,
    lengths: Vec<f32>,
    pub positions: Vec<Vec3>,
    previous: Vec<Vec3>,
    velocities: Vec<Vec3>,
    pub radii: Vec<f32>,
    absorption: Vec<[f32; 4]>,
    segments: Vec<i32>,
    active: Vec<bool>,
    just_spawned: Vec<bool>,
    waiting: VecDeque<usize>,
    spawn_cooldown: f32,
    bucket_count: usize,
    bucket_start: Vec<u32>,
    bucket_items: Vec<u32>,
    rng_state: u32,
}

impl MarbleSimulation {
    pub fn new(count: usize, track: &TrackPath) -> Self {
        let path = track.points.clone();
        let directions: Vec<Vec3> = path.windows(2).map(|w| (w[1] - w[0]).normalize()).collect();
        let lengths: Vec<f32> = path.windows(2).map(|w| (w[1] - w[0]).length()).collect();
        let bucket_count = (count * 4).max(4096);
        let mut sim = Self {
            path,
            directions,
            lengths,
            positions: vec![PARKED; count],
            previous: vec![PARKED; count],
            velocities: vec![Vec3::ZERO; count],
            radii: vec![0.12; count],
            absorption: vec![[0.0; 4]; count],
            segments: vec![FREE; count],
            active: vec![false; count],
            just_spawned: vec![false; count],
            waiting: VecDeque::new(),
            spawn_cooldown: 0.0,
            bucket_count,
            bucket_start: vec![0; bucket_count + 1],
            bucket_items: vec![0; count],
            rng_state: 0x9E37_79B9,
        };

        // Colour of light after passing through the width of one marble.
        let palette: [[f32; 3]; 8] = [
            [0.99, 0.995, 0.995], // clear
            [0.99, 0.86, 0.62],   // amber
            [0.98, 0.62, 0.62],   // ruby
            [0.62, 0.95, 0.72],   // emerald
            [0.62, 0.74, 0.99],   // cobalt
            [0.99, 0.995, 0.995], // clear again, because clear glass sells the effect
            [0.84, 0.66, 0.98],   // violet
            [0.62, 0.93, 0.95],   // teal
        ];
        for i in 0..count {
            sim.radii[i] = 0.1 + sim.next_random() * 0.04;
            let color = palette[i % palette.len()];
            let per_unit = |c: f32| -c.ln() / (2.0 * 0.12);
            sim.absorption[i] = [per_unit(color[0]), per_unit(color[1]), per_unit(color[2]), 1.5];
        }

        // Start with some marbles in the channel and the rest spread out over the dish,
        // so the first frames are representative. A few seconds of settling follow.
        let in_channel = (count * 2 / 5).min(100);
        let first_helix = track.first_visible + 30;
        let usable = sim.path.len() - first_helix - 2;
        for i in 0..in_channel {
            let segment = first_helix + usable * i / in_channel.max(1);
            sim.positions[i] = sim.path[segment];
            sim.segments[i] = segment as i32;
            sim.active[i] = true;
        }
        let (mut ring_radius, mut angle, mut layer) = (1.0_f32, 0.0_f32, 0.25_f32);
        for i in in_channel..count {
            sim.positions[i] = Vec3::new(angle.cos() * ring_radius, layer, angle.sin() * ring_radius);
            sim.active[i] = true;
            angle += 0.3 / ring_radius;
            if angle > TAU {
                angle = 0.0;
                ring_radius += 0.3;
                if ring_radius > layout::DISH_RADIUS - 0.15 {
                    ring_radius = 1.0;
                    layer += 0.3;
                }
            }
        }
        for _ in 0..(3 * 120) {
            sim.substep(1.0 / 120.0);
        }
        sim.begin_frame();
        sim
    }

    pub fn count(&self) -> usize {
        self.positions.len()
    }

    /// Call before stepping a frame; motion vectors are measured from this point.
    pub fn begin_frame(&mut self) {
        self.previous.copy_from_slice(&self.positions);
        self.just_spawned.fill(false);
    }

    pub fn step(&mut self, dt: f32) {
        let clamped = dt.min(1.0 / 30.0);
        for _ in 0..2 {
            self.substep(clamped / 2.0);
        }
    }

    pub fn write_gpu(&self, spheres: &mut [GpuSphere], boxes: &mut [Aabb]) {
        for i in 0..self.count() {
            let p = self.positions[i];
            let r = self.radii[i];
            // A teleported marble has no meaningful previous position.
            let prev = if self.just_spawned[i] { p } else { self.previous[i] };
            spheres[i] = GpuSphere {
                center_radius: [p.x, p.y, p.z, r],
                prev_center: [prev.x, prev.y, prev.z, 0.0],
                absorption: self.absorption[i],
            };
            boxes[i] = Aabb { min: (p - Vec3::splat(r)).to_array(), max: (p + Vec3::splat(r)).to_array() };
        }
    }

    fn next_random(&mut self) -> f32 {
        // xorshift: reproducible runs make benchmarks comparable.
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        x as f32 / u32::MAX as f32
    }

    fn substep(&mut self, dt: f32) {
        for i in 0..self.count() {
            if !self.active[i] {
                continue;
            }
            let in_channel = self.segments[i] >= 0;
            // 5/7: a rolling sphere accelerates slower than a sliding one.
            self.velocities[i].y -= 9.8 * dt * if in_channel { 0.71 } else { 1.0 };
            let drag = if in_channel { 0.55 } else { 0.28 };
            self.velocities[i] *= (1.0 - drag * dt).max(0.0);
            self.positions[i] += self.velocities[i] * dt;
        }
        self.solve_marble_contacts();
        for i in 0..self.count() {
            if self.active[i] {
                if self.segments[i] >= 0 {
                    self.solve_channel(i);
                } else {
                    self.solve_dish(i);
                }
            }
        }
        self.respawn_waiting(dt);
    }

    /// Keeps a marble inside the channel, treated as a closed tube for robustness.
    fn solve_channel(&mut self, i: usize) {
        let p = self.positions[i];
        let segment = self.segments[i] as usize;
        let last = self.directions.len() - 1;
        // Marbles move slowly relative to segment length, so a local search is enough.
        let (mut best_distance, mut best_segment, mut best_point, mut best_along) = (f32::MAX, segment, p, 0.0);
        for s in segment.saturating_sub(3)..=(segment + 6).min(last) {
            let along = (p - self.path[s]).dot(self.directions[s]).clamp(0.0, self.lengths[s]);
            let q = self.path[s] + self.directions[s] * along;
            let distance = p.distance_squared(q);
            if distance < best_distance {
                (best_distance, best_segment, best_point, best_along) = (distance, s, q, along);
            }
        }
        self.segments[i] = best_segment as i32;
        if best_segment >= last && best_along >= self.lengths[best_segment] - 1e-3 {
            self.segments[i] = FREE; // rolled off the end of the channel
            return;
        }
        let offset = p - best_point;
        let distance = offset.length();
        let limit = layout::CHANNEL_RADIUS - self.radii[i];
        if distance > limit && distance > 1e-5 {
            let outward = offset / distance;
            self.positions[i] = best_point + outward * limit;
            let speed = self.velocities[i].dot(outward);
            if speed > 0.0 {
                self.velocities[i] -= outward * speed * 1.08;
            }
        }
    }

    fn solve_dish(&mut self, i: usize) {
        use layout::*;
        let mut p = self.positions[i];
        let mut v = self.velocities[i];
        let r = self.radii[i];
        let radial = Vec2::new(p.x, p.z);
        let distance = radial.length().max(1e-4);
        let direction = radial / distance;
        let mut new_distance = distance;

        if distance > HOLE_RADIUS {
            // Over the funnel.
            let height = dish_height(distance) + r;
            if p.y < height {
                p.y = height;
                let normal = Vec3::new(-DISH_SLOPE * direction.x, 1.0, -DISH_SLOPE * direction.y).normalize();
                let into = v.dot(normal);
                if into < 0.0 {
                    v -= normal * into * 1.2;
                }
                v *= 0.9994; // rolling resistance
            }
            let wall = DISH_RADIUS - r;
            if new_distance > wall {
                new_distance = wall;
                let out = Vec2::new(v.x, v.z).dot(direction);
                if out > 0.0 {
                    v.x -= direction.x * out * 1.4;
                    v.z -= direction.y * out * 1.4;
                }
            }
        } else if p.y < 0.0 {
            // In the drain shaft: ease towards the shaft so the marble drops in smoothly.
            let outer_limit = HOLE_RADIUS - r;
            if new_distance > outer_limit {
                new_distance += (outer_limit - new_distance) * 0.2;
            }
        }

        // Central column.
        let inner = COLUMN_RADIUS + r;
        if new_distance < inner && p.y < COLUMN_HEIGHT {
            new_distance = inner;
            let out = Vec2::new(v.x, v.z).dot(direction);
            if out < 0.0 {
                v.x -= direction.x * out * 1.3;
                v.z -= direction.y * out * 1.3;
            }
        }
        if new_distance != distance {
            p.x = direction.x * new_distance;
            p.z = direction.y * new_distance;
        }

        if p.y < -0.6 {
            self.active[i] = false;
            self.positions[i] = PARKED;
            self.velocities[i] = Vec3::ZERO;
            self.segments[i] = FREE;
            self.waiting.push_back(i);
            return;
        }
        self.positions[i] = p;
        self.velocities[i] = v;
    }

    fn respawn_waiting(&mut self, dt: f32) {
        self.spawn_cooldown -= dt;
        if self.spawn_cooldown > 0.0 {
            return;
        }
        let Some(i) = self.waiting.pop_front() else { return };
        self.positions[i] = self.path[0];
        self.velocities[i] = Vec3::new(0.0, -1.5, 0.0);
        self.segments[i] = 0;
        self.active[i] = true;
        self.just_spawned[i] = true;
        // Steady stream: limit how fast the channel is fed so marbles stay spaced out.
        self.spawn_cooldown = 0.09;
    }

    fn bucket(&self, x: i64, y: i64, z: i64) -> usize {
        let h = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791);
        h.rem_euclid(self.bucket_count as i64) as usize
    }

    fn cell(p: Vec3) -> (i64, i64, i64) {
        let c = (p / CELL_SIZE).floor();
        (c.x as i64, c.y as i64, c.z as i64)
    }

    fn solve_marble_contacts(&mut self) {
        // Counting sort of the marbles into grid buckets.
        self.bucket_start.fill(0);
        for i in 0..self.count() {
            if self.active[i] {
                let (x, y, z) = Self::cell(self.positions[i]);
                let b = self.bucket(x, y, z);
                self.bucket_start[b + 1] += 1;
            }
        }
        for b in 0..self.bucket_count {
            self.bucket_start[b + 1] += self.bucket_start[b];
        }
        for i in 0..self.count() {
            if self.active[i] {
                let (x, y, z) = Self::cell(self.positions[i]);
                let b = self.bucket(x, y, z);
                self.bucket_items[self.bucket_start[b] as usize] = i as u32;
                self.bucket_start[b] += 1;
            }
        }
        // The fill above shifted each start to the next bucket's start.
        for b in (1..=self.bucket_count).rev() {
            self.bucket_start[b] = self.bucket_start[b - 1];
        }
        self.bucket_start[0] = 0;

        for i in 0..self.count() {
            if !self.active[i] {
                continue;
            }
            let (cx, cy, cz) = Self::cell(self.positions[i]);
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let b = self.bucket(cx + dx, cy + dy, cz + dz);
                        for k in self.bucket_start[b]..self.bucket_start[b + 1] {
                            let j = self.bucket_items[k as usize] as usize;
                            if j > i {
                                self.collide(i, j);
                            }
                        }
                    }
                }
            }
        }
    }

    fn collide(&mut self, i: usize, j: usize) {
        let delta = self.positions[j] - self.positions[i];
        let min_distance = self.radii[i] + self.radii[j];
        let distance2 = delta.length_squared();
        if distance2 >= min_distance * min_distance || distance2 < 1e-10 {
            return;
        }
        let distance = distance2.sqrt();
        let normal = delta / distance;
        let correction = normal * ((min_distance - distance) * 0.5);
        self.positions[i] -= correction;
        self.positions[j] += correction;
        let closing = (self.velocities[j] - self.velocities[i]).dot(normal);
        if closing < 0.0 {
            let impulse = normal * closing * 0.65; // restitution 0.3, equal masses
            self.velocities[i] += impulse;
            self.velocities[j] -= impulse;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(count: usize, seconds: f32) -> MarbleSimulation {
        let mut sim = MarbleSimulation::new(count, &TrackPath::new());
        for _ in 0..(seconds * 60.0) as usize {
            sim.begin_frame();
            sim.step(1.0 / 60.0);
        }
        sim
    }

    #[test]
    fn marbles_stay_inside_the_scene() {
        let sim = run(300, 40.0);
        for (i, p) in sim.positions.iter().enumerate() {
            assert!(p.is_finite(), "marble {i} is not finite");
            if sim.active[i] {
                assert!(Vec2::new(p.x, p.z).length() < layout::DISH_RADIUS + 0.2, "marble {i} left the dish at {p}");
                assert!(p.y > -1.0 && p.y < 8.0, "marble {i} at bad height {p}");
            }
        }
    }

    #[test]
    fn marbles_keep_circulating() {
        let mut sim = MarbleSimulation::new(150, &TrackPath::new());
        let (mut drained, mut was_active) = (0, sim.active.clone());
        for _ in 0..(90 * 60) {
            sim.begin_frame();
            sim.step(1.0 / 60.0);
            for i in 0..sim.count() {
                if was_active[i] && !sim.active[i] {
                    drained += 1;
                }
            }
            was_active.copy_from_slice(&sim.active);
        }
        assert!(drained > 50, "only {drained} marbles reached the drain in 90 seconds");
        assert!(sim.active.iter().filter(|a| **a).count() > 100, "most marbles are stuck in the respawn queue");
    }

    #[test]
    fn gpu_data_matches_the_simulation() {
        let sim = run(50, 1.0);
        let mut spheres = vec![GpuSphere::zeroed(); 50];
        let mut boxes = vec![Aabb::zeroed(); 50];
        sim.write_gpu(&mut spheres, &mut boxes);
        for i in 0..50 {
            let [x, y, z, r] = spheres[i].center_radius;
            assert_eq!(boxes[i].min, [x - r, y - r, z - r]);
            assert_eq!(boxes[i].max, [x + r, y + r, z + r]);
            assert!(spheres[i].absorption[0] >= 0.0 && spheres[i].absorption[3] == 1.5);
        }
    }
}
