// Real-time path tracer for glass marbles on a wooden track.
//
// Geometry: triangles (wood) and analytic spheres (glass) in a hardware BVH.
// Output: noisy HDR colour plus the guide buffers MetalFX needs to denoise it.
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

// ---------------------------------------------------------------- shared data

struct Uniforms {
    float4x4 viewProj;
    float4x4 prevViewProj;
    float4 camPos;
    float4 camRight;    // w = aspect ratio
    float4 camUp;       // w = tan(fov / 2)
    float4 camForward;
    float4 paperOrigin;  // xy = centre of the sheet on the xz plane
    float4 paperAxes;    // xy = across axis / half width, zw = up axis / half height
    float2 jitter;
    uint2 size;
    uint frame;
    uint maxBounces;
    uint diffuseBounces;
    uint sphereCount;
    float time;
    uint pad0;
    uint pad1;
};

struct Sphere {
    float4 centerRadius;
    float4 prevCenter;
    float4 absorption;  // xyz = absorption per unit length, w = index of refraction
};

struct Vertex {
    float4 position;    // w = material id
    float4 normal;
};

constant uint kMaterialFloor = 0;
constant uint kMaterialDish = 1;
constant uint kMaterialTrack = 2;
constant uint kMaterialPost = 3;
constant uint kMaterialDark = 4;
constant uint kMaterialPaper = 5;

constant uint kInstanceTriangles = 0;
constant float kEpsilon = 6e-4f;
constant float kPi = 3.14159265f;

typedef intersector<triangle_data, instancing> Intersector;
typedef intersection_function_table<triangle_data, instancing> FunctionTable;
typedef instance_acceleration_structure Accel;

// ---------------------------------------------------------- sphere intersection

struct SphereHit {
    bool accept [[accept_intersection]];
    float distance [[distance]];
};

// Called by the hardware for each candidate bounding box. Returns the near root,
// or the far root when the ray starts inside the sphere.
[[intersection(bounding_box, triangle_data, instancing)]]
SphereHit sphereIntersection(float3 origin [[origin]],
                             float3 direction [[direction]],
                             float minDistance [[min_distance]],
                             float maxDistance [[max_distance]],
                             uint primitive [[primitive_id]],
                             const device Sphere *spheres [[buffer(0)]]) {
    SphereHit hit;
    hit.accept = false;
    hit.distance = 0;
    float3 oc = origin - spheres[primitive].centerRadius.xyz;
    float r = spheres[primitive].centerRadius.w;
    float b = dot(oc, direction);
    float h = b * b - (dot(oc, oc) - r * r);
    if (h < 0) return hit;
    h = sqrt(h);
    float t = -b - h;
    if (t < minDistance) t = -b + h;
    if (t < minDistance || t > maxDistance) return hit;
    hit.accept = true;
    hit.distance = t;
    return hit;
}

// ---------------------------------------------------------------------- random

struct Rng { uint state; };

inline float rnd(thread Rng &rng) {
    rng.state = rng.state * 747796405u + 2891336453u;
    uint w = ((rng.state >> ((rng.state >> 28u) + 4u)) ^ rng.state) * 277803737u;
    return float((w >> 22u) ^ w) * (1.0f / 4294967296.0f);
}

inline float hash31(float3 p) {
    p = fract(p * 0.1031f);
    p += dot(p, p.zyx + 31.32f);
    return fract((p.x + p.y) * p.z);
}

inline float valueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float a = mix(mix(hash31(i), hash31(i + float3(1, 0, 0)), f.x),
                  mix(hash31(i + float3(0, 1, 0)), hash31(i + float3(1, 1, 0)), f.x), f.y);
    float b = mix(mix(hash31(i + float3(0, 0, 1)), hash31(i + float3(1, 0, 1)), f.x),
                  mix(hash31(i + float3(0, 1, 1)), hash31(i + float3(1, 1, 1)), f.x), f.y);
    return mix(a, b, f.z);
}

// -------------------------------------------------------------------- materials

struct Surface {
    float3 albedo;
    float roughness;
};

// Procedural wood: stretched growth rings disturbed by low-frequency noise.
Surface woodSurface(uint material, float3 p, bool detailed) {
    Surface s;
    if (material == kMaterialDark) {
        s.albedo = float3(0.012f, 0.009f, 0.007f);
        s.roughness = 0.8f;
        return s;
    }
    float3 q = p;
    float3 light, dark;
    float scale = 5.0f;
    s.roughness = 0.38f;
    if (material == kMaterialFloor) {
        light = float3(0.085f, 0.046f, 0.026f);
        dark = float3(0.040f, 0.021f, 0.012f);
        // Planks along x.
        float plank = floor(q.z / 1.4f);
        q.x += plank * 3.7f;
        q.z += plank * 11.0f;
        s.roughness = 0.5f;
    } else if (material == kMaterialDish) {
        light = float3(0.27f, 0.15f, 0.075f);
        dark = float3(0.17f, 0.085f, 0.04f);
        scale = 7.0f;
    } else if (material == kMaterialTrack) {
        light = float3(0.60f, 0.41f, 0.23f);
        dark = float3(0.27f, 0.15f, 0.07f);
        scale = 15.0f;
        s.roughness = 0.42f;
    } else {
        light = float3(0.42f, 0.23f, 0.11f);
        dark = float3(0.21f, 0.10f, 0.045f);
        scale = 18.0f;
        q = q.yxz;
    }
    // Grain is invisible after a diffuse bounce, so indirect hits skip the noise evaluation.
    if (!detailed) {
        s.albedo = (light + dark) * 0.5f;
        return s;
    }
    float3 stretched = float3(q.x * 0.16f, q.y, q.z) * scale;
    float warp = valueNoise(stretched * 0.55f);
    float rings = fract(length(stretched.yz) * 0.55f + warp * 1.1f);
    rings = smoothstep(0.0f, 0.75f, rings) * (1.0f - smoothstep(0.75f, 1.0f, rings));
    float grain = valueNoise(stretched * float3(1.0f, 11.0f, 11.0f));
    s.albedo = mix(dark, light, saturate(rings * 0.75f + grain * 0.35f));
    return s;
}

inline float schlick(float cosine, float f0) {
    return f0 + (1.0f - f0) * pow(saturate(1.0f - cosine), 5.0f);
}

float dielectricFresnel(float cosI, float eta) {
    float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T >= 1.0f) return 1.0f;
    float cosT = sqrt(1.0f - sin2T);
    float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rs * rs + rp * rp);
}

inline float3 orthogonal(float3 n) {
    return abs(n.x) > 0.5f ? normalize(cross(n, float3(0, 1, 0))) : normalize(cross(n, float3(1, 0, 0)));
}

inline float3 sampleCosine(float3 n, thread Rng &rng) {
    float u = rnd(rng) * 2.0f * kPi;
    float v = rnd(rng);
    float r = sqrt(v);
    float3 t = orthogonal(n);
    float3 b = cross(n, t);
    return normalize(t * (r * cos(u)) + b * (r * sin(u)) + n * sqrt(max(0.0f, 1.0f - v)));
}

// GGX half-vector sample, used for the glossy lobe of the varnished wood.
inline float3 sampleGgxReflection(float3 n, float3 v, float roughness, thread Rng &rng) {
    float a = roughness * roughness;
    float u = rnd(rng);
    float phi = rnd(rng) * 2.0f * kPi;
    float cosT = sqrt((1.0f - u) / (1.0f + (a * a - 1.0f) * u));
    float sinT = sqrt(max(0.0f, 1.0f - cosT * cosT));
    float3 t = orthogonal(n);
    float3 b = cross(n, t);
    float3 h = normalize(t * (sinT * cos(phi)) + b * (sinT * sin(phi)) + n * cosT);
    return reflect(-v, h);
}

float3 surfaceBrdf(Surface s, float3 n, float3 v, float3 l) {
    float3 h = normalize(v + l);
    float nl = saturate(dot(n, l));
    float nv = max(dot(n, v), 1e-3f);
    float nh = saturate(dot(n, h));
    float vh = saturate(dot(v, h));
    float a = s.roughness * s.roughness;
    float a2 = a * a;
    float denom = nh * nh * (a2 - 1.0f) + 1.0f;
    float d = a2 / max(kPi * denom * denom, 1e-5f);
    float k = a * 0.5f;
    float vis = 1.0f / max(4.0f * (nl * (1.0f - k) + k) * (nv * (1.0f - k) + k), 1e-4f);
    float f = schlick(vh, 0.04f);
    return s.albedo * (1.0f - f) / kPi + float3(min(d * vis * f, 8.0f));
}

// ------------------------------------------------------------------------ lights

struct RectLight {
    float3 center;
    float3 u;        // half-extent vectors
    float3 v;
    float3 normal;
    float3 emission;
};

RectLight rectLight(float3 center, float3 target, float halfU, float halfV, float3 emission) {
    RectLight l;
    l.center = center;
    l.normal = normalize(target - center);
    float3 side = normalize(cross(l.normal, float3(0, 1, 0)));
    l.u = side * halfU;
    l.v = cross(side, l.normal) * halfV;
    l.emission = emission;
    return l;
}

inline RectLight lightAt(uint index) {
    if (index == 0) {
        return rectLight(float3(3.8f, 7.5f, 3.0f), float3(0, 0.8f, 0), 1.6f, 1.6f,
                         float3(1.0f, 0.78f, 0.52f) * 15.0f);
    }
    return rectLight(float3(-6.5f, 3.6f, -2.4f), float3(0, 1.0f, 0), 0.9f, 1.8f,
                     float3(0.62f, 0.74f, 1.0f) * 7.0f);
}

inline float3 hitLights(float3 origin, float3 dir, float maxT) {
    for (uint i = 0; i < 2; i++) {
        RectLight l = lightAt(i);
        float denom = dot(dir, l.normal);
        if (denom > -1e-4f) continue;
        float t = dot(l.center - origin, l.normal) / denom;
        if (t <= 0.0f || t >= maxT) continue;
        float3 d = origin + dir * t - l.center;
        float a = dot(d, l.u) / dot(l.u, l.u);
        float b = dot(d, l.v) / dot(l.v, l.v);
        // Round with a soft edge: a hard rectangle looks like a pixelated square in a small marble.
        float falloff = smoothstep(1.0f, 0.55f, a * a + b * b);
        if (falloff > 0.0f) return l.emission * falloff;
    }
    return float3(0);
}

// A dim warm room, so glass has something to refract and reflect in every direction.
inline float3 environment(float3 d) {
    float up = d.y * 0.5f + 0.5f;
    float3 c = mix(float3(0.03f, 0.022f, 0.017f), float3(0.13f, 0.095f, 0.07f), smoothstep(0.35f, 0.95f, up));
    // A soft horizontal band reads as a far wall in reflections.
    c += float3(0.10f, 0.06f, 0.034f) * exp(-pow((d.y - 0.1f) * 5.0f, 2.0f));
    return c;
}

// ------------------------------------------------------------------- ray queries

// Light passing through a glass ball, treated as a thick lens. This turns each shadow
// ray into a cheap caustic: bright focus in the middle of the shadow, darker around it,
// with a slight colour fringe from a per-channel index of refraction.
float3 lensTransmission(Sphere s, float3 p, float3 wi) {
    float3 toCenter = s.centerRadius.xyz - p;
    float r = s.centerRadius.w;
    float along = dot(toCenter, wi);
    float impact = length(toCenter - wi * along);
    float3 n = s.absorption.w + float3(-0.012f, 0.0f, 0.02f);
    float3 focal = n * r / (2.0f * (n - 1.0f));
    float3 m = max(abs(1.0f - along / focal), float3(0.24f));
    float3 inside = smoothstep(r * m * 1.15f, r * m * 0.7f, float3(impact));
    float3 gain = min(1.0f / (m * m), float3(9.0f));
    float chord = 2.0f * sqrt(max(r * r - impact * impact, 0.0f));
    float3 absorb = exp(-s.absorption.xyz * chord);
    // Glass redirects light rather than blocking it. A black shadow seen through a marble
    // is what makes it read as opaque, so the area outside the focus stays fairly bright.
    return max(mix(float3(0.42f), gain * 0.88f, inside), float3(0.42f)) * absorb;
}

float3 shadowTransmission(float3 p, float3 wi, float maxT, Accel accel, FunctionTable table,
                          const device Sphere *spheres) {
    Intersector isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    float3 transmission = 1.0f;
    float start = kEpsilon;
    for (uint i = 0; i < 4; i++) {
        ray r(p, wi, start, maxT);
        auto hit = isect.intersect(r, accel, 0xFF, table);
        if (hit.type == intersection_type::none) return transmission;
        if (hit.type == intersection_type::triangle) return float3(0);
        Sphere s = spheres[hit.primitive_id];
        transmission *= lensTransmission(s, p, wi);
        // Skip past the back of this ball.
        float3 oc = p - s.centerRadius.xyz;
        float b = dot(oc, wi);
        float h = max(b * b - (dot(oc, oc) - s.centerRadius.w * s.centerRadius.w), 0.0f);
        start = -b + sqrt(h) + kEpsilon;
    }
    return transmission * 0.3f;
}

float3 directLight(float3 p, float3 n, float3 v, Surface s, thread Rng &rng, Accel accel,
                   FunctionTable table, const device Sphere *spheres) {
    // Uniform light choice: the key is brighter but both are about the same solid angle
    // from the scene, so the fill does not get undersampled.
    uint index = rnd(rng) < 0.75f ? 0 : 1;
    float pick = index == 0 ? 0.75f : 0.25f;
    RectLight l = lightAt(index);
    float3 target = l.center + l.u * (rnd(rng) * 2.0f - 1.0f) + l.v * (rnd(rng) * 2.0f - 1.0f);
    float3 toLight = target - p;
    float dist2 = dot(toLight, toLight);
    float dist = sqrt(dist2);
    float3 wi = toLight / dist;
    float nl = dot(n, wi);
    float cosLight = -dot(wi, l.normal);
    if (nl <= 0.0f || cosLight <= 0.0f) return float3(0);
    float area = 4.0f * length(l.u) * length(l.v);
    float3 visible = shadowTransmission(p + n * kEpsilon, wi, dist - kEpsilon, accel, table, spheres);
    if (all(visible <= 0.0f)) return float3(0);
    float geometry = nl * cosLight * area / (dist2 * pick);
    return l.emission * visible * surfaceBrdf(s, n, v, wi) * geometry;
}

// Lighting with no randomness, for surfaces seen through glass. Those pixels skip the denoiser
// to stay crisp, so they must not contain noise: one shadow ray to the centre of each light,
// and a constant in place of indirect light.
float3 cleanLight(float3 p, float3 n, float3 v, Surface s, Accel accel, FunctionTable table,
                  const device Sphere *spheres) {
    float3 result = s.albedo * float3(0.20f, 0.15f, 0.115f);
    for (uint i = 0; i < 2; i++) {
        RectLight l = lightAt(i);
        float3 toLight = l.center - p;
        float dist2 = dot(toLight, toLight);
        float dist = sqrt(dist2);
        float3 wi = toLight / dist;
        float nl = dot(n, wi);
        float cosLight = -dot(wi, l.normal);
        if (nl <= 0.0f || cosLight <= 0.0f) continue;
        float3 visible = shadowTransmission(p + n * kEpsilon, wi, dist - kEpsilon, accel, table, spheres);
        float area = 4.0f * length(l.u) * length(l.v);
        result += l.emission * visible * surfaceBrdf(s, n, v, wi) * nl * cosLight * area / dist2;
    }
    return result;
}

// --------------------------------------------------------------------- path tracer

struct FirstHit {
    float3 position;
    float3 normal;
    float3 motionPosition;  // where this point was in the previous frame
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float roughness;
    bool valid;
    bool glass;
    bool awaitingSurfaceBehindGlass;
};

float3 tracePath(ray r, thread Rng &rng, constant Uniforms &u, Accel accel, FunctionTable table,
                 const device Vertex *vertices, const device Sphere *spheres,
                 texture2d<float> paper, thread FirstHit &first, bool recordFirst, bool primaryReflection) {
    Intersector isect;
    isect.set_triangle_cull_mode(triangle_cull_mode::none);
    float3 radiance = 0.0f;
    float3 throughput = 1.0f;
    bool sharpBounce = true;  // emitters are visible only after mirror/glass bounces
    uint diffuseHits = 0;

    for (uint bounce = 0; bounce < u.maxBounces; bounce++) {
        auto hit = isect.intersect(r, accel, 0xFF, table);
        float maxT = hit.type == intersection_type::none ? 1e6f : hit.distance;
        // Lights stay out of direct view but show up in glass.
        if (sharpBounce && bounce > 0) radiance += throughput * hitLights(r.origin, r.direction, maxT);

        if (hit.type == intersection_type::none) {
            // Glass shows the room directly, so it gets a brighter one than the soft bounce light.
            radiance += throughput * environment(r.direction) * (sharpBounce && bounce > 0 ? 3.0f : 1.0f);
            return radiance;
        }

        float3 p = r.origin + r.direction * hit.distance;

        if (hit.instance_id != kInstanceTriangles) {
            // ------------------------------------------------------- glass
            Sphere s = spheres[hit.primitive_id];
            float3 outward = (p - s.centerRadius.xyz) / s.centerRadius.w;
            bool entering = dot(r.direction, outward) < 0.0f;
            float3 n = entering ? outward : -outward;
            if (!entering) throughput *= exp(-s.absorption.xyz * hit.distance);
            if (recordFirst && bounce == 0) {
                first.valid = true;
                first.glass = true;
                first.position = p;
                first.normal = outward;
                first.motionPosition = p - s.centerRadius.xyz + s.prevCenter.xyz;
                first.diffuseAlbedo = float3(0);
                first.specularAlbedo = float3(1);
                first.roughness = 0.0f;
                first.awaitingSurfaceBehindGlass = true;
            }
            float eta = entering ? 1.0f / s.absorption.w : s.absorption.w;
            float cosI = saturate(-dot(r.direction, n));
            float fresnel = dielectricFresnel(cosI, eta);
            // No random choice here, since random choices are what make glass look frosted
            // once denoised. The first hit splits exactly into a reflection path and a
            // refraction path (see the kernel); later hits follow the dominant direction.
            bool reflects;
            if (bounce == 0) {
                reflects = primaryReflection;
                throughput *= reflects ? fresnel : 1.0f - fresnel;
            } else {
                reflects = fresnel >= 0.5f;
            }
            // Internal reflection without an extra ray: look up what it would see directly.
            // This gives the secondary highlight on the far side of real glass balls.
            if (bounce > 0 && !entering && !reflects) {
                float3 inner = reflect(r.direction, n);
                radiance += throughput * fresnel * (hitLights(p, inner, 1e6f) + environment(inner) * 3.0f);
                throughput *= 1.0f - fresnel;
            }
            float3 dir;
            if (reflects) {
                dir = reflect(r.direction, n);
                r.origin = p + n * kEpsilon;
            } else {
                dir = refract(r.direction, n, eta);
                r.origin = p - n * kEpsilon;
            }
            r.direction = normalize(dir);
            r.min_distance = kEpsilon;
            r.max_distance = 1e6f;
            sharpBounce = true;
            continue;
        }

        // ----------------------------------------------------------- wood
        uint base = hit.primitive_id * 3;
        float2 bary = hit.triangle_barycentric_coord;
        float w = 1.0f - bary.x - bary.y;
        float3 n = normalize(vertices[base].normal.xyz * w +
                             vertices[base + 1].normal.xyz * bary.x +
                             vertices[base + 2].normal.xyz * bary.y);
        if (dot(n, r.direction) > 0.0f) n = -n;
        uint material = uint(vertices[base].position.w);
        Surface surface;
        if (material == kMaterialPaper) {
            surface.roughness = 0.75f;
            surface.albedo = float3(0.60f, 0.575f, 0.52f);
            if (diffuseHits == 0) {
                constexpr sampler linear(filter::linear, mip_filter::linear, address::clamp_to_edge);
                float2 offset = p.xz - u.paperOrigin.xy;
                float2 uv = float2(dot(offset, u.paperAxes.xy), dot(offset, u.paperAxes.zw)) * 0.5f + 0.5f;
                // Print is sampled sharper through glass, where a marble magnifies it.
                float ink = paper.sample(linear, float2(uv.x, 1.0f - uv.y), level(bounce > 0 ? 0.5f : 1.5f)).r;
                surface.albedo = mix(float3(0.035f), float3(0.76f, 0.73f, 0.66f), ink);
            }
        } else {
            surface = woodSurface(material, p, diffuseHits == 0);
        }
        float3 v = -r.direction;

        if (recordFirst && bounce == 0) {
            first.valid = true;
            first.glass = false;
            first.position = p;
            first.normal = n;
            first.motionPosition = p;
            first.diffuseAlbedo = surface.albedo;
            first.specularAlbedo = float3(0.04f);
            first.roughness = surface.roughness;
        }

        // Show the denoiser what is seen through the glass. It can then keep that image
        // sharp and clean only the lighting noise, as it does for any other surface.
        if (recordFirst && first.awaitingSurfaceBehindGlass) {
            first.awaitingSurfaceBehindGlass = false;
            first.normal = n;
            first.diffuseAlbedo = saturate(surface.albedo * throughput);
            first.specularAlbedo = float3(0.04f);
            first.roughness = surface.roughness;
        }

        if (bounce > 0 && diffuseHits == 0) {
            radiance += throughput * cleanLight(p, n, v, surface, accel, table, spheres);
            return radiance;
        }

        radiance += throughput * directLight(p, n, v, surface, rng, accel, table, spheres);

        diffuseHits += 1;
        if (diffuseHits > u.diffuseBounces) return radiance;

        float nv = saturate(dot(n, v));
        float specularChance = max(schlick(nv, 0.04f), 0.15f);
        float3 dir;
        if (rnd(rng) < specularChance) {
            dir = sampleGgxReflection(n, v, surface.roughness, rng);
            if (dot(dir, n) <= 0.0f) return radiance;
            throughput *= schlick(nv, 0.04f) / specularChance;
        } else {
            dir = sampleCosine(n, rng);
            throughput *= surface.albedo * (1.0f - schlick(nv, 0.04f)) / (1.0f - specularChance);
        }
        r.origin = p + n * kEpsilon;
        r.direction = dir;
        r.min_distance = kEpsilon;
        r.max_distance = 1e6f;
        sharpBounce = false;
    }
    // Out of bounces, usually inside glass. A dim ambient guess beats a black pixel.
    radiance += throughput * float3(0.09f, 0.065f, 0.045f);
    return radiance;
}

inline float2 toPixel(float4x4 viewProj, float3 world, float2 size) {
    float4 clip = viewProj * float4(world, 1.0f);
    float2 ndc = clip.xy / clip.w;
    return float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f) * size;
}

kernel void pathTrace(uint2 pixel [[thread_position_in_grid]],
                      texture2d<float, access::write> colorOut [[texture(0)]],
                      texture2d<float, access::write> depthOut [[texture(1)]],
                      texture2d<float, access::write> motionOut [[texture(2)]],
                      texture2d<float, access::write> normalOut [[texture(3)]],
                      texture2d<float, access::write> diffuseOut [[texture(4)]],
                      texture2d<float, access::write> specularOut [[texture(5)]],
                      texture2d<float, access::write> roughnessOut [[texture(6)]],
                      texture2d<float, access::write> reactiveOut [[texture(7)]],
                      texture2d<float, access::write> denoiseMaskOut [[texture(8)]],
                      texture2d<float> paper [[texture(9)]],
                      constant Uniforms &u [[buffer(0)]],
                      Accel accel [[buffer(1)]],
                      FunctionTable table [[buffer(2)]],
                      const device Vertex *vertices [[buffer(3)]],
                      const device Sphere *spheres [[buffer(4)]]) {
    Rng rng;
    rng.state = (pixel.x * 1973u + pixel.y * 9277u + u.frame * 26699u) | 1u;
    rnd(rng);

    float2 size = float2(u.size);
    float2 sample = (float2(pixel) + 0.5f - u.jitter) / size * 2.0f - 1.0f;
    float tanFov = u.camUp.w;
    float3 dir = normalize(u.camForward.xyz +
                           u.camRight.xyz * (sample.x * tanFov * u.camRight.w) -
                           u.camUp.xyz * (sample.y * tanFov));
    ray r(u.camPos.xyz, dir, 0.0f, 1e6f);

    FirstHit first;
    first.valid = false;
    first.glass = false;
    first.awaitingSurfaceBehindGlass = false;
    float3 color = tracePath(r, rng, u, accel, table, vertices, spheres, paper, first, true, false);

    if (first.glass) {
        // The first run followed the refraction, this one adds the reflection.
        color += tracePath(r, rng, u, accel, table, vertices, spheres, paper, first, false, true);
    }
    // Keeps rare very bright paths from turning into persistent denoiser blobs.
    float luminance = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    if (luminance > 24.0f) color *= 24.0f / luminance;
    if (any(isnan(color)) || any(isinf(color))) color = float3(0);
    colorOut.write(float4(color, 1.0f), pixel);

    // What glass shows changes faster than its surface moves, so old frames are less useful there.
    reactiveOut.write(float4(first.glass ? 0.3f : 0.0f), pixel);
    // The denoiser wipes out the fine detail that makes glass look clear, so it skips these
    // pixels. They are free of noise anyway, thanks to cleanLight.
    denoiseMaskOut.write(float4(first.glass ? 1.0f : 0.0f), pixel);

    if (first.valid) {
        float4 clip = u.viewProj * float4(first.position, 1.0f);
        depthOut.write(float4(clip.z / clip.w), pixel);
        float2 now = toPixel(u.viewProj, first.position, size);
        float2 before = toPixel(u.prevViewProj, first.motionPosition, size);
        motionOut.write(float4(before - now, 0, 0), pixel);
        normalOut.write(float4(first.normal, 0), pixel);
        diffuseOut.write(float4(first.diffuseAlbedo, 1), pixel);
        specularOut.write(float4(first.specularAlbedo, 1), pixel);
        roughnessOut.write(float4(first.roughness), pixel);
    } else {
        float3 far = u.camPos.xyz + dir * 1000.0f;
        depthOut.write(float4(1.0f), pixel);
        float2 before = toPixel(u.prevViewProj, far, size);
        float2 now = toPixel(u.viewProj, far, size);
        motionOut.write(float4(before - now, 0, 0), pixel);
        normalOut.write(float4(-dir, 0), pixel);
        diffuseOut.write(float4(0, 0, 0, 1), pixel);
        specularOut.write(float4(0, 0, 0, 1), pixel);
        roughnessOut.write(float4(1.0f), pixel);
    }
}

// ------------------------------------------------------------------ final display

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenOut fullscreenVertex(uint id [[vertex_id]]) {
    FullscreenOut out;
    out.uv = float2((id << 1) & 2, id & 2);
    out.position = float4(out.uv.x * 2.0f - 1.0f, 1.0f - out.uv.y * 2.0f, 0.0f, 1.0f);
    return out;
}

inline float3 acesTonemap(float3 x) {
    return saturate((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f));
}

fragment float4 displayFragment(FullscreenOut in [[stage_in]],
                                texture2d<float> image [[texture(0)]]) {
    constexpr sampler linear(filter::linear, address::clamp_to_edge);
    float3 hdr = image.sample(linear, in.uv).rgb;
    // The swapchain format is sRGB, so the hardware applies the transfer curve.
    return float4(acesTonemap(hdr * 0.8f), 1.0f);
}
