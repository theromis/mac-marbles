// Resources shared by every compute pass. They all use one descriptor set layout,
// so a pass names images by index instead of needing its own set.

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, std140) uniform Uniforms {
    mat4 viewProj;
    mat4 prevViewProj;
    vec4 camPos;
    vec4 camRight;    // w = aspect ratio
    vec4 camUp;       // w = tan(fov / 2)
    vec4 camForward;
    vec4 paperOrigin; // xy = centre of the sheet on the xz plane
    vec4 paperAxes;   // xy = across axis / half width, zw = up axis / half height
    vec2 jitter;
    uvec2 size;
    uint frame;
    uint maxBounces;
    uint diffuseBounces;
    uint sphereCount;
    float time;
} u;

layout(set = 0, binding = 5, rgba16f) uniform image2D images[11];
layout(set = 0, binding = 6, rgba8) uniform writeonly image2D displayImage;

layout(push_constant) uniform Push {
    int source;
    int target;
    int stepSize;
    int denoise;
} push;

// Indices into `images`. Must match the IMAGE_ constants in renderer.rs.
const int COLOR = 0;         // rgb = traced radiance
const int ALBEDO = 1;        // rgb = first surface albedo, a = 1 for glass
const int NORMAL_DEPTH = 2;  // xyz = normal, w = distance from camera (0 = nothing hit)
const int MOTION = 3;        // xy = offset in pixels to this point in the previous frame
const int PREV_NORMAL_DEPTH = 4;
const int HISTORY_NEW = 5;   // rgb = accumulated irradiance, a = frames accumulated
const int HISTORY_OLD = 6;
const int FILTER_A = 7;
const int FILTER_B = 8;
const int TAA_NEW = 9;       // rgb = resolved colour, a = 1 once valid
const int TAA_OLD = 10;

#define PIXEL_GUARD                                       \
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);        \
    if (any(greaterThanEqual(pixel, ivec2(u.size)))) return;
