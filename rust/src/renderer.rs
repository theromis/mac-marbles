//! Vulkan path tracing renderer. All GPU work happens in compute passes; the result is then
//! copied to the window. One frame runs at a time, which keeps resource lifetimes simple:
//! nothing the CPU updates can still be in use by the GPU.

use ash::{ext, khr, vk};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use glass_marbles::physics::{Aabb, GpuSphere, MarbleSimulation};
use glass_marbles::scene::{build_scene_mesh, layout, TrackPath, Vertex};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{c_char, c_void, CStr};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// Indices into the shared image array. Must match common.glsl.
const IMAGE_NORMAL_DEPTH: usize = 2;
const IMAGE_PREV_NORMAL_DEPTH: usize = 4;
const IMAGE_HISTORY_NEW: usize = 5;
const IMAGE_HISTORY_OLD: usize = 6;
const IMAGE_FILTER_A: i32 = 7;
const IMAGE_FILTER_B: i32 = 8;
const IMAGE_TAA_NEW: usize = 9;
const IMAGE_TAA_OLD: usize = 10;
const IMAGE_COUNT: usize = 11;

/// Timestamps per frame: start, after acceleration structures, after tracing, end.
const TIMESTAMPS: u32 = 4;

/// Layout matches the `Uniforms` block in common.glsl (std140).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    cam_pos: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_forward: [f32; 4],
    paper_origin: [f32; 4],
    paper_axes: [f32; 4],
    jitter: [f32; 2],
    size: [u32; 2],
    frame: u32,
    max_bounces: u32,
    diffuse_bounces: u32,
    sphere_count: u32,
    time: f32,
    pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    source: i32,
    target: i32,
    step_size: i32,
    denoise: i32,
}

/// One step of the quality ladder. Index 0 is the highest quality.
#[derive(Clone, Copy, PartialEq)]
pub struct QualityTier {
    pub name: &'static str,
    /// Traced size relative to the window; the final copy stretches the rest.
    pub scale: f32,
    pub bounces: u32,
    pub diffuse_bounces: u32,
}

pub const TIERS: [QualityTier; 5] = [
    QualityTier { name: "cinematic", scale: 1.0, bounces: 14, diffuse_bounces: 2 },
    QualityTier { name: "high", scale: 1.0, bounces: 12, diffuse_bounces: 2 },
    QualityTier { name: "medium", scale: 0.75, bounces: 8, diffuse_bounces: 1 },
    QualityTier { name: "balanced", scale: 0.66, bounces: 6, diffuse_bounces: 1 },
    QualityTier { name: "low", scale: 0.5, bounces: 6, diffuse_bounces: 1 },
];

/// GPU time of the last finished frame, in milliseconds.
#[derive(Clone, Copy, Default)]
pub struct FrameTimes {
    pub build_ms: f32,
    pub trace_ms: f32,
    pub post_ms: f32,
}

impl FrameTimes {
    pub fn total(&self) -> f32 {
        self.build_ms + self.trace_ms + self.post_ms
    }
}

pub struct WindowHandles {
    pub display: RawDisplayHandle,
    pub window: RawWindowHandle,
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    address: vk::DeviceAddress,
}

struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

struct Accel {
    handle: vk::AccelerationStructureKHR,
    buffer: Buffer,
    address: vk::DeviceAddress,
}

struct Swapchain {
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    render_finished: Vec<vk::Semaphore>,
}

/// Everything sized by the marble count.
struct SphereResources {
    spheres: Buffer,
    boxes: Buffer,
    bottom: Accel,
    top: Accel,
    scratch: Buffer,
}

/// Everything sized by the render resolution.
struct RenderTargets {
    images: Vec<Image>,
    display: Image,
    width: u32,
    height: u32,
}

struct Pipelines {
    trace: vk::Pipeline,
    temporal: vk::Pipeline,
    atrous: vk::Pipeline,
    resolve: vk::Pipeline,
    tonemap: vk::Pipeline,
}

pub struct Renderer {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    memory: vk::PhysicalDeviceMemoryProperties,
    accel_api: khr::acceleration_structure::Device,
    scratch_alignment: u64,
    timestamp_period: f32,
    pub device_name: String,

    surface_api: khr::surface::Instance,
    swapchain_api: Option<khr::swapchain::Device>,
    surface: vk::SurfaceKHR,
    swapchain: Option<Swapchain>,
    vsync: bool,

    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    query_pool: vk::QueryPool,
    timestamps_ready: bool,
    last_times: FrameTimes,

    descriptor_pool: vk::DescriptorPool,
    set_layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Pipelines,

    sampler: vk::Sampler,
    paper: Image,
    vertex_buffer: Buffer,
    triangle_count: u32,
    triangle_accel: Accel,
    uniforms: Buffer,
    instances: Buffer,
    spheres: SphereResources,
    targets: RenderTargets,
    window_width: u32,
    window_height: u32,

    path: TrackPath,
    pub simulation: MarbleSimulation,
    pub tier: QualityTier,
    pub denoise: bool,
    pub camera_paused: bool,
    pub close_up: bool,
    camera_phase: f32,
    frame_index: u32,
    previous_view_proj: Option<Mat4>,
    time: f32,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*data).p_message).to_string_lossy();
    eprintln!("[vulkan {severity:?}] {message}");
    vk::FALSE
}

pub struct RendererConfig {
    pub marbles: usize,
    pub tier: QualityTier,
    pub denoise: bool,
    pub validate: bool,
    pub vsync: bool,
    pub width: u32,
    pub height: u32,
}

impl Renderer {
    pub fn new(config: &RendererConfig, handles: Option<WindowHandles>) -> Result<Self> {
        unsafe { Self::create(config, handles) }
    }

    unsafe fn create(config: &RendererConfig, handles: Option<WindowHandles>) -> Result<Self> {
        let entry = ash::Entry::load().map_err(|e| {
            format!("Could not load Vulkan ({e}). Update the graphics driver; it provides the Vulkan library.")
        })?;

        // ------------------------------------------------- instance
        let app_name = c"Glass Marbles";
        let app = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::API_VERSION_1_2);
        let mut instance_extensions: Vec<*const c_char> = Vec::new();
        if let Some(h) = &handles {
            instance_extensions.extend_from_slice(ash_window::enumerate_required_extensions(h.display)?);
        }
        let mut layers: Vec<*const c_char> = Vec::new();
        if config.validate {
            let available = entry.enumerate_instance_layer_properties()?;
            let wanted = c"VK_LAYER_KHRONOS_validation";
            if available.iter().any(|l| l.layer_name_as_c_str() == Ok(wanted)) {
                layers.push(wanted.as_ptr());
                instance_extensions.push(ext::debug_utils::NAME.as_ptr());
            } else {
                eprintln!("--validate: the validation layer is missing. Install the Vulkan SDK to get it.");
            }
        }
        let validating = !layers.is_empty();
        let instance = entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(&instance_extensions)
                .enabled_layer_names(&layers),
            None,
        )?;
        let debug = if validating {
            let api = ext::debug_utils::Instance::new(&entry, &instance);
            let messenger = api.create_debug_utils_messenger(
                &vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(debug_callback)),
                None,
            )?;
            Some((api, messenger))
        } else {
            None
        };

        let surface_api = khr::surface::Instance::new(&entry, &instance);
        let surface = match &handles {
            Some(h) => ash_window::create_surface(&entry, &instance, h.display, h.window, None)?,
            None => vk::SurfaceKHR::null(),
        };

        // ------------------------------------------------- device
        let required: Vec<&CStr> = {
            let mut list = vec![
                khr::acceleration_structure::NAME,
                khr::ray_query::NAME,
                khr::deferred_host_operations::NAME,
            ];
            if handles.is_some() {
                list.push(khr::swapchain::NAME);
            }
            list
        };
        let mut candidates = Vec::new();
        let mut rejected = Vec::new();
        for physical in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical);
            let name = properties.device_name_as_c_str().unwrap_or(c"unknown").to_string_lossy().into_owned();
            let extensions = instance.enumerate_device_extension_properties(physical)?;
            let missing: Vec<_> = required
                .iter()
                .filter(|wanted| !extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(**wanted)))
                .map(|m| m.to_string_lossy().into_owned())
                .collect();
            if !missing.is_empty() {
                rejected.push(format!("{name}: missing {}", missing.join(", ")));
                continue;
            }
            if properties.api_version < vk::API_VERSION_1_2 {
                rejected.push(format!("{name}: needs Vulkan 1.2"));
                continue;
            }
            let families = instance.get_physical_device_queue_family_properties(physical);
            let family = families.iter().enumerate().position(|(index, f)| {
                let capable = f.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
                    && f.timestamp_valid_bits > 0;
                let presents = handles.is_none()
                    || surface_api.get_physical_device_surface_support(physical, index as u32, surface).unwrap_or(false);
                capable && presents
            });
            let Some(family) = family else {
                rejected.push(format!("{name}: no suitable queue"));
                continue;
            };
            let rank = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 0,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 2,
            };
            candidates.push((rank, physical, family as u32, properties, name));
        }
        candidates.sort_by_key(|c| c.0);
        let Some((_, physical, family, properties, device_name)) = candidates.into_iter().next() else {
            return Err(format!(
                "No GPU with Vulkan hardware ray tracing was found. Update the driver if the card should support it.\n{}",
                rejected.join("\n")
            )
            .into());
        };

        let mut accel_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut accel_properties);
        instance.get_physical_device_properties2(physical, &mut properties2);
        let scratch_alignment = accel_properties.min_acceleration_structure_scratch_offset_alignment as u64;

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let extension_pointers: Vec<*const c_char> = required.iter().map(|e| e.as_ptr()).collect();
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_array_dynamic_indexing(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
        let mut accel_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let device = instance.create_device(
            physical,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&extension_pointers)
                .enabled_features(&features)
                .push_next(&mut features12)
                .push_next(&mut accel_features)
                .push_next(&mut query_features),
            None,
        )?;
        let queue = device.get_device_queue(family, 0);
        let memory = instance.get_physical_device_memory_properties(physical);
        let accel_api = khr::acceleration_structure::Device::new(&instance, &device);
        let swapchain_api = handles.as_ref().map(|_| khr::swapchain::Device::new(&instance, &device));

        // ------------------------------------------------- commands
        let pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let commands = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        let fence = device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?;
        let image_available = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        let query_pool = device.create_query_pool(
            &vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(TIMESTAMPS),
            None,
        )?;

        // ------------------------------------------------- descriptors and pipelines
        let bindings = [
            layout_binding(0, vk::DescriptorType::UNIFORM_BUFFER, 1),
            layout_binding(1, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, 1),
            layout_binding(2, vk::DescriptorType::STORAGE_BUFFER, 1),
            layout_binding(3, vk::DescriptorType::STORAGE_BUFFER, 1),
            layout_binding(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1),
            layout_binding(5, vk::DescriptorType::STORAGE_IMAGE, IMAGE_COUNT as u32),
            layout_binding(6, vk::DescriptorType::STORAGE_IMAGE, 1),
        ];
        let set_layout = device
            .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: IMAGE_COUNT as u32 + 1 },
        ];
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes),
            None,
        )?;
        let set_layouts = [set_layout];
        let descriptor_set = device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool).set_layouts(&set_layouts),
        )?[0];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<Push>() as u32)];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push_ranges),
            None,
        )?;
        let make = |bytes: &[u8]| create_compute_pipeline(&device, pipeline_layout, bytes);
        let pipelines = Pipelines {
            trace: make(include_bytes!("../shaders/spv/trace.spv"))?,
            temporal: make(include_bytes!("../shaders/spv/temporal.spv"))?,
            atrous: make(include_bytes!("../shaders/spv/atrous.spv"))?,
            resolve: make(include_bytes!("../shaders/spv/resolve.spv"))?,
            tonemap: make(include_bytes!("../shaders/spv/tonemap.spv"))?,
        };

        let sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )?;

        let path = TrackPath::new();
        let mesh = build_scene_mesh(&path);
        let triangle_count = (mesh.len() / 3) as u32;
        let simulation = MarbleSimulation::new(config.marbles, &path);

        // The fields below need helper methods, which need `self`. Placeholders are replaced
        // right after, before anything can use them.
        let placeholder_buffer = || Buffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            address: 0,
        };
        let placeholder_image =
            || Image { image: vk::Image::null(), memory: vk::DeviceMemory::null(), view: vk::ImageView::null() };
        let placeholder_accel = || Accel {
            handle: vk::AccelerationStructureKHR::null(),
            buffer: placeholder_buffer(),
            address: 0,
        };

        let mut renderer = Self {
            _entry: entry,
            instance,
            debug,
            physical,
            device,
            queue,
            memory,
            accel_api,
            scratch_alignment,
            timestamp_period: properties.limits.timestamp_period,
            device_name,
            surface_api,
            swapchain_api,
            surface,
            swapchain: None,
            vsync: config.vsync,
            pool,
            commands,
            fence,
            image_available,
            query_pool,
            timestamps_ready: false,
            last_times: FrameTimes::default(),
            descriptor_pool,
            set_layout,
            descriptor_set,
            pipeline_layout,
            pipelines,
            sampler,
            paper: placeholder_image(),
            vertex_buffer: placeholder_buffer(),
            triangle_count,
            triangle_accel: placeholder_accel(),
            uniforms: placeholder_buffer(),
            instances: placeholder_buffer(),
            spheres: SphereResources {
                spheres: placeholder_buffer(),
                boxes: placeholder_buffer(),
                bottom: placeholder_accel(),
                top: placeholder_accel(),
                scratch: placeholder_buffer(),
            },
            targets: RenderTargets { images: vec![], display: placeholder_image(), width: 0, height: 0 },
            window_width: config.width,
            window_height: config.height,
            path,
            simulation,
            tier: config.tier,
            denoise: config.denoise,
            camera_paused: false,
            close_up: false,
            camera_phase: 0.0,
            frame_index: 0,
            previous_view_proj: None,
            time: 0.0,
        };

        renderer.uniforms = renderer.create_buffer(
            std::mem::size_of::<Uniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            true,
        )?;
        renderer.paper = renderer.create_paper_texture()?;
        renderer.vertex_buffer = renderer.upload_buffer(
            bytemuck::cast_slice(&mesh),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        renderer.triangle_accel = renderer.build_triangle_accel()?;
        renderer.instances = renderer.create_buffer(
            2 * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            true,
        )?;
        renderer.spheres = renderer.create_sphere_resources(renderer.simulation.count())?;
        renderer.write_instances();
        renderer.targets = renderer.create_targets()?;
        renderer.update_descriptors();
        if handles.is_some() {
            renderer.recreate_swapchain()?;
        }
        Ok(renderer)
    }

    // ------------------------------------------------------------------ resource helpers

    fn memory_type(&self, bits: u32, preferred: vk::MemoryPropertyFlags, required: vk::MemoryPropertyFlags) -> Result<u32> {
        let find = |flags: vk::MemoryPropertyFlags| {
            (0..self.memory.memory_type_count).find(|&i| {
                bits & (1 << i) != 0 && self.memory.memory_types[i as usize].property_flags.contains(flags)
            })
        };
        find(preferred).or_else(|| find(required)).ok_or_else(|| "No suitable GPU memory type".into())
    }

    /// Host-visible buffers stay mapped. They use memory the GPU reads fast when the card has it.
    fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags, host_visible: bool) -> Result<Buffer> {
        unsafe {
            let buffer = self
                .device
                .create_buffer(&vk::BufferCreateInfo::default().size(size.max(4)).usage(usage), None)?;
            let needs = self.device.get_buffer_memory_requirements(buffer);
            let (preferred, required) = if host_visible {
                let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
                (host | vk::MemoryPropertyFlags::DEVICE_LOCAL, host)
            } else {
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty())
            };
            let type_index = self.memory_type(needs.memory_type_bits, preferred, required)?;
            let wants_address = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut info = vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(type_index);
            if wants_address {
                info = info.push_next(&mut flags);
            }
            let memory = self.device.allocate_memory(&info, None)?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            let mapped = if host_visible {
                self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? as *mut u8
            } else {
                std::ptr::null_mut()
            };
            let address = if wants_address {
                self.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
            } else {
                0
            };
            Ok(Buffer { buffer, memory, mapped, address })
        }
    }

    fn destroy_buffer(&self, buffer: &Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    /// Runs one-off GPU work and waits for it to finish.
    fn run_once(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        unsafe {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let commands = self.device.allocate_command_buffers(&info)?;
            self.device.begin_command_buffer(
                commands[0],
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(commands[0]);
            // Everything this work wrote must be visible to later frames and to the CPU.
            memory_barrier(
                &self.device,
                commands[0],
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::HOST_READ,
            );
            self.device.end_command_buffer(commands[0])?;
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            self.device.queue_submit(self.queue, &submit, vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            self.device.free_command_buffers(self.pool, &commands);
            Ok(())
        }
    }

    /// Copies data into fast GPU-only memory: geometry is read many times per ray.
    fn upload_buffer(&self, data: &[u8], usage: vk::BufferUsageFlags) -> Result<Buffer> {
        let staging = self.create_buffer(data.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), staging.mapped, data.len()) };
        let buffer = self.create_buffer(data.len() as u64, usage | vk::BufferUsageFlags::TRANSFER_DST, false)?;
        self.run_once(|cmd| unsafe {
            let region = [vk::BufferCopy::default().size(data.len() as u64)];
            self.device.cmd_copy_buffer(cmd, staging.buffer, buffer.buffer, &region);
        })?;
        self.destroy_buffer(&staging);
        Ok(buffer)
    }

    fn create_image(&self, format: vk::Format, width: u32, height: u32, mips: u32, usage: vk::ImageUsageFlags) -> Result<Image> {
        unsafe {
            let image = self.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(mips)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let needs = self.device.get_image_memory_requirements(image);
            let type_index = self.memory_type(
                needs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            )?;
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(needs.size).memory_type_index(type_index),
                None,
            )?;
            self.device.bind_image_memory(image, memory, 0)?;
            let view = self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(color_range(mips)),
                None,
            )?;
            Ok(Image { image, memory, view })
        }
    }

    fn destroy_image(&self, image: &Image) {
        unsafe {
            self.device.destroy_image_view(image.view, None);
            self.device.destroy_image(image.image, None);
            self.device.free_memory(image.memory, None);
        }
    }

    /// Decodes the newspaper page and uploads it with mipmaps averaged on the CPU.
    fn create_paper_texture(&self) -> Result<Image> {
        let decoder = png::Decoder::new(&include_bytes!("../assets/newspaper.png")[..]);
        let mut reader = decoder.read_info()?;
        let mut pixels = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut pixels)?;
        if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
            return Err("assets/newspaper.png must be 8-bit grayscale".into());
        }
        let (mut width, mut height) = (info.width, info.height);
        pixels.truncate((width * height) as usize);

        let mut levels: Vec<(u32, u32, Vec<u8>)> = vec![(width, height, pixels)];
        while width > 1 || height > 1 {
            let (nw, nh) = ((width / 2).max(1), (height / 2).max(1));
            let source = &levels.last().unwrap().2;
            let mut next = vec![0u8; (nw * nh) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    let mut sum = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx).min(width - 1);
                        let sy = (y * 2 + dy).min(height - 1);
                        sum += source[(sy * width + sx) as usize] as u32;
                    }
                    next[(y * nw + x) as usize] = (sum / 4) as u8;
                }
            }
            levels.push((nw, nh, next));
            (width, height) = (nw, nh);
        }

        let mips = levels.len() as u32;
        let image = self.create_image(
            vk::Format::R8_UNORM,
            levels[0].0,
            levels[0].1,
            mips,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )?;
        let total: usize = levels.iter().map(|l| l.2.len()).sum();
        let staging = self.create_buffer(total as u64, vk::BufferUsageFlags::TRANSFER_SRC, true)?;
        let mut regions = Vec::new();
        let mut offset = 0usize;
        for (level, (w, h, data)) in levels.iter().enumerate() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), staging.mapped.add(offset), data.len()) };
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset as u64)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level as u32,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D { width: *w, height: *h, depth: 1 }),
            );
            offset += data.len();
        }
        self.run_once(|cmd| unsafe {
            image_barrier(
                &self.device,
                cmd,
                image.image,
                mips,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
            image_barrier(
                &self.device,
                cmd,
                image.image,
                mips,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
            );
        })?;
        self.destroy_buffer(&staging);
        Ok(image)
    }

    // ---------------------------------------------------------- acceleration structures

    fn create_accel(&self, kind: vk::AccelerationStructureTypeKHR, size: u64) -> Result<Accel> {
        let buffer = self.create_buffer(
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            false,
        )?;
        unsafe {
            let handle = self.accel_api.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::default().buffer(buffer.buffer).size(size).ty(kind),
                None,
            )?;
            let address = self.accel_api.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
            );
            Ok(Accel { handle, buffer, address })
        }
    }

    fn destroy_accel(&self, accel: &Accel) {
        unsafe { self.accel_api.destroy_acceleration_structure(accel.handle, None) };
        self.destroy_buffer(&accel.buffer);
    }

    fn triangle_geometry(&self) -> vk::AccelerationStructureGeometryKHR<'static> {
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: self.vertex_buffer.address })
            .vertex_stride(std::mem::size_of::<Vertex>() as u64)
            .max_vertex(self.triangle_count * 3 - 1)
            .index_type(vk::IndexType::NONE_KHR);
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
    }

    /// Static wood and paper: built once, optimised for trace speed, never touched again.
    fn build_triangle_accel(&self) -> Result<Accel> {
        let geometry = [self.triangle_geometry()];
        let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometry);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.accel_api.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[self.triangle_count],
                &mut sizes,
            );
        }
        let accel = self.create_accel(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, sizes.acceleration_structure_size)?;
        let scratch = self.create_scratch(sizes.build_scratch_size)?;
        info = info
            .dst_acceleration_structure(accel.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: align(scratch.address, self.scratch_alignment) });
        let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(self.triangle_count)];
        self.run_once(|cmd| unsafe {
            self.accel_api.cmd_build_acceleration_structures(cmd, std::slice::from_ref(&info), &[&range]);
        })?;
        self.destroy_buffer(&scratch);
        Ok(accel)
    }

    fn create_scratch(&self, size: u64) -> Result<Buffer> {
        self.create_buffer(
            size + self.scratch_alignment,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            false,
        )
    }

    fn box_geometry(boxes: &Buffer) -> vk::AccelerationStructureGeometryKHR<'static> {
        let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR { device_address: boxes.address })
            .stride(std::mem::size_of::<Aabb>() as u64);
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .flags(vk::GeometryFlagsKHR::OPAQUE)
            .geometry(vk::AccelerationStructureGeometryDataKHR { aabbs })
    }

    fn instance_geometry(instances: &Buffer) -> vk::AccelerationStructureGeometryKHR<'static> {
        let data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR { device_address: instances.address });
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances: data })
    }

    fn create_sphere_resources(&self, count: usize) -> Result<SphereResources> {
        let spheres = self.create_buffer(
            (count * std::mem::size_of::<GpuSphere>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            true,
        )?;
        let boxes = self.create_buffer(
            (count * std::mem::size_of::<Aabb>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            true,
        )?;

        let bottom_geometry = [Self::box_geometry(&boxes)];
        let bottom_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&bottom_geometry);
        let mut bottom_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        let top_geometry = [Self::instance_geometry(&self.instances)];
        let top_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&top_geometry);
        let mut top_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.accel_api.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &bottom_info,
                &[count as u32],
                &mut bottom_sizes,
            );
            self.accel_api.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &top_info,
                &[2],
                &mut top_sizes,
            );
        }
        let bottom =
            self.create_accel(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL, bottom_sizes.acceleration_structure_size)?;
        let top = self.create_accel(vk::AccelerationStructureTypeKHR::TOP_LEVEL, top_sizes.acceleration_structure_size)?;
        let scratch = self.create_scratch(bottom_sizes.build_scratch_size.max(top_sizes.build_scratch_size))?;
        Ok(SphereResources { spheres, boxes, bottom, top, scratch })
    }

    fn destroy_sphere_resources(&self) {
        self.destroy_buffer(&self.spheres.spheres);
        self.destroy_buffer(&self.spheres.boxes);
        self.destroy_accel(&self.spheres.bottom);
        self.destroy_accel(&self.spheres.top);
        self.destroy_buffer(&self.spheres.scratch);
    }

    fn write_instances(&self) {
        let transform = vk::TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        };
        let make = |reference: vk::DeviceAddress| vk::AccelerationStructureInstanceKHR {
            transform,
            instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: reference },
        };
        let items = [make(self.triangle_accel.address), make(self.spheres.bottom.address)];
        unsafe {
            std::ptr::copy_nonoverlapping(
                items.as_ptr() as *const u8,
                self.instances.mapped,
                std::mem::size_of_val(&items),
            );
        }
    }

    // ------------------------------------------------------------ images and descriptors

    fn create_targets(&self) -> Result<RenderTargets> {
        let width = ((self.window_width as f32 * self.tier.scale) as u32).max(16);
        let height = ((self.window_height as f32 * self.tier.scale) as u32).max(16);
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let mut images = Vec::with_capacity(IMAGE_COUNT);
        for _ in 0..IMAGE_COUNT {
            images.push(self.create_image(vk::Format::R16G16B16A16_SFLOAT, width, height, 1, usage)?);
        }
        let display = self.create_image(vk::Format::R8G8B8A8_UNORM, width, height, 1, usage)?;
        // All images stay in GENERAL. Clearing them also means "no history yet" to the shaders.
        self.run_once(|cmd| unsafe {
            for image in images.iter().chain(std::iter::once(&display)) {
                image_barrier(
                    &self.device,
                    cmd,
                    image.image,
                    1,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                self.device.cmd_clear_color_image(
                    cmd,
                    image.image,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { float32: [0.0; 4] },
                    &[color_range(1)],
                );
            }
        })?;
        Ok(RenderTargets { images, display, width, height })
    }

    fn destroy_targets(&self) {
        for image in &self.targets.images {
            self.destroy_image(image);
        }
        self.destroy_image(&self.targets.display);
    }

    fn update_descriptors(&self) {
        let uniform_info = [vk::DescriptorBufferInfo::default().buffer(self.uniforms.buffer).range(vk::WHOLE_SIZE)];
        let vertex_info = [vk::DescriptorBufferInfo::default().buffer(self.vertex_buffer.buffer).range(vk::WHOLE_SIZE)];
        let sphere_info = [vk::DescriptorBufferInfo::default().buffer(self.spheres.spheres.buffer).range(vk::WHOLE_SIZE)];
        let paper_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(self.paper.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let image_info: Vec<_> = self
            .targets
            .images
            .iter()
            .map(|i| vk::DescriptorImageInfo::default().image_view(i.view).image_layout(vk::ImageLayout::GENERAL))
            .collect();
        let display_info = [vk::DescriptorImageInfo::default()
            .image_view(self.targets.display.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let accels = [self.spheres.top.handle];
        let mut accel_info =
            vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&accels);
        let set = self.descriptor_set;
        let write = |binding: u32, kind: vk::DescriptorType| {
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(kind)
        };
        let mut accel_write = write(1, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut accel_info);
        accel_write.descriptor_count = 1;
        let writes = [
            write(0, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&uniform_info),
            accel_write,
            write(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&vertex_info),
            write(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&sphere_info),
            write(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&paper_info),
            write(5, vk::DescriptorType::STORAGE_IMAGE).image_info(&image_info),
            write(6, vk::DescriptorType::STORAGE_IMAGE).image_info(&display_info),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    // ------------------------------------------------------------------------ swapchain

    fn recreate_swapchain(&mut self) -> Result<()> {
        let Some(api) = self.swapchain_api.clone() else { return Ok(()) };
        unsafe {
            let caps = self.surface_api.get_physical_device_surface_capabilities(self.physical, self.surface)?;
            if !caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST) {
                return Err("The window surface does not allow copying to it".into());
            }
            let formats = self.surface_api.get_physical_device_surface_formats(self.physical, self.surface)?;
            // A UNORM format: the tonemap pass has already applied the sRGB curve.
            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
                .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
                .copied()
                .ok_or("The window surface has no 8-bit UNORM format")?;
            let modes = self.surface_api.get_physical_device_surface_present_modes(self.physical, self.surface)?;
            let present_mode = if !self.vsync && modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            };
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: self.window_width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: self.window_height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            let mut count = caps.min_image_count + 1;
            if caps.max_image_count > 0 {
                count = count.min(caps.max_image_count);
            }
            let old = self.swapchain.as_ref().map_or(vk::SwapchainKHR::null(), |s| s.handle);
            let handle = api.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface)
                    .min_image_count(count)
                    .image_format(format.format)
                    .image_color_space(format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(caps.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(present_mode)
                    .clipped(true)
                    .old_swapchain(old),
                None,
            )?;
            if let Some(previous) = self.swapchain.take() {
                for semaphore in previous.render_finished {
                    self.device.destroy_semaphore(semaphore, None);
                }
                api.destroy_swapchain(previous.handle, None);
            }
            let images = api.get_swapchain_images(handle)?;
            let mut render_finished = Vec::new();
            for _ in &images {
                render_finished.push(self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            }
            self.swapchain = Some(Swapchain { handle, images, extent, render_finished });
            Ok(())
        }
    }

    /// Call when the window size changes. A size of zero (minimised) is ignored.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        unsafe { self.device.device_wait_idle()? };
        self.window_width = width;
        self.window_height = height;
        self.recreate_swapchain()?;
        self.recreate_targets()
    }

    fn recreate_targets(&mut self) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        self.destroy_targets();
        self.targets = self.create_targets()?;
        self.update_descriptors();
        self.previous_view_proj = None;
        Ok(())
    }

    pub fn set_tier(&mut self, tier: QualityTier) -> Result<()> {
        let rescale = tier.scale != self.tier.scale;
        self.tier = tier;
        if rescale {
            self.recreate_targets()?;
        }
        Ok(())
    }

    pub fn set_marble_count(&mut self, count: usize) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        self.destroy_sphere_resources();
        self.simulation = MarbleSimulation::new(count, &self.path);
        self.spheres = self.create_sphere_resources(count)?;
        self.write_instances();
        self.update_descriptors();
        Ok(())
    }

    pub fn render_size(&self) -> (u32, u32) {
        (self.targets.width, self.targets.height)
    }

    pub fn triangle_count(&self) -> u32 {
        self.triangle_count
    }

    pub fn last_times(&self) -> FrameTimes {
        self.last_times
    }

    // ---------------------------------------------------------------------------- frame

    /// Waits for the previous frame, then reads back how long it took on the GPU.
    fn wait_for_previous_frame(&mut self) -> Result<()> {
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            if self.timestamps_ready {
                let mut stamps = [0u64; TIMESTAMPS as usize];
                self.device.get_query_pool_results(
                    self.query_pool,
                    0,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
                let ms = |a: usize, b: usize| {
                    stamps[b].wrapping_sub(stamps[a]) as f32 * self.timestamp_period / 1_000_000.0
                };
                self.last_times = FrameTimes { build_ms: ms(0, 1), trace_ms: ms(1, 2), post_ms: ms(2, 3) };
            }
        }
        Ok(())
    }

    fn update_scene(&mut self, dt: f32) {
        self.simulation.begin_frame();
        self.simulation.step(dt);
        let count = self.simulation.count();
        unsafe {
            let spheres = std::slice::from_raw_parts_mut(self.spheres.spheres.mapped as *mut GpuSphere, count);
            let boxes = std::slice::from_raw_parts_mut(self.spheres.boxes.mapped as *mut Aabb, count);
            self.simulation.write_gpu(spheres, boxes);
        }

        // Swing around the lit side. Behind the track the key light is out of sight,
        // the marbles lose their highlights and stop reading as glass.
        if !self.camera_paused {
            self.camera_phase += 0.12 * dt;
        }
        self.time += dt;
        let (target, eye) = if self.close_up {
            // Low over the newspaper, to see the print through the marbles rolling across it.
            let sway = 0.12 * self.camera_phase.sin();
            let target = Vec3::new(layout::PAPER_CENTER.x, 0.12, layout::PAPER_CENTER.y);
            (target, target + Vec3::new(-1.45 + sway, 0.62, -0.75 - sway))
        } else {
            let angle = 0.6 + 0.6 * self.camera_phase.sin();
            (Vec3::new(0.0, 0.95, 0.0), Vec3::new(angle.cos() * 5.8, 3.8, angle.sin() * 5.8))
        };
        let forward = (target - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let aspect = self.targets.width as f32 / self.targets.height as f32;
        let tan_half = (38.0_f32.to_radians() / 2.0).tan();
        let view_proj = perspective(tan_half, aspect, 0.1, 200.0) * look_at(eye, right, up, forward);
        let jitter = Vec2::new(halton(self.frame_index % 32 + 1, 2) - 0.5, halton(self.frame_index % 32 + 1, 3) - 0.5);
        let across = layout::PAPER_ACROSS / layout::PAPER_HALF_WIDTH;
        let paper_up = layout::PAPER_UP / layout::PAPER_HALF_HEIGHT;
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array(),
            prev_view_proj: self.previous_view_proj.unwrap_or(view_proj).to_cols_array(),
            cam_pos: eye.extend(0.0).to_array(),
            cam_right: right.extend(aspect).to_array(),
            cam_up: up.extend(tan_half).to_array(),
            cam_forward: forward.extend(0.0).to_array(),
            paper_origin: [layout::PAPER_CENTER.x, layout::PAPER_CENTER.y, 0.0, 0.0],
            paper_axes: Vec4::new(across.x, across.y, paper_up.x, paper_up.y).to_array(),
            jitter: if self.denoise { jitter.to_array() } else { [0.0, 0.0] },
            size: [self.targets.width, self.targets.height],
            frame: self.frame_index,
            max_bounces: self.tier.bounces,
            diffuse_bounces: self.tier.diffuse_bounces,
            sphere_count: count as u32,
            time: self.time,
            pad: [0.0; 3],
        };
        self.previous_view_proj = Some(view_proj);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytemuck::bytes_of(&uniforms).as_ptr(),
                self.uniforms.mapped,
                std::mem::size_of::<Uniforms>(),
            );
        }
    }

    unsafe fn record_passes(&self, cmd: vk::CommandBuffer) {
        let device = &self.device;
        device.cmd_reset_query_pool(cmd, self.query_pool, 0, TIMESTAMPS);
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, 0);

        // Host writes (marbles, uniforms) must be visible to the build and to the shaders.
        memory_barrier(
            device,
            cmd,
            vk::PipelineStageFlags::HOST,
            vk::AccessFlags::HOST_WRITE,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ | vk::AccessFlags::UNIFORM_READ,
        );

        // ------------------------------------------- marbles: bottom level, then top level
        let count = self.simulation.count() as u32;
        let scratch = vk::DeviceOrHostAddressKHR { device_address: align(self.spheres.scratch.address, self.scratch_alignment) };
        let bottom_geometry = [Self::box_geometry(&self.spheres.boxes)];
        let bottom = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.spheres.bottom.handle)
            .geometries(&bottom_geometry)
            .scratch_data(scratch);
        let bottom_range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
        self.accel_api.cmd_build_acceleration_structures(cmd, std::slice::from_ref(&bottom), &[&bottom_range]);
        let build_stage = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR;
        let build_access =
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR;
        memory_barrier(device, cmd, build_stage, build_access, build_stage, build_access);

        let top_geometry = [Self::instance_geometry(&self.instances)];
        let top = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(self.spheres.top.handle)
            .geometries(&top_geometry)
            .scratch_data(scratch);
        let top_range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(2)];
        self.accel_api.cmd_build_acceleration_structures(cmd, std::slice::from_ref(&top), &[&top_range]);
        memory_barrier(
            device,
            cmd,
            build_stage,
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_COMMANDS, self.query_pool, 1);

        // ----------------------------------------------------------------- compute passes
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            0,
            &[self.descriptor_set],
            &[],
        );
        let denoise = self.denoise as i32;
        let groups = (self.targets.width.div_ceil(8), self.targets.height.div_ceil(8));
        let run = |pipeline: vk::Pipeline, push: Push| {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
            device.cmd_dispatch(cmd, groups.0, groups.1, 1);
            compute_barrier(device, cmd);
        };
        let push = |source: i32, target: i32, step_size: i32| Push { source, target, step_size, denoise };

        run(self.pipelines.trace, push(0, 0, 0));
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::ALL_COMMANDS, self.query_pool, 2);
        if self.denoise {
            run(self.pipelines.temporal, push(0, 0, 0));
            run(self.pipelines.atrous, push(IMAGE_HISTORY_NEW as i32, IMAGE_FILTER_A, 1));
            run(self.pipelines.atrous, push(IMAGE_FILTER_A, IMAGE_FILTER_B, 2));
            run(self.pipelines.atrous, push(IMAGE_FILTER_B, IMAGE_FILTER_A, 4));
        }
        run(self.pipelines.resolve, push(0, 0, 0));
        run(self.pipelines.tonemap, push(0, 0, 0));

        // This frame becomes the next frame's history.
        memory_barrier(
            device,
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
        );
        let extent = vk::Extent3D { width: self.targets.width, height: self.targets.height, depth: 1 };
        let region = [vk::ImageCopy::default()
            .src_subresource(color_layers())
            .dst_subresource(color_layers())
            .extent(extent)];
        for (from, to) in [
            (IMAGE_NORMAL_DEPTH, IMAGE_PREV_NORMAL_DEPTH),
            (IMAGE_HISTORY_NEW, IMAGE_HISTORY_OLD),
            (IMAGE_TAA_NEW, IMAGE_TAA_OLD),
        ] {
            device.cmd_copy_image(
                cmd,
                self.targets.images[from].image,
                vk::ImageLayout::GENERAL,
                self.targets.images[to].image,
                vk::ImageLayout::GENERAL,
                &region,
            );
        }
        memory_barrier(
            device,
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ,
        );
    }

    /// Renders one frame without a window, as fast as the GPU allows.
    pub fn render_headless(&mut self, dt: f32) -> Result<()> {
        self.wait_for_previous_frame()?;
        self.update_scene(dt);
        unsafe {
            self.device.reset_fences(&[self.fence])?;
            let cmd = self.commands;
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.record_passes(cmd);
            self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, 3);
            self.device.end_command_buffer(cmd)?;
            let buffers = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
            self.device.queue_submit(self.queue, &submit, self.fence)?;
            self.timestamps_ready = true;
        }
        Ok(())
    }

    /// Renders one frame and presents it. A stale swapchain is rebuilt, and that frame is skipped.
    pub fn render_to_window(&mut self, dt: f32) -> Result<()> {
        self.wait_for_previous_frame()?;
        let api = self.swapchain_api.clone().ok_or("This renderer has no window")?;
        let (handle, extent) = {
            let swapchain = self.swapchain.as_ref().ok_or("This renderer has no window")?;
            (swapchain.handle, swapchain.extent)
        };
        unsafe {
            let (index, _) = match api.acquire_next_image(handle, u64::MAX, self.image_available, vk::Fence::null()) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.device.device_wait_idle()?;
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(error) => return Err(error.into()),
            };
            self.update_scene(dt);
            self.device.reset_fences(&[self.fence])?;
            let cmd = self.commands;
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.record_passes(cmd);

            let swapchain = self.swapchain.as_ref().unwrap();
            let target = swapchain.images[index as usize];
            image_barrier(
                &self.device,
                cmd,
                target,
                1,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
            );
            let corner = |x: u32, y: u32| vk::Offset3D { x: x as i32, y: y as i32, z: 1 };
            let origin = vk::Offset3D { x: 0, y: 0, z: 0 };
            let blit = [vk::ImageBlit::default()
                .src_subresource(color_layers())
                .src_offsets([origin, corner(self.targets.width, self.targets.height)])
                .dst_subresource(color_layers())
                .dst_offsets([origin, corner(extent.width, extent.height)])];
            self.device.cmd_blit_image(
                cmd,
                self.targets.display.image,
                vk::ImageLayout::GENERAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blit,
                vk::Filter::LINEAR,
            );
            image_barrier(
                &self.device,
                cmd,
                target,
                1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
            );
            self.device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, 3);
            self.device.end_command_buffer(cmd)?;

            let wait = [self.image_available];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let buffers = [cmd];
            let signal = [swapchain.render_finished[index as usize]];
            let submit = [vk::SubmitInfo::default()
                .wait_semaphores(&wait)
                .wait_dst_stage_mask(&stages)
                .command_buffers(&buffers)
                .signal_semaphores(&signal)];
            self.device.queue_submit(self.queue, &submit, self.fence)?;
            self.timestamps_ready = true;

            let swapchains = [handle];
            let indices = [index];
            let present = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal)
                .swapchains(&swapchains)
                .image_indices(&indices);
            match api.queue_present(self.queue, &present) {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.device.device_wait_idle()?;
                    self.recreate_swapchain()?;
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    /// Saves the most recently rendered image as a PNG.
    pub fn save_png(&mut self, path: &str) -> Result<()> {
        unsafe { self.device.device_wait_idle()? };
        let (width, height) = self.render_size();
        let staging = self.create_buffer((width * height * 4) as u64, vk::BufferUsageFlags::TRANSFER_DST, true)?;
        self.run_once(|cmd| unsafe {
            let region = [vk::BufferImageCopy::default()
                .image_subresource(color_layers())
                .image_extent(vk::Extent3D { width, height, depth: 1 })];
            self.device.cmd_copy_image_to_buffer(
                cmd,
                self.targets.display.image,
                vk::ImageLayout::GENERAL,
                staging.buffer,
                &region,
            );
        })?;
        let bytes = unsafe { std::slice::from_raw_parts(staging.mapped, (width * height * 4) as usize) };
        let file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(bytes)?;
        self.destroy_buffer(&staging);
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.destroy_targets();
            self.destroy_sphere_resources();
            self.destroy_accel(&self.triangle_accel);
            self.destroy_buffer(&self.vertex_buffer);
            self.destroy_buffer(&self.uniforms);
            self.destroy_buffer(&self.instances);
            self.destroy_image(&self.paper);
            self.device.destroy_sampler(self.sampler, None);
            for pipeline in [
                self.pipelines.trace,
                self.pipelines.temporal,
                self.pipelines.atrous,
                self.pipelines.resolve,
                self.pipelines.tonemap,
            ] {
                self.device.destroy_pipeline(pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_query_pool(self.query_pool, None);
            self.device.destroy_semaphore(self.image_available, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            if let (Some(api), Some(swapchain)) = (&self.swapchain_api, self.swapchain.take()) {
                for semaphore in swapchain.render_finished {
                    self.device.destroy_semaphore(semaphore, None);
                }
                api.destroy_swapchain(swapchain.handle, None);
            }
            self.device.destroy_device(None);
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_api.destroy_surface(self.surface, None);
            }
            if let Some((api, messenger)) = self.debug.take() {
                api.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// --------------------------------------------------------------------------- free helpers

fn layout_binding(binding: u32, kind: vk::DescriptorType, count: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(kind)
        .descriptor_count(count)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn create_compute_pipeline(device: &ash::Device, layout: vk::PipelineLayout, spirv: &[u8]) -> Result<vk::Pipeline> {
    let code = ash::util::read_spv(&mut std::io::Cursor::new(spirv))?;
    unsafe {
        let module = device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let result = device.create_compute_pipelines(vk::PipelineCache::null(), &info, None);
        device.destroy_shader_module(module, None);
        match result {
            Ok(pipelines) => Ok(pipelines[0]),
            Err((_, error)) => Err(error.into()),
        }
    }
}

fn color_range(mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mips,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn color_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }
}

fn align(address: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    address.div_ceil(alignment) * alignment
}

unsafe fn memory_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stage: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = [vk::MemoryBarrier::default().src_access_mask(src_access).dst_access_mask(dst_access)];
    device.cmd_pipeline_barrier(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), &barrier, &[], &[]);
}

unsafe fn compute_barrier(device: &ash::Device, cmd: vk::CommandBuffer) {
    memory_barrier(
        device,
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
    );
}

#[allow(clippy::too_many_arguments)]
unsafe fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    mips: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) {
    let barrier = [vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(color_range(mips))];
    device.cmd_pipeline_barrier(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[], &barrier);
}

fn halton(index: u32, base: u32) -> f32 {
    let (mut f, mut result, mut i) = (1.0, 0.0, index);
    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }
    result
}

fn look_at(eye: Vec3, right: Vec3, up: Vec3, forward: Vec3) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(right.x, up.x, -forward.x, 0.0),
        Vec4::new(right.y, up.y, -forward.y, 0.0),
        Vec4::new(right.z, up.z, -forward.z, 0.0),
        Vec4::new(-right.dot(eye), -up.dot(eye), forward.dot(eye), 1.0),
    )
}

/// Right-handed perspective projection with clip-space depth in [0, 1]. Only used for motion
/// vectors: rays are generated directly from the camera axes.
fn perspective(tan_half_fov: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let y = 1.0 / tan_half_fov;
    let x = y / aspect;
    let z = far / (near - far);
    Mat4::from_cols(
        Vec4::new(x, 0.0, 0.0, 0.0),
        Vec4::new(0.0, y, 0.0, 0.0),
        Vec4::new(0.0, 0.0, z, -1.0),
        Vec4::new(0.0, 0.0, z * near, 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_match_the_std140_block() {
        // Offsets from common.glsl: 128 for matrices, 96 for vec4s, then vec2, uvec2, 4 uints, float.
        assert_eq!(std::mem::offset_of!(Uniforms, jitter), 224);
        assert_eq!(std::mem::offset_of!(Uniforms, size), 232);
        assert_eq!(std::mem::offset_of!(Uniforms, frame), 240);
        assert_eq!(std::mem::offset_of!(Uniforms, time), 256);
    }

    #[test]
    fn camera_projects_the_look_target_to_the_centre() {
        let eye = Vec3::new(4.0, 3.0, 2.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let forward = (target - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let m = perspective(0.34, 1.6, 0.1, 200.0) * look_at(eye, right, up, forward);
        let clip = m * target.extend(1.0);
        assert!((clip.x / clip.w).abs() < 1e-4 && (clip.y / clip.w).abs() < 1e-4);
        // A point above the target must land higher in clip space, or motion vectors flip.
        let above = m * (target + Vec3::Y * 0.5).extend(1.0);
        assert!(above.y / above.w > 0.0);
    }
}
