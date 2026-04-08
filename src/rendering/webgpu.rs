use crate::engine::{self, define};
use crate::rendering::common;
use crate::types::Shared;
use wasm_bindgen::JsCast;
use wgpu::util::DeviceExt;
use wgpu::TextureViewDescriptor;

const WEBGPU_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const WEBGPU_FRONT_FACE: wgpu::FrontFace = wgpu::FrontFace::Ccw;
const WEBGPU_CULL_MODE: wgpu::Face = wgpu::Face::Back;

// --------------------------------------------------------------------------------------------

/// WebGPU interface contains webgpu contexts such as device, queue, and surface. It is shared between objects.
pub struct WebGPUInterface<'a> {
    pub surface: wgpu::Surface<'a>,
    pub _adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub swapchain_format: wgpu::TextureFormat,
    pub depth_texture: wgpu::Texture,
}

/// shader resource contains shader module and pipeline layout. It is shared between objects.
pub struct WebGPUShaderContext {
    pub _shader_name: std::string::String,
    pub render_pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub _bind_group_layout_2: Option<wgpu::BindGroupLayout>,
    pub uniform_size: u64,
}

/// rendering resource contains GPU buffers and bind groups. It is created for each object.
pub struct WebGPUMeshRenderingResource {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub uniform_buffer: Option<wgpu::Buffer>,
    pub bind_group: Option<wgpu::BindGroup>,
    pub bind_group_2: Option<wgpu::BindGroup>,
    pub _base_color_texture: Option<wgpu::Texture>,
    pub base_color_texture_view: Option<wgpu::TextureView>,
    pub base_color_texture_sampler: Option<wgpu::Sampler>,
    pub _normal_texture: Option<wgpu::Texture>,
    pub normal_texture_view: Option<wgpu::TextureView>,
    pub normal_texture_sampler: Option<wgpu::Sampler>,
    pub _metallic_roughness_texture: Option<wgpu::Texture>,
    pub metallic_roughness_texture_view: Option<wgpu::TextureView>,
    pub metallic_roughness_texture_sampler: Option<wgpu::Sampler>,
}

// Webgpu Contexts -----------------------------------------------------------------------------

pub async fn init_interface<'a>() -> WebGPUInterface<'a> {
    let canvas: web_sys::Element = gloo::utils::document()
        .get_element_by_id(define::CANVAS_ELEMENT_ID)
        .expect("Failed to get canvas element");
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into()
        .expect("Failed to dynamic cast canvas element");

    let width: u32 = canvas.client_width() as u32;
    let height: u32 = canvas.client_height() as u32;

    // Initialize webgpu

    let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let surface_target = wgpu::SurfaceTarget::Canvas(canvas);

    let surface: wgpu::Surface = instance
        .create_surface(surface_target)
        .expect("Failed to create surface from canvas");

    let adapter: wgpu::Adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to request adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        )
        .await
        .expect("Failed to request device");

    let swapchain_capabilities: wgpu::SurfaceCapabilities = surface.get_capabilities(&adapter);
    let swapchain_format: wgpu::TextureFormat = swapchain_capabilities.formats[0];

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        format: swapchain_format,
        width: width,
        height: height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface.configure(&device, &config);

    let depth_texture: wgpu::Texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WEBGPU_DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // Return webgpu resource

    let interface: WebGPUInterface<'_> = WebGPUInterface {
        surface,
        _adapter: adapter,
        device,
        queue,
        swapchain_format,
        depth_texture,
    };

    return interface;
}

pub fn create_shader_context(interface: &WebGPUInterface, name: &str) -> WebGPUShaderContext {
    match name {
        "Phong" => {
            return create_phong_shader_context(interface);
        }
        "DifferedGBuffer" => {
            return create_differed_gbuffer_shader_context(interface);
        }
        _ => {
            return create_phong_shader_context(interface);
        }
    }
}

pub fn create_mesh_rendering_resource(
    interface: &WebGPUInterface,
    mesh: &common::Mesh,
    materials: &Vec<engine::scene::SceneMaterial>,
) -> WebGPUMeshRenderingResource {
    let vertex_data: &Vec<common::Vertex> = &mesh.vertices;
    let index_data: &Vec<u32> = &mesh.indices;

    let vertex_buf: wgpu::Buffer =
        interface
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertex_data),
                usage: wgpu::BufferUsages::VERTEX,
            });

    let index_buf: wgpu::Buffer =
        interface
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&index_data),
                usage: wgpu::BufferUsages::INDEX,
            });

    let index_count: u32 = index_data.len() as u32;

    let mat_idx = mesh.material.unwrap() as usize;
    let mat = materials.get(mat_idx).unwrap();
    let (base_color_texture, base_color_texture_view, base_color_texture_sampler) =
        create_and_write_texture(
            &interface.device,
            &interface.queue,
            "base color texture",
            mat.base_color_texture_size,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &mat.base_color_texture,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

    let (normal_texture, normal_texture_view, normal_texture_sampler) = create_and_write_texture(
        &interface.device,
        &interface.queue,
        "normal texture",
        mat.normal_texture_size,
        wgpu::TextureFormat::Rgba8Unorm,
        &mat.normal_texture,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );

    let (
        metallic_roughness_texture,
        metallic_roughness_texture_view,
        metallic_roughness_texture_sampler,
    ) = create_and_write_texture(
        &interface.device,
        &interface.queue,
        "metallic roughness texture",
        mat.metallic_roughness_texture_size,
        wgpu::TextureFormat::Rgba8Unorm,
        &mat.metallic_roughness_texture,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );

    return WebGPUMeshRenderingResource {
        vertex_buffer: vertex_buf,
        index_buffer: index_buf,
        index_count: index_count,
        uniform_buffer: None,
        bind_group: None,
        bind_group_2: None,
        _base_color_texture: Some(base_color_texture),
        base_color_texture_view: Some(base_color_texture_view),
        base_color_texture_sampler: Some(base_color_texture_sampler),
        _normal_texture: Some(normal_texture),
        normal_texture_view: Some(normal_texture_view),
        normal_texture_sampler: Some(normal_texture_sampler),
        _metallic_roughness_texture: Some(metallic_roughness_texture),
        metallic_roughness_texture_view: Some(metallic_roughness_texture_view),
        metallic_roughness_texture_sampler: Some(metallic_roughness_texture_sampler),
    };
}

/// Main rendering function. It is called in each frame and updates rendering resources and executes render pass.
pub fn update_rendering_main(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    shader_map: &mut std::collections::HashMap<std::string::String, WebGPUShaderContext>,
    _global_resource_map: &mut std::collections::HashMap<
        std::string::String,
        WebGPUMeshRenderingResource,
    >,
) {
    let frame: wgpu::SurfaceTexture = interface
        .surface
        .get_current_texture()
        .expect("Failed to acquire next swap chain texture");

    let view: wgpu::TextureView = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut main_command_encoder: wgpu::CommandEncoder =
        interface
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Main command encoder"),
            });

    let shading_type: engine::scene::ShadingType = scene.borrow().variables.scene_shading_type;

    match shading_type {
        engine::scene::ShadingType::Forward => {
            forwawrd_shading_pass(
                &interface,
                &scene,
                shader_map,
                &mut main_command_encoder,
                &view,
            );
        }

        engine::scene::ShadingType::Differed => {
            differed_shading_pass(
                &interface,
                &scene,
                shader_map,
                &mut main_command_encoder,
                &view,
            );
        }
    }

    line_grid_pass(interface, scene, &mut main_command_encoder, &view);

    interface
        .queue
        .submit(std::iter::once(main_command_encoder.finish()));
    frame.present();
}

// Forward Rendering ---------------------------------------------------------------------------

/// Forward rendering main function. It updates rendering resources and executes render pass for forward shading.
fn forwawrd_shading_pass(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    shader_map: &mut std::collections::HashMap<std::string::String, WebGPUShaderContext>,
    command_encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    // Update phong resources
    {
        let phong_shader_context: &WebGPUShaderContext = &shader_map
            .entry("Phong".to_string())
            .or_insert(create_shader_context(&interface, "Phong"));

        for scene_object in scene.borrow().objects.iter() {
            if scene_object.mesh_rendering_resource.is_some() {
                update_phong_shading_resource(
                    &interface,
                    &phong_shader_context,
                    &scene.clone(),
                    &scene_object,
                );
            }
        }
    }

    // Begin forward rendering pass
    let scene_value = scene.borrow();
    let mut forward_shading_pass: wgpu::RenderPass<'_> =
        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Differed shading pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: scene_value.variables.background_color[0] as f64,
                        g: scene_value.variables.background_color[1] as f64,
                        b: scene_value.variables.background_color[2] as f64,
                        a: scene_value.variables.background_color[3] as f64,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &interface
                    .depth_texture
                    .create_view(&wgpu::TextureViewDescriptor {
                        label: Some("Depth texture view"),
                        format: Some(WEBGPU_DEPTH_FORMAT),
                        aspect: wgpu::TextureAspect::All,
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                    }),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

    // Rendering forward shading
    {
        let phong_shader_context: &WebGPUShaderContext =
            &shader_map.get("Phong").expect("Phong shader is not exist");

        forward_shading_pass.set_pipeline(&phong_shader_context.render_pipeline);

        for object in scene.borrow().objects.iter() {
            if object.mesh_rendering_resource.is_none() {
                continue;
            }

            forward_shading_pass.set_bind_group(
                0,
                &object
                    .mesh_rendering_resource
                    .as_ref()
                    .expect("[draw] Rendering resource is empty")
                    .borrow()
                    .bind_group,
                &[],
            );
            forward_shading_pass.set_bind_group(
                1,
                &object
                    .mesh_rendering_resource
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .bind_group_2,
                &[],
            );
            forward_shading_pass.set_index_buffer(
                object
                    .mesh_rendering_resource
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .index_buffer
                    .slice(..),
                wgpu::IndexFormat::Uint32,
            );
            forward_shading_pass.set_vertex_buffer(
                0,
                object
                    .mesh_rendering_resource
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .vertex_buffer
                    .slice(..),
            );
            forward_shading_pass.draw_indexed(
                0..object
                    .mesh_rendering_resource
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .index_count,
                0,
                0..1,
            );
        }
    };
}

// Phong Shading -----------------------------------------------------------------------------

fn create_phong_shader_context(interface: &WebGPUInterface) -> WebGPUShaderContext {
    struct PhongUniform {
        _transform_matrix: [f32; 16],
        _rotation_matrix: [f32; 16],
        _directional_light: [f32; 4],
        _ambient_light: [f32; 4],
        _inverse_matrix: [f32; 16],
        _buffer_type: [f32; 4],
    }

    let shader: wgpu::ShaderModule =
        interface
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../shader/phong.wgsl"
                ))),
            });

    let vertex_size: usize = std::mem::size_of::<common::Vertex>();
    let vertex_buffer_layout: [wgpu::VertexBufferLayout<'_>; 1] = [wgpu::VertexBufferLayout {
        array_stride: vertex_size as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::size_of::<[f32; 9]>() as u64,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::size_of::<[f32; 7]>() as u64,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::size_of::<[f32; 12]>() as u64,
                shader_location: 3,
            },
        ],
    }];

    let uniform_size: u64 = std::mem::size_of::<PhongUniform>() as u64;
    let bind_group_layout: wgpu::BindGroupLayout =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(uniform_size),
                    },
                    count: None,
                }],
            });

    let bind_group_layout_2 =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("bind_group_layout_2"),
            });

    let pipeline_layout: wgpu::PipelineLayout =
        interface
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout, &bind_group_layout_2],
                push_constant_ranges: &[],
            });

    let render_pipeline: wgpu::RenderPipeline =
        interface
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(define::VS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffer_layout,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(define::FS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    targets: &[Some(interface.swapchain_format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: WEBGPU_FRONT_FACE,
                    cull_mode: Some(WEBGPU_CULL_MODE),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: WEBGPU_DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    return WebGPUShaderContext {
        _shader_name: "Phong".to_string(),
        bind_group_layout: bind_group_layout,
        _bind_group_layout_2: Some(bind_group_layout_2),
        render_pipeline: render_pipeline,
        uniform_size: uniform_size,
    };
}

fn update_phong_shading_resource(
    interface: &WebGPUInterface,
    shader_resource: &WebGPUShaderContext,
    scene: &Shared<engine::scene::Scene>,
    object: &engine::scene::SceneObject,
) {
    if object.mesh_rendering_resource.is_none() {
        log::debug!("[Update] Rendering resource is empty");
        return;
    }

    // Check if uniform buffer and bind groups are initialized
    let mut is_need_initialize: bool = false;
    {
        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .uniform_buffer
            .is_none();

        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .bind_group
            .is_none();

        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .bind_group_2
            .is_none();

        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .uniform_buffer
            .is_some()
            && object
                .mesh_rendering_resource
                .as_ref()
                .unwrap()
                .borrow()
                .uniform_buffer
                .as_ref()
                .unwrap()
                .size()
                != shader_resource.uniform_size;
    }

    // Initialize uniform buffer
    if is_need_initialize {
        let uniform_buf: wgpu::Buffer = interface.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("[Phong] Uniform Buffer"),
            size: shader_resource.uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow_mut()
            .uniform_buffer = Some(uniform_buf);
    }

    // Initialize bind group 1(uniform buffer)
    if is_need_initialize {
        let bind_group: wgpu::BindGroup =
            interface
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &shader_resource.bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: object
                            .mesh_rendering_resource
                            .as_ref()
                            .unwrap()
                            .borrow()
                            .uniform_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    }],
                    label: Some("Bind group 0"),
                });
        object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow_mut()
            .bind_group = Some(bind_group);
    }

    // Initialize bind group 2(textures and samplers)
    if is_need_initialize {
        let bind_group_2 = interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &shader_resource
                    ._bind_group_layout_2
                    .as_ref()
                    .expect("[Update] Bind group layout 2"),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .base_color_texture_view
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .base_color_texture_sampler
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .normal_texture_view
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .normal_texture_sampler
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                ],
                label: Some("texture_bind_group"),
            });

        object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow_mut()
            .bind_group_2 = Some(bind_group_2);
    }

    // Update uniform buffer
    {
        let canvas: web_sys::Element = gloo::utils::document()
            .get_element_by_id(define::CANVAS_ELEMENT_ID)
            .unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
        let width: u32 = canvas.client_width() as u32;
        let height: u32 = canvas.client_height() as u32;
        let aspect_ratio: f32 = width as f32 / height as f32;

        let scene_value = scene.borrow();

        let eye: glam::Vec3 = scene_value.variables.eye_location;
        let direction: glam::Vec3 = scene_value.variables.eye_direction;

        let mut model_matrix = glam::Mat4::from_cols_array_2d(&object.world_transform);

        // Force Y-up to Z-up
        if scene_value.variables.convert_y_to_z {
            let y_to_z_mat: glam::Mat4 = glam::Mat4::from_axis_angle(
                glam::Vec3::new(1.0, 0.0, 0.0),
                std::f32::consts::PI / 2.0,
            );
            model_matrix = y_to_z_mat * model_matrix;
        }

        let view_matrix = glam::Mat4::look_to_rh(eye, direction, glam::Vec3::Z);
        let projection_matrix: glam::Mat4 =
            glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.01, 100.0);
        let transform_matrix: glam::Mat4 = projection_matrix * view_matrix * model_matrix;

        let directional: [f32; 3] = scene_value.variables.directional_light_angle;
        let ambient: [f32; 4] = scene_value.variables.ambient_light_color;
        let inverse_projection: glam::Mat4 = transform_matrix.inverse();

        let rotaton_matrix: glam::Mat4 =
            glam::Mat4::from_quat(model_matrix.to_scale_rotation_translation().1);

        let mut uniform_total: Vec<f32> = transform_matrix.to_cols_array().to_vec();
        uniform_total.extend_from_slice(&rotaton_matrix.to_cols_array().to_vec());
        uniform_total.extend_from_slice(&directional);
        uniform_total.extend_from_slice(&[0.0]); // Padding!
        uniform_total.extend_from_slice(&ambient);
        uniform_total.extend_from_slice(&inverse_projection.to_cols_array().to_vec());
        uniform_total.extend_from_slice(&[
            scene_value.variables.forward_debug_type as f32,
            0.0,
            0.0,
            0.0,
        ]);

        let uniform_ref: &[f32] = uniform_total.as_ref();
        interface.queue.write_buffer(
            &object
                .mesh_rendering_resource
                .as_ref()
                .unwrap()
                .borrow()
                .uniform_buffer
                .as_ref()
                .unwrap(),
            0,
            bytemuck::cast_slice(uniform_ref),
        );
    }
}

// Differed shading -----------------------------------------------------------------------------

/// Differed shading rendering resource
struct WebGPUDifferedShadingResource {
    pub _shader: wgpu::ShaderModule,
    gbuffer_position_texture: wgpu::Texture,
    gbuffer_normal_texture: wgpu::Texture,
    gbuffer_albedo_texture: wgpu::Texture,
    gbuffer_metallic_roughness_texture: wgpu::Texture,
    pub bind_groups: Vec<wgpu::BindGroup>,
    pub uniform_buf: wgpu::Buffer,
    pub render_pipeline: wgpu::RenderPipeline,
    pub debug_pipeline: wgpu::RenderPipeline,
}

struct DifferedUniform {
    _directional_light: [f32; 4],
    _ambient_light: [f32; 4],
    _inverse_matrix: [f32; 16],
    _debug: DifferedDebugUniform,
}

struct DifferedDebugUniform {
    _buffer_type: f32,
    _padding: [f32; 3],
}

/// Differed rendering main function. It updates rendering resources and executes render pass for differed shading.
fn differed_shading_pass(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    shader_map: &mut std::collections::HashMap<std::string::String, WebGPUShaderContext>,
    command_encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    // Update differed gbuffer resources
    {
        let differed_gbuffer_shader: &WebGPUShaderContext = &shader_map
            .entry("DifferedGBuffer".to_string())
            .or_insert(create_shader_context(&interface, "DifferedGBuffer"));

        for scene_object in scene.borrow().objects.iter() {
            if scene_object.mesh_rendering_resource.is_some() {
                update_differed_gbuffer_shading_resource(
                    &interface,
                    &differed_gbuffer_shader,
                    &scene.clone(),
                    &scene_object,
                );
            }
        }
    }

    // Initialize differed shading resources
    let differed_shading_resource: WebGPUDifferedShadingResource =
        create_differed_shading_context(&interface);

    // Update differed shading resources
    {
        update_differed_shading_buffer(&interface, &scene, &differed_shading_resource);
    }

    // Rendering pass
    {
        // Render differed gbuffer writing pass
        {
            let differed_gbuffer_shader_context: &WebGPUShaderContext = &shader_map
                .get("DifferedGBuffer")
                .expect("DifferedGBuffer shader is not exist");

            let mut gbuffer_pass: wgpu::RenderPass<'_> =
                command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Differed gbuffer pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &differed_shading_resource
                                .gbuffer_position_texture
                                .create_view(&TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &differed_shading_resource
                                .gbuffer_normal_texture
                                .create_view(&TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 1.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &differed_shading_resource
                                .gbuffer_albedo_texture
                                .create_view(&TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &differed_shading_resource
                                .gbuffer_metallic_roughness_texture
                                .create_view(&TextureViewDescriptor::default()),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &interface
                            .depth_texture
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

            gbuffer_pass.set_pipeline(&differed_gbuffer_shader_context.render_pipeline);

            for object in scene.borrow().objects.iter() {
                if object.mesh_rendering_resource.is_none() {
                    continue;
                }

                gbuffer_pass.set_bind_group(
                    0,
                    &object
                        .mesh_rendering_resource
                        .as_ref()
                        .unwrap()
                        .borrow()
                        .bind_group,
                    &[],
                );
                if object
                    .mesh_rendering_resource
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .bind_group_2
                    .is_some()
                {
                    gbuffer_pass.set_bind_group(
                        1,
                        &object
                            .mesh_rendering_resource
                            .as_ref()
                            .unwrap()
                            .borrow()
                            .bind_group_2,
                        &[],
                    );
                }
                gbuffer_pass.set_index_buffer(
                    object
                        .mesh_rendering_resource
                        .as_ref()
                        .unwrap()
                        .borrow()
                        .index_buffer
                        .slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                gbuffer_pass.set_vertex_buffer(
                    0,
                    object
                        .mesh_rendering_resource
                        .as_ref()
                        .unwrap()
                        .borrow()
                        .vertex_buffer
                        .slice(..),
                );
                gbuffer_pass.draw_indexed(
                    0..object
                        .mesh_rendering_resource
                        .as_ref()
                        .unwrap()
                        .borrow()
                        .index_count,
                    0,
                    0..1,
                );
            }
        }

        // differed shading pass
        {
            let scene_value = scene.borrow();

            let mut differed_shading_pass: wgpu::RenderPass<'_> = command_encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Differed shading pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: scene_value.variables.background_color[0] as f64,
                                g: scene_value.variables.background_color[1] as f64,
                                b: scene_value.variables.background_color[2] as f64,
                                a: scene_value.variables.background_color[3] as f64,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

            if scene_value.variables.differed_debug_type == 0 {
                differed_shading_pass.set_pipeline(&differed_shading_resource.render_pipeline);
            } else {
                differed_shading_pass.set_pipeline(&differed_shading_resource.debug_pipeline);
            }

            differed_shading_pass.set_bind_group(0, &differed_shading_resource.bind_groups[0], &[]);
            differed_shading_pass.set_bind_group(1, &differed_shading_resource.bind_groups[1], &[]);
            differed_shading_pass.draw(0..6, 0..1);
        }
    }
}

/// Create shader context for differed gbuffer writing shader
fn create_differed_gbuffer_shader_context(interface: &WebGPUInterface) -> WebGPUShaderContext {
    struct WriteGBuffersUniform {
        _model_matrix: [f32; 16],
        _view_matrix: [f32; 16],
        _projection_matrix: [f32; 16],
        _rotation_matrix: [f32; 16],
    }

    let shader: wgpu::ShaderModule =
        interface
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../shader/differed_write_gbuffers.wgsl"
                ))),
            });

    let uniform_size: u64 = std::mem::size_of::<WriteGBuffersUniform>() as u64;

    let uniform_bind_group_layout: wgpu::BindGroupLayout = interface
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let texture_bind_group_layout =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

    let pipeline_layout: wgpu::PipelineLayout =
        interface
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

    let vertex_buffers: [wgpu::VertexBufferLayout<'_>; 1] = [wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<common::Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::size_of::<[f32; 9]>() as u64,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::size_of::<[f32; 7]>() as u64,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: std::mem::size_of::<[f32; 12]>() as u64,
                shader_location: 3,
            },
        ],
    }];

    let render_pipeline: wgpu::RenderPipeline =
        interface
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(define::VS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    buffers: &vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(define::FS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::all(),
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::all(),
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::all(),
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::all(),
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(WEBGPU_CULL_MODE),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: WEBGPU_DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    return WebGPUShaderContext {
        _shader_name: "DifferedGBuffer".to_string(),
        render_pipeline,
        bind_group_layout: uniform_bind_group_layout,
        _bind_group_layout_2: Some(texture_bind_group_layout),
        uniform_size,
    };
}

fn update_differed_gbuffer_shading_resource(
    interface: &WebGPUInterface,
    shader_context: &WebGPUShaderContext,
    scene: &Shared<engine::scene::Scene>,
    object: &engine::scene::SceneObject,
) {
    if object.mesh_rendering_resource.is_none() {
        log::debug!("[Update] Rendering resource is empty");
        return;
    }

    // Check if uniform buffer and bind groups are initialized
    let mut is_need_initialize: bool = false;
    {
        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .uniform_buffer
            .is_none();

        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .bind_group
            .is_none();

        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .bind_group_2
            .is_none();

        is_need_initialize |= object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow()
            .uniform_buffer
            .is_some()
            && object
                .mesh_rendering_resource
                .as_ref()
                .unwrap()
                .borrow()
                .uniform_buffer
                .as_ref()
                .unwrap()
                .size()
                != shader_context.uniform_size;
    }

    // Initialize uniform buffer
    if is_need_initialize {
        let uniform_buf: wgpu::Buffer = interface.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("[Writing GBuffer] Uniform Buffer]"),
            size: shader_context.uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow_mut()
            .uniform_buffer = Some(uniform_buf);
    }

    // Intialize bind group 1 (Uniform)
    if is_need_initialize {
        let bind_group: wgpu::BindGroup =
            interface
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: &shader_context.bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: object
                            .mesh_rendering_resource
                            .as_ref()
                            .unwrap()
                            .borrow()
                            .uniform_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    }],
                    label: Some("Bind group 0"),
                });
        object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow_mut()
            .bind_group = Some(bind_group);
    }

    // Initialize bind group 2 (Textures)
    if is_need_initialize {
        let bind_group_2 = interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &shader_context
                    ._bind_group_layout_2
                    .as_ref()
                    .expect("[Update] Bind group layout 2"),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .base_color_texture_view
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .base_color_texture_sampler
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .normal_texture_view
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .normal_texture_sampler
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .metallic_roughness_texture_view
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(
                            &object
                                .mesh_rendering_resource
                                .as_ref()
                                .unwrap()
                                .borrow()
                                .metallic_roughness_texture_sampler
                                .as_ref()
                                .unwrap(),
                        ),
                    },
                ],
                label: Some("texture_bind_group"),
            });

        object
            .mesh_rendering_resource
            .as_ref()
            .unwrap()
            .borrow_mut()
            .bind_group_2 = Some(bind_group_2);
    }

    // Update uniform buffer
    {
        let canvas: web_sys::Element = gloo::utils::document()
            .get_element_by_id(define::CANVAS_ELEMENT_ID)
            .unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
        let width: u32 = canvas.client_width() as u32;
        let height: u32 = canvas.client_height() as u32;
        let aspect_ratio: f32 = width as f32 / height as f32;

        let scene_value = scene.borrow();
        let eye: glam::Vec3 = scene_value.variables.eye_location;
        let direction: glam::Vec3 = scene_value.variables.eye_direction;

        let mut model_matrix = glam::Mat4::from_cols_array_2d(&object.world_transform);

        // Force Y-up to Z-up
        if scene_value.variables.convert_y_to_z {
            let y_to_z_mat: glam::Mat4 = glam::Mat4::from_axis_angle(
                glam::Vec3::new(1.0, 0.0, 0.0),
                std::f32::consts::PI / 2.0,
            );
            model_matrix = y_to_z_mat * model_matrix;
        }

        // Create matrices and write buffer
        let view_matrix = glam::Mat4::look_to_rh(eye, direction, glam::Vec3::Z);
        let projection_matrix: glam::Mat4 =
            glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.01, 100.0);

        let rotaton_matrix: glam::Mat4 =
            glam::Mat4::from_quat(model_matrix.to_scale_rotation_translation().1);

        let mut uniform_total = model_matrix.to_cols_array().to_vec();
        uniform_total.extend_from_slice(&view_matrix.to_cols_array());
        uniform_total.extend_from_slice(&projection_matrix.to_cols_array());
        uniform_total.extend_from_slice(&rotaton_matrix.to_cols_array());
        let uniform_ref: &[f32] = uniform_total.as_ref();
        interface.queue.write_buffer(
            &object
                .mesh_rendering_resource
                .as_ref()
                .unwrap()
                .borrow()
                .uniform_buffer
                .as_ref()
                .unwrap(),
            0,
            bytemuck::cast_slice(uniform_ref),
        );
    }
}

/// Initialize differed shading resources, including gbuffer textures and pipelines.
fn create_differed_shading_context(interface: &WebGPUInterface) -> WebGPUDifferedShadingResource {
    let shader: wgpu::ShaderModule =
        interface
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../shader/differed.wgsl"
                ))),
            });

    let canvas: web_sys::Element = gloo::utils::document()
        .get_element_by_id(define::CANVAS_ELEMENT_ID)
        .expect("Failed to get canvas element");
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into()
        .expect("Failed to dynamic cast canvas element");

    let width: u32 = canvas.client_width() as u32;
    let height: u32 = canvas.client_height() as u32;

    // Gbuffers

    let gbuffer_position_texture: wgpu::Texture =
        interface.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("position texture"),
            size: wgpu::Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

    let gbuffer_normal_texture: wgpu::Texture =
        interface.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("normal texture"),
            size: wgpu::Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

    let gbuffer_albedo_texture: wgpu::Texture =
        interface.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("albedo texture"),
            size: wgpu::Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

    let gbuffer_metallic_roughness_texture: wgpu::Texture =
        interface.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("metallic roughness texture"),
            size: wgpu::Extent3d {
                width: width,
                height: height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

    // bindings

    let gbuffer_bind_group_layout: wgpu::BindGroupLayout = interface
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

    let gbuffer_bind_group: wgpu::BindGroup =
        interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &gbuffer_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &gbuffer_position_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &gbuffer_normal_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &interface
                                .depth_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            &gbuffer_albedo_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(
                            &gbuffer_metallic_roughness_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                ],
                label: Some("Bind group 0"),
            });

    let uniform_size: u64 = std::mem::size_of::<DifferedUniform>() as u64;
    let uniform_buf: wgpu::Buffer = interface.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Differed uniform buffer"),
        size: uniform_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let uniform_bind_group_layout: wgpu::BindGroupLayout = interface
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let uniform_bind_group: wgpu::BindGroup =
        interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                }],
                label: Some("Bind group 1"),
            });

    // pipeline

    let pipeline_layout: wgpu::PipelineLayout =
        interface
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&gbuffer_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

    let render_pipeline: wgpu::RenderPipeline =
        interface
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(define::VS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(engine::define::FS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    targets: &[Some(interface.swapchain_format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(WEBGPU_CULL_MODE),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    let debug_pipeline: wgpu::RenderPipeline =
        interface
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(define::VS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_debug_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(interface.swapchain_format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(WEBGPU_CULL_MODE),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    let mut bind_groups: Vec<wgpu::BindGroup> = Vec::new();
    bind_groups.push(gbuffer_bind_group);
    bind_groups.push(uniform_bind_group);

    let resource: WebGPUDifferedShadingResource = WebGPUDifferedShadingResource {
        _shader: shader,
        gbuffer_position_texture,
        gbuffer_normal_texture,
        gbuffer_albedo_texture,
        gbuffer_metallic_roughness_texture,
        bind_groups,
        uniform_buf,
        render_pipeline,
        debug_pipeline,
    };

    return resource;
}

fn update_differed_shading_buffer(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    resource: &WebGPUDifferedShadingResource,
) {
    let canvas: web_sys::Element = gloo::utils::document()
        .get_element_by_id(define::CANVAS_ELEMENT_ID)
        .unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
    let width: u32 = canvas.client_width() as u32;
    let height: u32 = canvas.client_height() as u32;
    let aspect_ratio: f32 = width as f32 / height as f32;

    let scene_value = scene.borrow();

    let eye: glam::Vec3 = scene_value.variables.eye_location;
    let direction: glam::Vec3 = scene_value.variables.eye_direction;

    // Create matrices and write buffer
    let view_matrix = glam::Mat4::look_to_rh(eye, direction, glam::Vec3::Z);
    let projection_matrix: glam::Mat4 =
        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.01, 100.0);
    let transform_matrix: glam::Mat4 = projection_matrix * view_matrix;

    let directional: [f32; 3] = scene_value.variables.directional_light_angle;
    let ambient: [f32; 4] = scene_value.variables.ambient_light_color;
    let inverse_projection: glam::Mat4 = transform_matrix.inverse();

    let mut uniform_total: Vec<f32> = Vec::new();
    uniform_total.extend_from_slice(&directional);
    uniform_total.extend_from_slice(&[0.0]); // Padding!
    uniform_total.extend_from_slice(&ambient);
    uniform_total.extend_from_slice(&inverse_projection.to_cols_array().to_vec());
    uniform_total.extend_from_slice(&[
        scene_value.variables.differed_debug_type as f32,
        0.0,
        0.0,
        0.0,
    ]);

    let uniform_ref: &[f32] = uniform_total.as_ref();
    interface
        .queue
        .write_buffer(&resource.uniform_buf, 0, bytemuck::cast_slice(uniform_ref));
}

// [Debug] Line grid shading ------------------------------------------------------------------------------

struct WebGPULineGridShadingResource {
    pub _shader: wgpu::ShaderModule,
    pub screen_texture: wgpu::Texture,
    pub uniform_bind_group: wgpu::BindGroup,
    pub texture_bind_group: wgpu::BindGroup,
    pub uniform_buffer: wgpu::Buffer,
    pub render_pipeline: wgpu::RenderPipeline,
}

struct LineGridUniform {
    _inv_projection_matrix: [f32; 16],
    _inv_view_matrix: [f32; 16],
    _camera_position: [f32; 4],
    _grid_spacing: f32,
    _line_thickness: f32,
    _fade_radius: f32,
    _padding: f32,
}

fn line_grid_pass(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    command_encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    let line_grid_shader_resource: WebGPULineGridShadingResource =
        create_line_grid_shader_resource(&interface);

    update_line_grid_shader_resource(&interface, &scene, &line_grid_shader_resource);

    // Copy the current frame's texture to the line grid shader's screen texture
    command_encoder.copy_texture_to_texture(
        // 送り側（Source）の指定
        wgpu::ImageCopyTexture {
            texture: &interface.surface.get_current_texture().unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        // 受け側（Destination）の指定
        wgpu::ImageCopyTexture {
            texture: &line_grid_shader_resource.screen_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        // コピーするサイズ（Extent）
        wgpu::Extent3d {
            width: line_grid_shader_resource.screen_texture.size().width,
            height: line_grid_shader_resource.screen_texture.size().height,
            depth_or_array_layers: 1,
        },
    );

    let mut line_grid_shading_pass: wgpu::RenderPass<'_> =
        command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Line grid shading pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

    line_grid_shading_pass.set_pipeline(&line_grid_shader_resource.render_pipeline);
    line_grid_shading_pass.set_bind_group(0, &line_grid_shader_resource.uniform_bind_group, &[]);
    line_grid_shading_pass.set_bind_group(1, &line_grid_shader_resource.texture_bind_group, &[]);
    line_grid_shading_pass.draw(0..3, 0..1);
    // Assuming vertex buffer and draw call are set up elsewhere
}

fn create_line_grid_shader_resource(interface: &WebGPUInterface) -> WebGPULineGridShadingResource {
    let shader: wgpu::ShaderModule =
        interface
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../shader/line.wgsl"
                ))),
            });

    let canvas: web_sys::Element = gloo::utils::document()
        .get_element_by_id(define::CANVAS_ELEMENT_ID)
        .expect("Failed to get canvas element");
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into()
        .expect("Failed to dynamic cast canvas element");

    let width: u32 = canvas.client_width() as u32;
    let height: u32 = canvas.client_height() as u32;

    let screen_texture: wgpu::Texture = interface.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Screen texture"),
        size: wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let uniform_size: u64 = std::mem::size_of::<LineGridUniform>() as u64;
    let uniform_buffer: wgpu::Buffer = interface.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Line grid uniform buffer"),
        size: uniform_size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });


    let uniform_bind_group_layout: wgpu::BindGroupLayout = interface
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(uniform_size),
                },
                count: None,
            }],
        });

    let uniform_bind_group: wgpu::BindGroup =
        interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &uniform_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding()
                }],
                label: Some("Line Grid Uniform Bind Group"),
            });


    let texture_bind_group_layout =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Render target texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    // Depth texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

    let texture_bind_group: wgpu::BindGroup =
        interface
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &screen_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &interface
                                .depth_texture
                                .create_view(&wgpu::TextureViewDescriptor::default()),
                        ),
                    },
                ],
                label: Some("Line Grid Texture Bind Group"),
            });

    let pipeline_layout: wgpu::PipelineLayout =
        interface
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

    let render_pipeline: wgpu::RenderPipeline =
        interface
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(define::VS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(define::FS_ENTRY_POINT),
                    compilation_options: Default::default(),
                    targets: &[
                        /*Some(interface.swapchain_format.into())*/
                        Some(wgpu::ColorTargetState {
                            format: interface.swapchain_format.into(),
                            // ここでブレンド設定を行う
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::SrcAlpha, // ソースのアルファ
                                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, // (1 - アルファ)
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: wgpu::PrimitiveState {
                    front_face: WEBGPU_FRONT_FACE,
                    cull_mode: Some(WEBGPU_CULL_MODE),
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

    return WebGPULineGridShadingResource {
        _shader: shader,
        screen_texture: screen_texture,
        uniform_bind_group: uniform_bind_group,
        texture_bind_group: texture_bind_group,
        uniform_buffer: uniform_buffer,
        render_pipeline: render_pipeline,
    };
}

fn update_line_grid_shader_resource(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    line_grid_shader_resource: &WebGPULineGridShadingResource,
) {
    let canvas: web_sys::Element = gloo::utils::document()
        .get_element_by_id(define::CANVAS_ELEMENT_ID)
        .unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
    let width: u32 = canvas.client_width() as u32;
    let height: u32 = canvas.client_height() as u32;
    let aspect_ratio: f32 = width as f32 / height as f32;

    let scene_value = scene.borrow();

    let eye: glam::Vec3 = scene_value.variables.eye_location;
    let direction: glam::Vec3 = scene_value.variables.eye_direction;

    let inv_view_matrix = glam::Mat4::look_to_rh(eye, direction, glam::Vec3::Z).inverse();
    let inv_projection_matrix: glam::Mat4 =
        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.01, 100.0)
            .inverse();

    let camera_position: [f32; 4] = [eye.x, eye.y, eye.z, 1.0];
    let grid_spacing: f32 = 100.0;
    let line_thickness: f32 = 1.0;
    let fade_radius: f32 = 1000.0;

    let mut uniform_total: Vec<f32> = Vec::new();
    uniform_total.extend_from_slice(&inv_projection_matrix.to_cols_array());
    uniform_total.extend_from_slice(&inv_view_matrix.to_cols_array());
    uniform_total.extend_from_slice(&camera_position);
    uniform_total.push(grid_spacing);
    uniform_total.push(line_thickness);
    uniform_total.push(fade_radius);
    uniform_total.push(0.0); // Padding!

    let uniform_ref: &[f32] = uniform_total.as_ref();
    interface.queue.write_buffer(
        &line_grid_shader_resource.uniform_buffer,
        0,
        bytemuck::cast_slice(uniform_ref),
    );
}

// Utils --------------------------------------------------------------------------------------

/// Creates a texture, writes data to it, and returns the texture, its view, and a sampler.
fn create_and_write_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    data: &[u8],
    usage: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * size[0]),
            rows_per_image: Some(size[1]),
        },
        wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    (texture, view, sampler)
}
