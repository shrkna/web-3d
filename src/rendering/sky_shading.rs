use crate::engine::{self, define};
use crate::rendering::webgpu::{WebGPUInterface, WebGPUUniqueResources};
use crate::Shared;
use wasm_bindgen::JsCast;

// sky shading pass --------------------------------------------------------------------------------------

pub fn sky_pass(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    command_encoder: &mut wgpu::CommandEncoder,
    _view: &wgpu::TextureView,
    global_resources: &mut WebGPUUniqueResources,
) {
    // Create sky shader resource and convert HDR to cube texture on the first update
    let is_first_update = global_resources.sky_shading_resource.is_none();
    if is_first_update {
        global_resources.sky_shading_resource = Some(create_sky_shader_resource(&interface));

        // Convert hdr to cube pass
        {
            let mut convert_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("HDR Conversion Pass"),
                    timestamp_writes: None,
                });

            convert_pass.set_pipeline(
                &global_resources
                    .sky_shading_resource
                    .as_ref()
                    .unwrap()
                    .hdr_convert_pipeline,
            );
            convert_pass.set_bind_group(
                0,
                &global_resources
                    .sky_shading_resource
                    .as_ref()
                    .unwrap()
                    .hdr_convert_bind_group,
                &[],
            );

            let workgroup_count = (512 + 15) / 16;
            convert_pass.dispatch_workgroups(workgroup_count, workgroup_count, 6);
        }
    }

    update_sky_shader_resource(
        &interface,
        &scene,
        global_resources.sky_shading_resource.as_ref().unwrap(),
    );

    // Sky render pass
    {
        let mut sky_render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Sky Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &interface
                    .intermediate_texture_2
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        sky_render_pass.set_pipeline(
            &global_resources
                .sky_shading_resource
                .as_ref()
                .unwrap()
                .sky_pipeline,
        );
        sky_render_pass.set_bind_group(
            1,
            &global_resources
                .sky_shading_resource
                .as_ref()
                .unwrap()
                .sky_bind_group,
            &[],
        );

        sky_render_pass.draw(0..3, 0..1); // フルスクリーントライアングルを描画
    }
}

// sky shader resource creation and update functions ----------------------------------------------------------------------------
pub struct WebGPUSkyShadingResource {
    pub _shader: wgpu::ShaderModule,
    pub hdr_convert_pipeline: wgpu::ComputePipeline,
    pub hdr_convert_bind_group: wgpu::BindGroup,
    pub sky_uniform_buffer: wgpu::Buffer,
    pub sky_pipeline: wgpu::RenderPipeline,
    pub sky_bind_group: wgpu::BindGroup,
}

pub struct SkyUniformBuffer {
    pub _inv_view_projection_matrix: [[f32; 4]; 4],
}

fn create_sky_shader_resource(interface: &WebGPUInterface) -> WebGPUSkyShadingResource {
    let shader: wgpu::ShaderModule =
        interface
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../shader/sky.wgsl"
                ))),
            });

    let sky_box_size: u32 = 512;

    let hdr_texture_view = interface.sky_hdr_texture.create_view(&Default::default());

    let sky_cube_texture = interface.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Cube Target"),
        size: wgpu::Extent3d {
            width: sky_box_size,
            height: sky_box_size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float, // HDR精度を維持
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    let hdr_sampler = interface.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    // Convert hdr to cube texture pass

    let cube_storage_view = sky_cube_texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });

    let hdr_convert_bind_group_layout =
        interface
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("HDR Convert Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    },
                ],
            });

    let hdr_convert_bind_group = interface
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &hdr_convert_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&hdr_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&cube_storage_view),
                },
            ],
        });

    let hdr_convert_pipeline_layout =
        interface
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("HDR Convert Pipeline Layout"),
                bind_group_layouts: &[&hdr_convert_bind_group_layout],
                push_constant_ranges: &[],
            });

    let hdr_convert_pipeline =
        interface
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&hdr_convert_pipeline_layout),
                module: &shader,
                entry_point: Some("cs_convert_main"),
                compilation_options: Default::default(),
                cache: None,
            });

    // Render sky pass

    let cube_sample_view = sky_cube_texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("Cube Sample View"),
        dimension: Some(wgpu::TextureViewDimension::Cube), // ここをCubeにする！
        ..Default::default()
    });

    let sky_uniform_buffer = interface.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sky Uniform Buffer"),
        size: std::mem::size_of::<SkyUniformBuffer>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let sky_pipeline = interface
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: interface.intermediate_texture.format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None, // 深度テストはシェーダー内のdiscardで行うため不要
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let sky_bind_group = interface
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Bind Group"),
            layout: &sky_pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sky_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &interface
                            .depth_texture
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &interface
                            .intermediate_texture
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&cube_sample_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&hdr_sampler),
                },
            ],
        });

    return WebGPUSkyShadingResource {
        _shader: shader,
        hdr_convert_pipeline,
        hdr_convert_bind_group,
        sky_uniform_buffer,
        sky_pipeline,
        sky_bind_group,
    };
}

fn update_sky_shader_resource(
    interface: &WebGPUInterface,
    scene: &Shared<engine::scene::Scene>,
    sky_shader_resource: &WebGPUSkyShadingResource,
) {
    let canvas: web_sys::Element = gloo::utils::document()
        .get_element_by_id(define::CANVAS_ELEMENT_ID)
        .unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
    let width: u32 = canvas.client_width() as u32;
    let height: u32 = canvas.client_height() as u32;
    let aspect_ratio: f32 = width as f32 / height as f32;

    let scene_value = scene.borrow();

    let eye: glam::Vec3 = scene_value.parameters.eye_location;
    let direction: glam::Vec3 = scene_value.parameters.eye_direction;

    let inv_view_matrix = glam::Mat4::look_to_rh(eye, direction, glam::Vec3::Z).inverse();
    let inv_projection_matrix: glam::Mat4 =
        glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect_ratio, 0.01, 100.0)
            .inverse();

    let inv_view_projection_matrix = inv_view_matrix * inv_projection_matrix;

    let mut uniform_total: Vec<f32> = Vec::new();
    uniform_total.extend_from_slice(&inv_view_projection_matrix.to_cols_array());

    let uniform_ref: &[f32] = uniform_total.as_ref();
    interface.queue.write_buffer(
        &sky_shader_resource.sky_uniform_buffer,
        0,
        bytemuck::cast_slice(uniform_ref),
    );
}
