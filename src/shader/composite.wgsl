struct VertexOutput 
{
    @builtin(position) position   : vec4<f32>,
    @location(0)       uv         : vec2<f32>
};

struct CompositeUniform
{
    is_use_tone_mapping     : f32,
    is_use_gamma_correction : f32,
}

@group(0) @binding(0) var           t_scene      : texture_2d<f32>;
@group(0) @binding(1) var           s_scene      : sampler;
@group(1) @binding(0) var<uniform>  u_composite  : CompositeUniform;

fn tonemap_aces(x: vec3f) -> vec3f 
{
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@vertex
fn vs_main( @builtin(vertex_index) vertex_index : u32 ) -> VertexOutput
{
    var out: VertexOutput;

    let x = f32(i32(vertex_index & 1u) << 2u) - 1.0;
    let y = f32(i32(vertex_index & 2u) << 1u) - 1.0;
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv       = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return out;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f 
{
    // 中間テクスチャの色をそのままサンプリング
    let color = textureSample(t_scene, s_scene, uv);
    
    // トーンマッピング
    var mapped = mix(color.xyz, tonemap_aces(color.xyz), u_composite.is_use_tone_mapping);

    // ガンマ補正 (Gamma Correction)
    let final_result = mix(mapped, pow(mapped, vec3f(1.0 / 2.2)), u_composite.is_use_gamma_correction);

    return vec4(final_result, 1.0);
}