struct VertexOutput {
    @builtin(position) position   : vec4<f32>,
    @location(0)       uv         : vec2<f32>
};


@group(0) @binding(0) var intermediate_texture      : texture_2d<f32>;
@group(0) @binding(1) var intermediate_sampler      : sampler;

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
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    // 中間テクスチャの色をそのままサンプリングして出力
    let color = textureSample(intermediate_texture, intermediate_sampler, uv);
    
    // ここでトーンマップなどの処理を挟むことが多い
    // let mapped = color.rgb / (color.rgb + vec3f(1.0));
    // return vec4f(mapped, color.a);

    return color;
}