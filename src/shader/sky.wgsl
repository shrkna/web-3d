// convert compute shader

@group(0) @binding(0) var t_hdr: texture_2d<f32>;
@group(0) @binding(1) var s_hdr: sampler;
@group(0) @binding(2) var t_cube: texture_storage_2d_array<rgba16float, write>;

// sky pass

struct VertexOutput 
{
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(1) @binding(0) var<uniform>  inv_view_proj: mat4x4f;   // ViewProjection行列の逆行列
@group(1) @binding(1) var t_depth:  texture_depth_2d;         // 既存の深度バッファ
@group(1) @binding(2) var t_screen: texture_2d<f32>;		  // 既存のスクリーン
@group(1) @binding(3) var t_skybox: texture_cube<f32>;        // 変換済みのキューブマップ
@group(1) @binding(4) var s_skybox: sampler;                  // サンプラー


const PI: f32 = 3.14159265359;

@compute @workgroup_size(16, 16, 1)
fn cs_convert_main(@builtin(global_invocation_id) id: vec3u)
{
	let size = textureDimensions(t_cube);
    if (id.x >= size.x || id.y >= size.y) { return; }

    // 0.5 を足してピクセル中心を指すようにする
    let uv = (vec2f(id.xy) + 0.5) / vec2f(size.xy) * 2.0 - 1.0;
    
    let face = id.z;
    var dir: vec3f;
    
    // Z-up 右手系基準のベクトル構築
    switch (face) 
    {
        case 0u: { dir = vec3f( 1.0, -uv.y, -uv.x); } // +X (Right)
        case 1u: { dir = vec3f(-1.0, -uv.y,  uv.x); } // -X (Left)
        case 2u: { dir = vec3f( uv.x,  1.0, uv.y); } // +Y (Forward)
        case 3u: { dir = vec3f(uv.x, -1.0, -uv.y); } // -Y (Back)
        case 4u: { dir = vec3f( uv.x, -uv.y,  1.0); } // +Z (Up)
        case 5u: { dir = vec3f(-uv.x, -uv.y, -1.0); } // -Z (Down)
        default: { dir = vec3f(0.0); }
    }
    
    let n_dir = normalize(dir);

    // 経度: -PI ~ PI -> 0.0 ~ 1.0
    let phi = atan2(n_dir.y, n_dir.x);
    // 緯度: 0(上) ~ PI(下) -> 0.0 ~ 1.0
    let theta = acos(n_dir.z); 

    let pano_uv = vec2f(
        (phi / (2.0 * 3.14159265)) + 0.5,
        theta / 3.14159265
    );

    let color = textureSampleLevel(t_hdr, s_hdr, pano_uv, 0.0);
    textureStore(t_cube, id.xy, face, color);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput 
{
    // 画面全体を覆う三角形の頂点データ
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );
    var out: VertexOutput;
    let p = pos[vertex_index];
    out.position = vec4f(p, 0.0, 1.0);
    out.uv = p * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y; // Y軸反転対応
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f 
{
	// スクリーンのピクセルを取得
	var final_color = textureLoad(t_screen, vec2u(in.position.xy), 0);

    // 現在のピクセルの深度を取得
    let depth = textureLoad(t_depth, vec2u(in.position.xy), 0);
	
    // NDC空間 (x: -1~1, y: -1~1, z: depth) からワールド空間の方向を復元
    let ndc = vec4f(
        in.uv.x * 2.0 - 1.0,
        (1.0 - in.uv.y) * 2.0 - 1.0,
        depth,
        1.0
    );

    let world_pos_h = inv_view_proj * ndc;
    let world_dir   = normalize(world_pos_h.xyz / world_pos_h.w);
	let sky_color   = textureSample(t_skybox, s_skybox, world_dir);
    
	return mix(final_color, sky_color, floor(depth));
}