#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 pos   [[attribute(0)]];
    float2 uv    [[attribute(1)]];
    float4 color [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

struct InstanceConstants {
    float4x4 view;
    float4x4 model;
    float4x4 projection;
    packed_float3 camera_pos;
    float time;
    float2 screen; // used for screen_size when rendering post effects
    float2 fade;
};

[[vertex]]
VertexOut vertex_main(VertexIn in [[stage_in]],
                      constant InstanceConstants &instance [[buffer(1)]])
{
    VertexOut out;
    out.position = instance.projection * instance.view * instance.model * float4(in.pos, 1.0f);
    out.position.xy += instance.screen.xy * out.position.w;
    out.color = in.color;
    out.color.a *= smoothstep(
        instance.fade.y, instance.fade.x, // fadeout far, near
        length(float4(instance.camera_pos, 1.0f) - instance.model * float4(in.pos, 1.0f))
    );
    out.uv = in.uv / 2048.0f; // ATLAS_GRID * ATLAS_SIZE

    out.position.z = out.position.z * 0.5 + out.position.w * 0.5; // Reshape GL clip space to Metal
    return out;
}

[[fragment]]
float4 fragment_main(VertexOut in [[stage_in]],
                     texture2d<float, access::sample> tex [[texture(0)]])
{
    constexpr sampler atlas_sampler(coord::normalized,
                                    mag_filter::nearest, min_filter::linear, mip_filter::none,
                                    address::clamp_to_edge);
    float4 tex_color = tex.sample(atlas_sampler, in.uv);
    float4 color = tex_color * in.color;
    if (color.a == 0.0f) {
        discard_fragment();
    }
    color.rgb = color.rgb * 2.0f;
    return color;
}

[[vertex]]
VertexOut vertex_post(VertexIn in [[stage_in]], constant InstanceConstants &instance [[buffer(1)]])
{
    VertexOut out;
    out.position = instance.projection * float4(in.pos, 1.0f);
    out.uv = in.uv;
    out.color = float4(1, 1, 1, 1);

    out.uv.y = 1 - out.uv.y; // Flip from GL to Metal
    out.position.z = out.position.z * 0.5 + out.position.w * 0.5; // Reshape GL clip space to Metal
    return out;
}

// Emulation of GLSL's mod in MSL; distinct from MSL's fmod
static float mod(float x, float y) {
    return x - y * floor(x / y);
}

[[fragment]]
float4 fragment_post(VertexOut in [[stage_in]],
                     texture2d<float, access::sample> tex [[texture(0)]])
{
    constexpr sampler bilinearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float4 color = tex.sample(bilinearSampler, in.uv);
    return color;
}

// CRT effect based on https://www.shadertoy.com/view/Ms23DR
// by https://github.com/mattiasgustavsson/

static float2 curve(float2 uv) {
    uv = (uv - 0.5) * 2.0;
    uv *= 1.1;
    uv.x *= 1.0 + powr((abs(uv.y) / 5.0), 2.0);
    uv.y *= 1.0 + powr((abs(uv.x) / 4.0), 2.0);
    uv  = (uv / 2.0) + 0.5;
    uv =  uv *0.92 + 0.04;
    return uv;
}

[[fragment]]
float4 fragment_post_crt(VertexOut in [[stage_in]],
                         constant InstanceConstants &instance [[buffer(0)]],
                         texture2d<float, access::sample> tex [[texture(0)]])
{
    constexpr sampler bilinearSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    float2 screen_size = instance.screen;
    float2 uv = curve(in.position.xy / screen_size);
    float3 color;
    float x = sin(0.3 * instance.time + uv.y * 21.0) *
              sin(0.7 * instance.time + uv.y * 29.0) *
              sin(0.3 + 0.33 * instance.time+uv.y * 31.0) * 0.0017;

    color.r = tex.sample(bilinearSampler, float2(x+uv.x + 0.001, uv.y + 0.001)).x + 0.05;
    color.g = tex.sample(bilinearSampler, float2(x+uv.x + 0.000, uv.y - 0.002)).y + 0.05;
    color.b = tex.sample(bilinearSampler, float2(x+uv.x - 0.002, uv.y + 0.000)).z + 0.05;
    color.r += 0.08 * tex.sample(bilinearSampler, 0.75 * float2(x+0.025, -0.027) + float2(uv.x + 0.001, uv.y + 0.001)).x;
    color.g += 0.05 * tex.sample(bilinearSampler, 0.75 * float2(x+-0.022, -0.02) + float2(uv.x + 0.000, uv.y - 0.002)).y;
    color.b += 0.08 * tex.sample(bilinearSampler, 0.75 * float2(x+-0.02, -0.018) + float2(uv.x - 0.002, uv.y + 0.000)).z;

    color = clamp(color * 0.6 + 0.4 * color * color * 1.0, 0.0, 1.0);

    float vignette = (0.0 + 1.0 * 16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y));
    color *= float3(powr(vignette, 0.25));

    color *= float3(0.95,1.05,0.95);
    color *= 2.8;

    float scanlines = clamp(0.35 + 0.35 * sin(3.5 * instance.time + uv.y * screen_size.y * 1.5), 0.0, 1.0);

    float s = powr(scanlines, 1.7);
    color = color * float3(0.4 + 0.7 * s);

    color *= 1.0 + 0.01 * sin(110.0 * instance.time);
    if (uv.x < 0.0 || uv.x > 1.0)
        color *= 0.0;
    if (uv.y < 0.0 || uv.y > 1.0)
        color *= 0.0;

    color *= 1.0 - 0.65 * float3(clamp((mod(in.position.x, 2.0) - 1.0) * 2.0, 0.0, 1.0));
    return float4(color,1.0);
}
