
#include "libs/stb_image_write.h"

#include "system.h"
#include "render.h"
#include "mem.h"
#include "utils.h"
#include "platform.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#define ATLAS_SIZE 64
#define ATLAS_GRID 32
#define ATLAS_BORDER 16

#define RENDER_TRIS_BUFFER_CAPACITY 2048
#define TEXTURES_MAX 1024

#define NEAR_PLANE 16.0
#define FAR_PLANE (RENDER_FADEOUT_FAR)
#define RENDER_BUFFER_INTERNAL_FORMAT MTLPixelFormatBGRA8Unorm
#define RENDER_DEPTH_BUFFER_INTERNAL_FORMAT MTLPixelFormatDepth32Float

typedef struct {
	vec2i_t offset;
	vec2i_t size;
} render_texture_t;

uint16_t RENDER_NO_TEXTURE;

typedef struct InstanceConstants {
    mat4_t view;
    mat4_t model;
    mat4_t projection;
    vec3_t camera_pos;
    float time;
    vec2_t screen; // Used for screen_size when rendering post effects
    vec2_t fade;
} InstanceConstants;

// -----------------------------------------------------------------------------
// Main game shaders

id<MTLRenderPipelineState> shader_game_init(id<MTLDevice> device, id<MTLLibrary> library, render_blend_mode_t blendMode) {
    NSError *error = nil;
    MTLVertexDescriptor *vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    vertex_descriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[0].offset = 0;
    vertex_descriptor.attributes[0].bufferIndex = 0;
    vertex_descriptor.attributes[1].format = MTLVertexFormatFloat2;
    vertex_descriptor.attributes[1].offset = 12;
    vertex_descriptor.attributes[1].bufferIndex = 0;
    vertex_descriptor.attributes[2].format = MTLVertexFormatUChar4Normalized;
    vertex_descriptor.attributes[2].offset = 20;
    vertex_descriptor.attributes[2].bufferIndex = 0;
    vertex_descriptor.layouts[0].stride = 24;
    id<MTLFunction> vertex_function = [library newFunctionWithName:@"vertex_main"];
    id<MTLFunction> fragment_function = [library newFunctionWithName:@"fragment_main"];
    MTLRenderPipelineDescriptor *pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.vertexFunction = vertex_function;
    pipelineDescriptor.fragmentFunction = fragment_function;
    pipelineDescriptor.vertexDescriptor = vertex_descriptor;
    pipelineDescriptor.colorAttachments[0].pixelFormat = RENDER_BUFFER_INTERNAL_FORMAT;
    pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
    switch (blendMode) {
        case RENDER_BLEND_NORMAL:
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            break;
        case RENDER_BLEND_LIGHTER:
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOne;
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOne;
            break;
        default:
            die("Unrecognized blend mode in Metal renderer");
            break;
    }
    // TODO: Blend state permutations
    pipelineDescriptor.depthAttachmentPixelFormat = RENDER_DEPTH_BUFFER_INTERNAL_FORMAT;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
	return pipeline;
}

// -----------------------------------------------------------------------------
// POST Effect shaders

id<MTLRenderPipelineState> shader_post_init(id<MTLDevice> device, id<MTLLibrary> library, bool crt) {
    NSError *error = nil;
    MTLVertexDescriptor *vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
    vertex_descriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertex_descriptor.attributes[0].offset = 0;
    vertex_descriptor.attributes[0].bufferIndex = 0;
    vertex_descriptor.attributes[1].format = MTLVertexFormatFloat2;
    vertex_descriptor.attributes[1].offset = 12;
    vertex_descriptor.attributes[1].bufferIndex = 0;
    vertex_descriptor.attributes[2].format = MTLVertexFormatUChar4Normalized;
    vertex_descriptor.attributes[2].offset = 20;
    vertex_descriptor.attributes[2].bufferIndex = 0;
    vertex_descriptor.layouts[0].stride = 24;
    id<MTLFunction> vertex_function = [library newFunctionWithName:@"vertex_post"];
    id<MTLFunction> fragment_function = [library newFunctionWithName:crt ? @"fragment_post_crt" : @"fragment_post"];
    MTLRenderPipelineDescriptor *pipelineDescriptor = [MTLRenderPipelineDescriptor new];
    pipelineDescriptor.vertexFunction = vertex_function;
    pipelineDescriptor.fragmentFunction = fragment_function;
    pipelineDescriptor.vertexDescriptor = vertex_descriptor;
    pipelineDescriptor.colorAttachments[0].pixelFormat = RENDER_BUFFER_INTERNAL_FORMAT;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipelineDescriptor error:&error];
    return pipeline;
}

// -----------------------------------------------------------------------------

static id<MTLBuffer> vertex_buffer;

static tris_t tris_buffer[RENDER_TRIS_BUFFER_CAPACITY];
static uint32_t tris_len = 0;

static vec2i_t screen_size;
static vec2i_t backbuffer_size;

static uint32_t atlas_map[ATLAS_SIZE] = {0};
static id<MTLTexture> atlas_texture = 0;
static render_blend_mode_t blend_mode = RENDER_BLEND_NORMAL;

static mat4_t projection_mat_2d = mat4_identity();
static mat4_t projection_mat_bb = mat4_identity();
static mat4_t projection_mat_3d = mat4_identity();
static mat4_t sprite_mat = mat4_identity();
static mat4_t view_mat = mat4_identity();

static bool depth_write_enabled = true;
static bool depth_test_enabled = true;

static InstanceConstants constants;

static render_texture_t textures[TEXTURES_MAX];
static uint32_t textures_len = 0;
static bool texture_mipmap_is_dirty = false;

static render_resolution_t render_res;
static id<MTLTexture> backbuffer_texture = nil;
static id<MTLTexture> backbuffer_depth_buffer = nil;

const int NUM_RENDER_BLEND_MODES = 2;
static id<MTLRenderPipelineState> prg_game[NUM_RENDER_BLEND_MODES];
static id<MTLRenderPipelineState> prg_post;
static id<MTLRenderPipelineState> prg_post_effects[NUM_RENDER_POST_EFFCTS] = {};

static void render_flush();

static id<MTLDevice> metal_device;
static id<MTLCommandQueue> command_queue = nil;
static id<MTLCommandBuffer> current_command_buffer = nil;
static id<MTLRenderCommandEncoder> current_command_encoder = nil;
static id<CAMetalDrawable> current_drawable = nil;
static id<MTLDepthStencilState> depth_read_write_state = nil;
static id<MTLDepthStencilState> depth_read_only_state = nil;
static id<MTLDepthStencilState> depth_disabled_state = nil;

void render_init(vec2i_t screen_size) {
    metal_device = MTLCreateSystemDefaultDevice();
    command_queue = [metal_device newCommandQueue];
    id<MTLLibrary> shader_library = [metal_device newDefaultLibrary];

    CAMetalLayer *metalLayer = (__bridge CAMetalLayer *)platform_get_metal_layer();
    [metalLayer setDevice:metal_device];
    [metalLayer setMaximumDrawableCount:2];

    MTLDepthStencilDescriptor *depthStencilDescriptor = [MTLDepthStencilDescriptor new];
    depthStencilDescriptor.depthCompareFunction = MTLCompareFunctionLess;
    depthStencilDescriptor.depthWriteEnabled = YES;
    depth_read_write_state = [metal_device newDepthStencilStateWithDescriptor:depthStencilDescriptor];
    depthStencilDescriptor.depthWriteEnabled = NO;
    depth_read_only_state = [metal_device newDepthStencilStateWithDescriptor:depthStencilDescriptor];
    depthStencilDescriptor.depthCompareFunction = MTLCompareFunctionAlways;
    depth_disabled_state = [metal_device newDepthStencilStateWithDescriptor:depthStencilDescriptor];

    // Atlas Texture

//	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy);

	uint32_t tw = ATLAS_SIZE * ATLAS_GRID;
	uint32_t th = ATLAS_SIZE * ATLAS_GRID;

    MTLTextureDescriptor *atlas_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                                width:tw
                                                                                               height:th
                                                                                            mipmapped:RENDER_USE_MIPMAPS];
    atlas_texture = [metal_device newTextureWithDescriptor:atlas_descriptor];

	// Tris buffer

    vertex_buffer = [metal_device newBufferWithLength:sizeof(vertex_t) * 65536 options:MTLResourceStorageModeShared];

	// Post Shaders

    prg_post_effects[RENDER_POST_NONE] = shader_post_init(metal_device, shader_library, false);
    prg_post_effects[RENDER_POST_CRT] = shader_post_init(metal_device, shader_library, true);
	render_set_post_effect(RENDER_POST_NONE);

	// Game shader

    prg_game[RENDER_BLEND_NORMAL] = shader_game_init(metal_device, shader_library, RENDER_BLEND_NORMAL);
    prg_game[RENDER_BLEND_LIGHTER] = shader_game_init(metal_device, shader_library, RENDER_BLEND_LIGHTER);

	render_set_view(vec3(0, 0, 0), vec3(0, 0, 0));
	render_set_model_mat(&mat4_identity());

    // Create white texture
	rgba_t white_pixels[4] = {
		rgba(128,128,128,255), rgba(128,128,128,255),
		rgba(128,128,128,255), rgba(128,128,128,255)
	};
	RENDER_NO_TEXTURE = render_texture_create(2, 2, white_pixels);

	// Backbuffer

	render_res = RENDER_RES_NATIVE;
	render_set_screen_size(screen_size);
}

void render_cleanup() {
    backbuffer_texture = nil;
    backbuffer_depth_buffer = nil;
    for (int i = 0; i < NUM_RENDER_BLEND_MODES; ++i) {
        prg_game[i] = nil;
    }
    prg_post = nil;
    for (int i = 0; i < NUM_RENDER_POST_EFFCTS; ++i) {
        prg_post_effects[i] = nil;
    }
    vertex_buffer = nil;
    atlas_texture = nil;
    command_queue = nil;
    metal_device = nil;
}

static mat4_t render_setup_2d_projection_mat(vec2i_t size) {
	float near = -1;
	float far = 1;
	float left = 0;
	float right = size.x;
	float bottom = size.y;
	float top = 0;
	float lr = 1 / (left - right);
	float bt = 1 / (bottom - top);
	float nf = 1 / (near - far);
	return mat4(
		-2 * lr,  0,  0,  0,
		0,  -2 * bt,  0,  0,
		0,        0,  2 * nf,    0, 
		(left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1
	);
}

static mat4_t render_setup_3d_projection_mat(vec2i_t size) {
	// wipeout has a horizontal fov of 90deg, but we want the fov to be fixed 
	// for the vertical axis, so that widescreen displays just have a wider 
	// view. For the original 4/3 aspect ratio this equates to a vertical fov
	// of 73.75deg.
	float aspect = (float)size.x / (float)size.y;
	float fov = (73.75 / 180.0) * 3.14159265358;
	float f = 1.0 / tan(fov / 2);
	float nf = 1.0 / (NEAR_PLANE - FAR_PLANE);
	return mat4(
		f / aspect, 0, 0, 0,
		0, f, 0, 0, 
		0, 0, (FAR_PLANE + NEAR_PLANE) * nf, -1, 
		0, 0, 2 * FAR_PLANE * NEAR_PLANE * nf, 0
	);
}

void render_set_screen_size(vec2i_t size) {
	screen_size = size;
	projection_mat_bb = render_setup_2d_projection_mat(screen_size);

	render_set_resolution(render_res);
}

void render_set_resolution(render_resolution_t res) {
	render_res = res;

	if (res == RENDER_RES_NATIVE) {
		backbuffer_size = screen_size;
	}
	else {
		float aspect = (float)screen_size.x / (float)screen_size.y;
		if (res == RENDER_RES_240P) {
			backbuffer_size = vec2i(240.0 * aspect, 240);
		}
		else if (res == RENDER_RES_480P) {
            backbuffer_size = vec2i(480.0 * aspect, 480);
		}
		else {
			die("Invalid resolution: %d", res);
		}
	}

    MTLTextureDescriptor *backbuffer_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:RENDER_BUFFER_INTERNAL_FORMAT
                                                                                                     width:backbuffer_size.x
                                                                                                    height:backbuffer_size.y
                                                                                                 mipmapped:NO];
    backbuffer_descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    backbuffer_descriptor.storageMode = MTLStorageModePrivate;
    backbuffer_texture = [metal_device newTextureWithDescriptor:backbuffer_descriptor];

    MTLTextureDescriptor *depth_buffer_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:RENDER_DEPTH_BUFFER_INTERNAL_FORMAT
                                                                                                       width:backbuffer_size.x
                                                                                                      height:backbuffer_size.y
                                                                                                   mipmapped:NO];
    depth_buffer_descriptor.usage = MTLTextureUsageRenderTarget;
    depth_buffer_descriptor.storageMode = MTLStorageModePrivate;
    backbuffer_depth_buffer = [metal_device newTextureWithDescriptor:depth_buffer_descriptor];

    projection_mat_2d = render_setup_2d_projection_mat(backbuffer_size);
	projection_mat_3d = render_setup_3d_projection_mat(backbuffer_size);

	// Use nearest texture min filter for 240p and 480p
	//glBindTexture(GL_TEXTURE_2D, atlas_texture);
	//if (res == RENDER_RES_NATIVE) {
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, RENDER_USE_MIPMAPS ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
	//}
	//else {
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//}
}

void render_set_post_effect(render_post_effect_t post) {
	error_if(post < 0 || post > NUM_RENDER_POST_EFFCTS, "Invalid post effect %d", post);
	prg_post = prg_post_effects[post];
}

vec2i_t render_size() {
	return backbuffer_size;
}

void render_frame_prepare() {
    CAMetalLayer *metalLayer = (__bridge CAMetalLayer *)platform_get_metal_layer();
    current_drawable = [metalLayer nextDrawable];

    current_command_buffer = [command_queue commandBuffer];

    MTLRenderPassDescriptor *main_pass_descriptor = [MTLRenderPassDescriptor renderPassDescriptor];
    main_pass_descriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
    main_pass_descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
    main_pass_descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    main_pass_descriptor.colorAttachments[0].texture = backbuffer_texture;
    main_pass_descriptor.depthAttachment.loadAction = MTLLoadActionClear;
    main_pass_descriptor.depthAttachment.storeAction = MTLStoreActionDontCare;
    main_pass_descriptor.depthAttachment.clearDepth = 1.0;
    main_pass_descriptor.depthAttachment.texture = backbuffer_depth_buffer;
    current_command_encoder = [current_command_buffer renderCommandEncoderWithDescriptor:main_pass_descriptor];

    [current_command_encoder setRenderPipelineState:prg_game[blend_mode]];
    [current_command_encoder setCullMode:MTLCullModeBack];
    [current_command_encoder setFrontFacingWinding:MTLWindingCounterClockwise];
    [current_command_encoder setDepthStencilState:depth_read_write_state];

    MTLViewport viewport = { 0.0, 0.0, backbuffer_size.x, backbuffer_size.y, 0.0, 1.0 };
    [current_command_encoder setViewport:viewport];

    [current_command_encoder setFragmentTexture:atlas_texture atIndex:0];
    [current_command_encoder setVertexBuffer:vertex_buffer offset:0 atIndex:0];

    constants.screen = (vec2_t){ 0, 0 };
}

void render_frame_end() {
	render_flush();

    [current_command_encoder endEncoding];

    MTLRenderPassDescriptor *post_pass_descriptor = [MTLRenderPassDescriptor renderPassDescriptor];
    post_pass_descriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
    post_pass_descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
    post_pass_descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    post_pass_descriptor.colorAttachments[0].texture = [current_drawable texture];
    current_command_encoder = [current_command_buffer renderCommandEncoderWithDescriptor:post_pass_descriptor];

    [current_command_encoder setRenderPipelineState:prg_post];

    MTLViewport viewport = { 0.0, 0.0, screen_size.x, screen_size.y, 0.0, 1.0 };
    [current_command_encoder setViewport:viewport];

    constants.projection = projection_mat_bb;
    constants.time = system_cycle_time();
    constants.screen = (vec2_t){ screen_size.x, screen_size.y };

    [current_command_encoder setVertexBuffer:vertex_buffer offset:0 atIndex:0];
    [current_command_encoder setFragmentTexture:backbuffer_texture atIndex:0];

    rgba_t white = rgba(128,128,128,255);
	tris_buffer[tris_len++] = (tris_t){
		.vertices = {
			{.pos = {0, screen_size.y, 0}, .uv = {0, 0}, .color = white},
			{.pos = {screen_size.x, 0, 0}, .uv = {1, 1}, .color = white},
			{.pos = {0, 0, 0}, .uv = {0, 1}, .color = white},
		}
	};
	tris_buffer[tris_len++] = (tris_t){
		.vertices = {
			{.pos = {screen_size.x, screen_size.y, 0}, .uv = {1, 0}, .color = white},
			{.pos = {screen_size.x, 0, 0}, .uv = {1, 1}, .color = white},
			{.pos = {0, screen_size.y, 0}, .uv = {0, 0}, .color = white},
		}
	};

	render_flush();

    [current_command_encoder endEncoding];
    current_command_encoder = nil;

    [current_command_buffer presentDrawable:current_drawable afterMinimumDuration:1.0 / 60.0];

    [current_command_buffer commit];
    current_command_buffer = nil;
}

void render_flush() {
	if (tris_len == 0) {
		return;
	}

	//if (texture_mipmap_is_dirty) {
	//	glGenerateMipmap(GL_TEXTURE_2D);
	//	texture_mipmap_is_dirty = false;
	//}

    if (depth_test_enabled) {
        if (depth_write_enabled) {
            [current_command_encoder setDepthStencilState:depth_read_write_state];
        } else {
            [current_command_encoder setDepthStencilState:depth_read_only_state];
        }
    } else {
        [current_command_encoder setDepthStencilState:depth_disabled_state];
    }
    assert(depth_test_enabled || !depth_write_enabled);

    [current_command_encoder setVertexBytes:&constants length:sizeof(InstanceConstants) atIndex:1];
    [current_command_encoder setFragmentBytes:&constants length:sizeof(InstanceConstants) atIndex:0];

    static int baseVertex = 0;
    if (baseVertex + sizeof(vertex_t) * tris_len * 3 > 65536) {
        baseVertex = 0;
    }
    memcpy(vertex_buffer.contents + baseVertex * sizeof(vertex_t), tris_buffer, sizeof(tris_t) * tris_len);
    [current_command_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:baseVertex vertexCount:tris_len * 3];
    baseVertex += tris_len * 3;

    tris_len = 0;
}


void render_set_view(vec3_t pos, vec3_t angles) {
	render_flush();
	render_set_depth_write(true);
	render_set_depth_test(true);

	view_mat = mat4_identity();
	mat4_set_translation(&view_mat, vec3(0, 0, 0));
	mat4_set_roll_pitch_yaw(&view_mat, vec3(angles.x, -angles.y + M_PI, angles.z + M_PI));
	mat4_translate(&view_mat, vec3_inv(pos));
	mat4_set_yaw_pitch_roll(&sprite_mat, vec3(-angles.x, angles.y - M_PI, 0));

	render_set_model_mat(&mat4_identity());

    constants.view = view_mat;
    constants.projection = projection_mat_3d;
    constants.camera_pos = pos;
    constants.fade = (vec2_t){ RENDER_FADEOUT_NEAR, RENDER_FADEOUT_FAR };
}

void render_set_view_2d() {
	render_flush();
	render_set_depth_test(false);
	render_set_depth_write(false);

	render_set_model_mat(&mat4_identity());
    constants.camera_pos = (vec3_t){ 0, 0, 0 };
    constants.view = mat4_identity();
    constants.projection = projection_mat_2d;
}

void render_set_model_mat(mat4_t *m) {
	render_flush();
    constants.model = *m;
}

void render_set_depth_write(bool enabled) {
	render_flush();
    depth_write_enabled = enabled;
}

void render_set_depth_test(bool enabled) {
	render_flush();
    depth_test_enabled = enabled;
}

void render_set_depth_offset(float offset) {
	render_flush();
    [current_command_encoder setDepthBias:offset slopeScale:1.0 clamp:0.0];
}

void render_set_screen_position(vec2_t pos) {
	render_flush();
    constants.screen = (vec2_t){ pos.x, -pos.y };
}

void render_set_blend_mode(render_blend_mode_t new_mode) {
	if (new_mode == blend_mode) {
		return;
	}
	render_flush();

	blend_mode = new_mode;
    [current_command_encoder setRenderPipelineState:prg_game[blend_mode]];
}

void render_set_cull_backface(bool enabled) {
	render_flush();
	if (enabled) {
        [current_command_encoder setCullMode:MTLCullModeBack];
	}
	else {
        [current_command_encoder setCullMode:MTLCullModeNone];
	}
}

vec3_t render_transform(vec3_t pos) {
	return vec3_transform(vec3_transform(pos, &view_mat), &projection_mat_3d);
}

void render_push_tris(tris_t tris, uint16_t texture_index) {
	error_if(texture_index >= textures_len, "Invalid texture %d", texture_index);
	
	if (tris_len >= RENDER_TRIS_BUFFER_CAPACITY) {
		render_flush();
	}

	render_texture_t *t = &textures[texture_index];

	for (int i = 0; i < 3; i++) {
		tris.vertices[i].uv.x += t->offset.x;
		tris.vertices[i].uv.y += t->offset.y;
	}
	tris_buffer[tris_len++] = tris;
}

void render_push_sprite(vec3_t pos, vec2i_t size, rgba_t color, uint16_t texture_index) {
	error_if(texture_index >= textures_len, "Invalid texture %d", texture_index);

	vec3_t p1 = vec3_add(pos, vec3_transform(vec3(-size.x * 0.5, -size.y * 0.5, 0), &sprite_mat));
	vec3_t p2 = vec3_add(pos, vec3_transform(vec3( size.x * 0.5, -size.y * 0.5, 0), &sprite_mat));
	vec3_t p3 = vec3_add(pos, vec3_transform(vec3(-size.x * 0.5,  size.y * 0.5, 0), &sprite_mat));
	vec3_t p4 = vec3_add(pos, vec3_transform(vec3( size.x * 0.5,  size.y * 0.5, 0), &sprite_mat));

	render_texture_t *t = &textures[texture_index];
	render_push_tris((tris_t){
		.vertices = {
			{
				.pos = p1,
				.uv = {0, 0},
				.color = color
			},
			{
				.pos = p2,
				.uv = {0 + t->size.x ,0},
				.color = color
			},
			{
				.pos = p3,
				.uv = {0, 0 + t->size.y},
				.color = color
			},
		}
	}, texture_index);
	render_push_tris((tris_t){
		.vertices = {
			{
				.pos = p3,
				.uv = {0, 0 + t->size.y},
				.color = color
			},
			{
				.pos = p2,
				.uv = {0 + t->size.x, 0},
				.color = color
			},
			{
				.pos = p4,
				.uv = {0 + t->size.x, 0 + t->size.y},
				.color = color
			},
		}
	}, texture_index);
}

void render_push_2d(vec2i_t pos, vec2i_t size, rgba_t color, uint16_t texture_index) {
	render_push_2d_tile(pos, vec2i(0, 0), render_texture_size(texture_index), size, color, texture_index);
}

void render_push_2d_tile(vec2i_t pos, vec2i_t uv_offset, vec2i_t uv_size, vec2i_t size, rgba_t color, uint16_t texture_index) {
	error_if(texture_index >= textures_len, "Invalid texture %d", texture_index);
	render_push_tris((tris_t){
		.vertices = {
			{
				.pos = {pos.x, pos.y + size.y, 0},
				.uv = {uv_offset.x , uv_offset.y + uv_size.y},
				.color = color
			},
			{
				.pos = {pos.x + size.x, pos.y, 0},
				.uv = {uv_offset.x +  uv_size.x, uv_offset.y},
				.color = color
			},
			{
				.pos = {pos.x, pos.y, 0},
				.uv = {uv_offset.x , uv_offset.y},
				.color = color
			},
		}
	}, texture_index);

	render_push_tris((tris_t){
		.vertices = {
			{
				.pos = {pos.x + size.x, pos.y + size.y, 0},
				.uv = {uv_offset.x + uv_size.x, uv_offset.y + uv_size.y},
				.color = color
			},
			{
				.pos = {pos.x + size.x, pos.y, 0},
				.uv = {uv_offset.x + uv_size.x, uv_offset.y},
				.color = color
			},
			{
				.pos = {pos.x, pos.y + size.y, 0},
				.uv = {uv_offset.x , uv_offset.y + uv_size.y},
				.color = color
			},
		}
	}, texture_index);
}


uint16_t render_texture_create(uint32_t tw, uint32_t th, rgba_t *pixels) {
	error_if(textures_len >= TEXTURES_MAX, "TEXTURES_MAX reached");

	uint32_t bw = tw + ATLAS_BORDER * 2;
	uint32_t bh = th + ATLAS_BORDER * 2;

	// Find a position in the atlas for this texture (with added border)
	uint32_t grid_width = (bw + ATLAS_GRID - 1) / ATLAS_GRID;
	uint32_t grid_height = (bh + ATLAS_GRID - 1) / ATLAS_GRID;
	uint32_t grid_x = 0;
	uint32_t grid_y = ATLAS_SIZE - grid_height + 1;

	for (uint32_t cx = 0; cx < ATLAS_SIZE - grid_width; cx++) {
		if (atlas_map[cx] >= grid_y) {
			continue;
		}

		uint32_t cy = atlas_map[cx];
		bool is_best = true;

		for (uint32_t bx = cx; bx < cx + grid_width; bx++) {
			if (atlas_map[bx] >= grid_y) {
				is_best = false;
				cx = bx;
				break;
			}
			if (atlas_map[bx] > cy) {
				cy = atlas_map[bx];
			}
		}
		if (is_best) {
			grid_y = cy;
			grid_x = cx;
		}
	}

	error_if(grid_y + grid_height > ATLAS_SIZE, "Render atlas ran out of space");

	for (uint32_t cx = grid_x; cx < grid_x + grid_width; cx++) {
		atlas_map[cx] = grid_y + grid_height;
	}

	// Add the border pixels for this texture
	rgba_t *pb = mem_temp_alloc(sizeof(rgba_t) * bw * bh);

	if (tw && th) {
		// Top border
		for (int32_t y = 0; y < ATLAS_BORDER; y++) {
			memcpy(pb + bw * y + ATLAS_BORDER, pixels, tw * sizeof(rgba_t));
		}

		// Bottom border
		for (int32_t y = 0; y < ATLAS_BORDER; y++) {
			memcpy(pb + bw * (bh - ATLAS_BORDER + y) + ATLAS_BORDER, pixels + tw * (th-1), tw * sizeof(rgba_t));
		}
		
		// Left border
		for (int32_t y = 0; y < bh; y++) {
			for (int32_t x = 0; x < ATLAS_BORDER; x++) {
				pb[y * bw + x] = pixels[clamp(y-ATLAS_BORDER, 0, th-1) * tw];
			}
		}

		// Right border
		for (int32_t y = 0; y < bh; y++) {
			for (int32_t x = 0; x < ATLAS_BORDER; x++) {
				pb[y * bw + x + bw - ATLAS_BORDER] = pixels[tw - 1 + clamp(y-ATLAS_BORDER, 0, th-1) * tw];
			}
		}

		// Texture
		for (int32_t y = 0; y < th; y++) {
			memcpy(pb + bw * (y + ATLAS_BORDER) + ATLAS_BORDER, pixels + tw * y, tw * sizeof(rgba_t));
		}
	}

	uint32_t x = grid_x * ATLAS_GRID;
	uint32_t y = grid_y * ATLAS_GRID;
    [atlas_texture replaceRegion:MTLRegionMake2D(x, y, bw, bh) mipmapLevel:0 withBytes:pb bytesPerRow:bw * 4];
    if (atlas_texture.storageMode == MTLStorageModeManaged) {
        // TODO: sync
    }
	mem_temp_free(pb);


	texture_mipmap_is_dirty = RENDER_USE_MIPMAPS;
	uint16_t texture_index = textures_len;
	textures_len++;
	textures[texture_index] = (render_texture_t){ {x + ATLAS_BORDER, y + ATLAS_BORDER}, {tw, th} };

	printf("inserted atlas texture (%3dx%3d) at (%3d,%3d)\n", tw, th, grid_x, grid_y);
	return texture_index;
}

vec2i_t render_texture_size(uint16_t texture_index) {
	error_if(texture_index >= textures_len, "Invalid texture %d", texture_index);
	return textures[texture_index].size;
}

void render_texture_replace_pixels(int16_t texture_index, rgba_t *pixels) {
	error_if(texture_index >= textures_len, "Invalid texture %d", texture_index);

	render_texture_t *t = &textures[texture_index];
    [atlas_texture replaceRegion:MTLRegionMake2D(t->offset.x, t->offset.y, t->size.x, t->size.y)
                     mipmapLevel:0
                       withBytes:pixels
                     bytesPerRow:t->size.x * 4];
    if (atlas_texture.storageMode == MTLStorageModeManaged) {
        // TODO: sync
    }
}

uint16_t render_textures_len() {
	return textures_len;
}

void render_textures_reset(uint16_t len) {
	error_if(len > textures_len, "Invalid texture reset len %d >= %d", len, textures_len);
	render_flush();

	textures_len = len;
	clear(atlas_map);

	// Clear completely and recreate the default white texture
	if (len == 0) {
		rgba_t white_pixels[4] = {
			rgba(128,128,128,255), rgba(128,128,128,255),
			rgba(128,128,128,255), rgba(128,128,128,255)
		};
		RENDER_NO_TEXTURE = render_texture_create(2, 2, white_pixels);
		return;
	}

	// Replay all texture grid insertions up to the reset len
	for (int i = 0; i < textures_len; i++) {
		uint32_t grid_x = (textures[i].offset.x - ATLAS_BORDER) / ATLAS_GRID;
		uint32_t grid_y = (textures[i].offset.y - ATLAS_BORDER) / ATLAS_GRID;
		uint32_t grid_width = (textures[i].size.x + ATLAS_BORDER * 2 + ATLAS_GRID - 1) / ATLAS_GRID;
		uint32_t grid_height = (textures[i].size.y + ATLAS_BORDER * 2 + ATLAS_GRID - 1) / ATLAS_GRID;
		for (uint32_t cx = grid_x; cx < grid_x + grid_width; cx++) {
			atlas_map[cx] = grid_y + grid_height;
		}
	}
}

void render_textures_dump(const char *path) {
	int width = ATLAS_SIZE * ATLAS_GRID;
	int height = ATLAS_SIZE * ATLAS_GRID;
	rgba_t *pixels = malloc(sizeof(rgba_t) * width * height);
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	stbi_write_png(path, width, height, 4, pixels, 0);
	free(pixels);
}
