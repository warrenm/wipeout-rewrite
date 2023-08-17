#ifndef PLATFORM_H
#define PLATFORM_H

#include "types.h"

void platform_exit();
vec2i_t platform_screen_size();
double platform_now();
void platform_set_fullscreen(bool fullscreen);
void platform_set_audio_mix_cb(void (*cb)(float *buffer, uint32_t len));

#if defined(RENDERER_SOFTWARE)
	rgba_t *platform_get_screenbuffer(int32_t *pitch);
#endif

#if defined(RENDERER_METAL)
    void *platform_get_metal_layer(void);
#endif

#endif
