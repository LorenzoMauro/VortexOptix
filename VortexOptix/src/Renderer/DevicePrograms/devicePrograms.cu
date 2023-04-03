#pragma once

#include <optix_device.h>
#include <cuda_runtime.h>
#include "Renderer/LaunchParams.h"
#include <optix.h>

namespace vtx {
	
    /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__radiance()
    { /*! for this simple example, this will remain empty */
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }



    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    { /*! for this simple example, this will remain empty */
    }



    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        const int frameID = optixLaunchParams.frameID;
        if (frameID == 0 &&
            optixGetLaunchIndex().x == 0 &&
            optixGetLaunchIndex().y == 0) {
            // we could of course also have used optixGetLaunchDims to query
            // the launch size, but accessing the optixLaunchParams here
            // makes sure they're not getting optimized away (because
            // otherwise they'd not get used)
            printf("############################################\n");
            printf("Hello world from OptiX 7 raygen program!\n(within a %ix%i-sized launch)\n",
                optixLaunchParams.fbSize.x,
                optixLaunchParams.fbSize.y);
            printf("############################################\n");
        }

        // ------------------------------------------------------------------
        // for this example, produce a simple test pattern:
        // ------------------------------------------------------------------

        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;

        math::vec2f pixel = math::vec2f(ix, iy);
        math::vec2f sample = math::vec2f(0.5f, 0.5f);
        math::vec2f screen = math::vec2f(optixLaunchParams.fbSize.x, optixLaunchParams.fbSize.y);
        const LensRay ray = optixDirectCall<LensRay, const math::vec2f, const math::vec2f, const math::vec2f>(0, screen, pixel, sample);

        const int r = ((ix + frameID) % 256);
        const int g = ((iy + frameID) % 256);
        const int b = ((ix + iy + frameID) % 256);

        // convert to 32-bit rgba value (we explicitly set alpha to 0xff
        // to make stb_image_write happy ...
        const uint32_t rgba = 0xff000000
            | (r << 0) | (g << 8) | (b << 16);

        // and write to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.fbSize.x;
        uint32_t* colorBuffer = reinterpret_cast<uint32_t*>(optixLaunchParams.colorBuffer); // This is a per device launch sized buffer in this renderer strategy.
        colorBuffer[fbIndex] = rgba;
    }
}