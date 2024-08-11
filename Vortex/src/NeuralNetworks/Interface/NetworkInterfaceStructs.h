#pragma once
#ifndef NETWORK_INTERFACE_STRUCTS_H
#define NETWORK_INTERFACE_STRUCTS_H

#define cudaFunction __forceinline__ __device__
#include "core/math.h"

namespace vtx
{
	struct BsdfExtension
	{
		math::vec3f       bsdf;
		math::vec3f       bsdfOverProb;
		float             bsdfProb;
		math::vec3f       wi;
		float             wiProb; // the same as bsdfProb if neural Sampling is not activated
		math::vec3f       Lo;
		math::vec3f       Li;
		bool              isSpecular;
		cudaFunction void reset()
		{
			bsdf         = 0;
			bsdfOverProb = 0;
			bsdfProb     = 0;
			wi           = 0;
			wiProb       = 0;
			Lo           = 0;
			Li           = 0;
			isSpecular   = false;
		}
	};

	struct LightExtension
	{
		math::vec3f bsdf;
		float       bsdfProb;
		math::vec3f wi;
		math::vec3f LiOverProb;
		float       misWeight;
		math::vec3f Lo;
		bool        valid;
		math::vec3f Li;
		float       wiProb;

		cudaFunction void reset()
		{
			bsdf       = 0;
			bsdfProb   = 0;
			wi         = 0;
			LiOverProb = 0;
			misWeight  = 1.0f;
			Lo         = 0;
			valid      = 0;
			Li         = 0;
		}
	};

	struct Hit
	{
		math::vec3f position;
		math::vec3f normal;
		unsigned    matId;
		unsigned    triangleId;
		unsigned    instanceId;

		cudaFunction void reset()
		{
			position   = 0;
			normal     = 0;
			matId      = 0;
			triangleId = 0;
			instanceId = 0;
		}
	};

	struct SurfaceEmission
	{
		math::vec3f Le;
		float       misWeight;

		cudaFunction void reset()
		{
			Le        = 0;
			misWeight = 1.0f;
		}
	};

	struct BounceData
	{
		Hit             hit;
		BsdfExtension   bsdfSample;
		LightExtension  lightSample;
		SurfaceEmission surfaceEmission;
		math::vec3f     wo;
		math::vec3f     Lo;

		cudaFunction void reset()
		{
			hit.reset();
			bsdfSample.reset();
			lightSample.reset();
			surfaceEmission.reset();
			wo = 0;
			Lo = 0;
		}
	};

	struct NetworkInterfaceDebugBuffers
	{
		math::vec3f* inferenceDebugBuffer = nullptr;
		math::vec4f* filmBuffer           = nullptr;
	};
}
#endif
