#pragma once
#ifndef RENDERER_FUNCTIONS_H
#define RENDERER_FUNCTIONS_H

#include "DebugDownloadData.h"
#include "Utils.h"
#include "Core/Math.h"
#include "Device/DevicePrograms/randomNumberGenerator.h"
#include "Device/DevicePrograms/MdlStructs.h"
#include "Device/DevicePrograms/ToneMapper.h"
#include "Device/Wrappers/SoaWorkItems.h"
#include "NeuralNetworks/Interface/NetworkInterface.h"

#ifdef ARCHITECTURE_OPTIX
#include <optix_device.h>
#else
typedef void (MaterialEvaluationFunction)(vtx::mdl::MdlRequest* request, vtx::mdl::MaterialEvaluation* result);
extern __constant__ unsigned int     mdl_functions_count;
extern __constant__ MaterialEvaluationFunction* mdl_functions[];

namespace vtx::mdl
{
    __forceinline__ __device__ void callEvaluateMaterial(int index, MdlRequest* request, MaterialEvaluation* result)
    {
        mdl_functions[index](request, result);
    }

}
#endif

#define DEBUG_PRINT_INFO false

namespace vtx
{
	enum ArchitectureType
	{
		A_FULL_OPTIX,
		A_WAVEFRONT_OPTIX_SHADE,
		A_WAVEFRONT_CUDA_SHADE
	};

	#define RED     math::vec3f(1.0f, 0.0f, 0.0f)
	#define GREEN   math::vec3f(0.0f, 1.0f, 0.0f)
	#define BLUE    math::vec3f(0.0f, 0.0f, 1.0f)

	#define NEURAL_SAMPLE_DIRECTION(idx) math::normalize(params.replayBuffer->inferenceInputs.action[idx])


	__forceinline__ __device__ Quadrant getQuadrant(const int& pixel, const math::vec2ui& frameSize)
	{
		const int  x      = pixel % (int)frameSize.x;
		const int  y      = pixel / (int)frameSize.x;
		const int  halfX  = (int)frameSize.x / 2;
		const int  halfY  = (int)frameSize.y / 2;
		const bool isTop  = y > halfY;
		const bool isLeft = x < halfX;
		if (isLeft && isTop)
		{
			return Q_TOP_LEFT;
		}
		if (!isLeft && isTop)
		{
			return Q_TOP_RIGHT;
		}
		if (isLeft && !isTop)
		{
			return Q_BOTTOM_LEFT;
		}
		return Q_BOTTOM_RIGHT;
	}

	__forceinline__ __device__ const QuadrantTechniqueSplit& getTechniqueSplit(const LaunchParams* params, const int pixel)
	{
		const Quadrant q = getQuadrant(pixel, params->frameBuffer.frameSize);
		switch (q)
		{
		case Q_TOP_LEFT:
			return params->settings.renderer.quadrantsSettings.topLeft;
		case Q_TOP_RIGHT:
			return params->settings.renderer.quadrantsSettings.topRight;
		case Q_BOTTOM_LEFT:
			return params->settings.renderer.quadrantsSettings.bottomLeft;
		case Q_BOTTOM_RIGHT:
			return params->settings.renderer.quadrantsSettings.bottomRight;
		}
	}

	__forceinline__ __device__ bool isMIS(const LaunchParams* params, const int pixel)
	{
		if (params->settings.renderer.quadrantsSettings.isActivated)
		{
			const QuadrantTechniqueSplit& split = getTechniqueSplit(params, pixel);
			return split.st == S_MIS;
		}
		return params->settings.renderer.samplingTechnique == S_MIS;
	}

	__forceinline__ __device__ bool isDirectLighting(const LaunchParams* params, const int pixel)
	{
		if (params->settings.renderer.quadrantsSettings.isActivated)
		{
			const QuadrantTechniqueSplit& split = getTechniqueSplit(params, pixel);
			return split.st == S_DIRECT_LIGHT;
		}
		return params->settings.renderer.samplingTechnique == S_DIRECT_LIGHT;
	}

	__forceinline__ __device__ bool neuralNetworkActive(const LaunchParams* params)
	{
		return params->settings.wavefront.active && params->settings.neural.active;
	}

	__forceinline__ __device__ bool neuralSamplingActivated(const LaunchParams* params, const int& depth, const int pixel)
	{
		bool quadrantsActive = true;
		if (params->settings.renderer.quadrantsSettings.isActivated)
		{
			const QuadrantTechniqueSplit& split = getTechniqueSplit(params, pixel);
			quadrantsActive                     = split.neuralActivated;
		}
		const bool doSampleNeural =
			neuralNetworkActive(params)
			&& params->settings.neural.doInference
			&& params->settings.renderer.iteration >= params->settings.neural.inferenceIterationStart &&
			quadrantsActive;


		return doSampleNeural;
	}

	__forceinline__ __device__ void evaluateMaterial(const int& programCallId, mdl::MdlRequest* request, mdl::MaterialEvaluation* matEval)
	{
		#ifdef ARCHITECTURE_OPTIX
		optixDirectCall<void, mdl::MdlRequest*, mdl::MaterialEvaluation*>(programCallId, request, matEval);
		#else
        callEvaluateMaterial(programCallId, request, matEval);
		#endif
	}

	__forceinline__ __device__ void nanCheckAdd(const math::vec3f& input, math::vec3f& buffer)
	{
		if (!utl::isNan(input))
		{
			buffer += input;
		}
	}

	__forceinline__ __device__ void nanCheckAddAtomic(const math::vec3f& input, math::vec3f& buffer)
	{
		if (!utl::isNan(input))
		{
			//buffer += input;
			cuAtomicAdd(&buffer.x, input.x);
			cuAtomicAdd(&buffer.y, input.y);
			cuAtomicAdd(&buffer.z, input.z);
		}
	}

	__forceinline__ __device__ void addDebug01(const math::vec3f& color, const int pixelId, const LaunchParams* params)
	{
		nanCheckAddAtomic(color, params->frameBuffer.debugColor1[pixelId]);
	}

	/*__forceinline__ __device__ void addDebug02(const math::vec3f& color, const int pixelId, const LaunchParams* params)
	{
		if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
		{
			nanCheckAddAtomic(color, params->frameBuffer.debugColor2[pixelId]);
		}
		else
		{
			nanCheckAdd(color, params->frameBuffer.debugColor2[pixelId]);
		}
	}*/

	__forceinline__ __device__ void accumulateRay(const AccumulationWorkItem& awi, const LaunchParams* params)
	{
		if (params->debugData->debugPixel == awi.originPixel)
		{
			nanCheckAddAtomic(awi.radiance, params->debugData->bounceData[0].accumulatedRadiance);
		}
		nanCheckAddAtomic(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
		//if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
		//{
		//    nanCheckAddAtomic(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
		//}
		//else
		//{
		//    nanCheckAdd(awi.radiance, params->frameBuffer.radianceAccumulator[awi.originPixel]);
		//}
	}

	__forceinline__ __device__ void cleanFrameBuffer(const int id, const LaunchParams* params)
	{
		const bool cleanOnInferenceStart = neuralNetworkActive(params) && params->settings.neural.doInference && params->settings.renderer.iteration == params->settings.neural.inferenceIterationStart && params->settings.neural.clearOnInferenceStart;
		if (params->settings.renderer.iteration <= 0 || !params->settings.renderer.accumulate || cleanOnInferenceStart)
		{
			const FrameBufferData frameBuffer   = params->frameBuffer;
			frameBuffer.radianceAccumulator[id] = 0.0f;
			frameBuffer.albedoAccumulator[id]   = 0.0f;
			frameBuffer.normalAccumulator[id]   = 0.0f;
			frameBuffer.tmRadiance[id]          = 0.0f;
			frameBuffer.hdriRadiance[id]        = 0.0f;
			frameBuffer.normalNormalized[id]    = 0.0f;
			frameBuffer.albedoNormalized[id]    = 0.0f;
			frameBuffer.trueNormal[id]          = 0.0f;
			frameBuffer.tangent[id]             = 0.0f;
			frameBuffer.orientation[id]         = 0.0f;
			frameBuffer.uv[id]                  = 0.0f;
			frameBuffer.fireflyPass[id]         = 0.0f;
			frameBuffer.samples[id]             = 0;
			frameBuffer.gBufferHistory[id].reset();
			frameBuffer.gBuffer[id]                                      = 0.0f;
			reinterpret_cast<math::vec4f*>(frameBuffer.outputBuffer)[id] = math::vec4f(0.0f);
			frameBuffer.noiseBuffer[id].adaptiveSamples                  = 1;
			frameBuffer.debugColor1[id]                                  = 0.0f;
			if (neuralNetworkActive(params))
			{
				params->networkInterface->debugBuffers->filmBuffer[id] = 0.0f;
			}
		}
		if (neuralNetworkActive(params))
		{
			params->networkInterface->debugBuffers->inferenceDebugBuffer[id] = 0.0f;
		}
	}

	__forceinline__ __device__ void generateCameraRay(int id, const LaunchParams* params, TraceWorkItem& twi)
	{
		math::vec2f pixel = math::vec2f((float)(id % params->frameBuffer.frameSize.x), (float)(id / params->frameBuffer.frameSize.x));
		math::vec2f screen{(float)params->frameBuffer.frameSize.x, (float)params->frameBuffer.frameSize.y};

		math::vec2f       rnd01    = rng2(twi.seed);
		math::vec2f       rnd02    = rng2(twi.seed);
		const math::vec2f fragment = pixel + math::vec2f(rnd01.x > 0.5f ? 0.5f : -0.5f, rnd01.y > 0.5f ? 0.5f : -0.5f) + (rnd02 * 0.5f - 0.25f);
		const math::vec2f ndc      = (fragment / screen) * 2.0f - 1.0f; // Normalized device coordinates in range [-1, 1].

		const CameraData camera = params->cameraData;

		math::vec3f origin              = camera.position;
		math::vec3f direction           = camera.horizontal * ndc.x + camera.vertical * ndc.y + camera.direction;
		math::vec3f normalizedDirection = math::normalize(direction);

		twi.origin      = origin;
		twi.direction   = normalizedDirection;
		twi.originPixel = id;
		twi.radiance    = math::vec3f(0.0f);
		twi.pdf         = 1.0f;
		twi.throughput  = math::vec3f(1.0f);
		twi.eventType   = mi::neuraylib::BSDF_EVENT_ABSORB; // Initialize for exit. (Otherwise miss programs do not work.)
		twi.depth       = 0;
		twi.mediumIor   = math::vec3f(1.0f);
		twi.extendRay   = true;
		params->frameBuffer.samples[id] += 1;
	}

	__forceinline__ __device__ math::vec3f missShader(EscapedWorkItem& ewi, const LaunchParams* params)
	{
		math::vec3f emission  = 0.0f;
		float       misWeight = 1.0f;
		if (!params->settings.renderer.viewBackground && ewi.depth == 0)
		{
			return math::vec3f(0.0f);
		}
		if (params->envLight != nullptr)
		{
			const LightData*       envLight = params->envLight;
			EnvLightAttributesData attrib   = *reinterpret_cast<EnvLightAttributesData*>(envLight->attributes);
			auto                   texture  = attrib.texture;

			math::vec3f dir = math::transformNormal3F(attrib.invTransformation, ewi.direction);

			{
				bool  computeOriginalUV = false;
				float u, v;
				if (computeOriginalUV)
				{
					u = fmodf(atan2f(dir.y, dir.x) * (float)(0.5 / M_PI) + 0.5f, 1.0f);
					v = acosf(fmax(fminf(-dir.z, 1.0f), -1.0f)) * (float)(1.0 / M_PI);
				}
				else
				{
					float theta = acosf(-dir.z);
					v           = theta / (float)M_PI;
					float phi   = atan2f(dir.y, dir.x); // + M_PI / 2.0f; // azimuth angle (theta)
					u           = (phi + (float)M_PI) / (float)(2.0f * M_PI);
				}

				const auto x = math::min<unsigned>((unsigned int)(u * (float)texture->dimension.x), texture->dimension.x - 1);
				const auto y = math::min<unsigned>((unsigned int)(v * (float)texture->dimension.y), texture->dimension.y - 1);
				emission     = tex2D<float4>(texture->texObj, u, v);
				emission     = emission * attrib.scaleLuminosity;

				// to incorporate the point light selection probability
				// If the last surface intersection was a diffuse event which was directly lit with multiple importance sampling,
				// then calculate light emission with multiple importance sampling for this implicit light hit as well.
				bool  MiSCondition = (isMIS(params, ewi.originPixel) && (ewi.eventType & (mi::neuraylib::BSDF_EVENT_DIFFUSE | mi::neuraylib::BSDF_EVENT_GLOSSY)));
				float envSamplePdf = attrib.aliasMap[y * texture->dimension.x + x].pdf;
				if (ewi.pdf > 0.0f && MiSCondition)
				{
					misWeight = utl::heuristic(ewi.pdf, envSamplePdf);
				}
				ewi.radiance += ewi.throughput * emission * misWeight;
				if (ewi.depth == 0)
				{
					params->frameBuffer.noiseBuffer[ewi.originPixel].adaptiveSamples = -1; //Let's inform adaptive not to sample again a direct miss;
					nanCheckAdd(math::normalize(emission), params->frameBuffer.albedoAccumulator[ewi.originPixel]);
				}
			}
		}

		if (neuralNetworkActive(params))
		{
			BounceData& networkBounceData                            = params->networkInterface->getAndReset(ewi.originPixel, ewi.depth);
			params->networkInterface->maxPathLength[ewi.originPixel] = ewi.depth;
			networkBounceData.surfaceEmission.Le                     = emission;
			networkBounceData.surfaceEmission.misWeight              = misWeight;
			networkBounceData.lightSample.valid                      = false;
		}

		return emission;
	}

	__forceinline__ __device__ LightSample sampleMeshLight(const LightData& light, RayWorkItem& prd, LaunchParams& params)
	{
		MeshLightAttributesData meshLightAttributes = *(MeshLightAttributesData*)(light.attributes);
		LightSample             lightSample;

		lightSample.pdf = 0.0f;

		const float3 sample3D = rng3(prd.seed);

		// Uniformly sample the triangles over their surface area.
		// Note that zero-area triangles (e.g. at the poles of spheres) are automatically never sampled with this method!
		// The cdfU is one bigger than res.y.
		unsigned int idxTriangle = utl::binarySearchCdf(meshLightAttributes.cdfArea, meshLightAttributes.size, sample3D.z);
		idxTriangle              = meshLightAttributes.actualTriangleIndices[idxTriangle];
		unsigned instanceId      = meshLightAttributes.instanceId;

		// Barycentric coordinates.
		const float sqrtSampleX = sqrtf(sample3D.x);
		const float alpha       = 1.0f - sqrtSampleX;
		const float beta        = sample3D.y * sqrtSampleX;
		const float gamma       = 1.0f - alpha - beta;
		const auto  baricenter  = math::vec3f(alpha, beta, gamma);

		HitProperties hitP;
		hitP.init(instanceId, idxTriangle, baricenter, 0.0f); // zero because position needs to be calculated
		hitP.determineMaterialInfo(*params.instances[instanceId]);
		if (!hitP.hasEdf)
		{
			//printf("Warning: sampled Triangle doesn't have light, well that's weird, however we return\n");
			return lightSample;
		}
		hitP.calculateForMeshLightSampling(&params);

		lightSample.direction = hitP.position - prd.hitProperties.position;
		lightSample.distance  = math::length(lightSample.direction);
		lightSample.direction = math::normalize(lightSample.direction);
		lightSample.position  = hitP.position;
		lightSample.normal    = hitP.shadingNormal;
		hitP.isFrontFace      = dot(lightSample.direction, hitP.shadingNormal) <= 0.0f; // Explicitly include edge-on cases as frontface condition!

		if (lightSample.distance < DENOMINATOR_EPSILON)
		{
			return lightSample;
		}

		mdl::MdlRequest request;

		request.opacity           = true;
		request.outgoingDirection = -lightSample.direction;
		request.surroundingIor    = 1.0f;
		request.seed              = &prd.seed;
		request.hitProperties     = &hitP;
		request.edf               = true;

		mdl::MaterialEvaluation matEval;

		evaluateMaterial(hitP.programCall, &request, &matEval);

		if (matEval.opacity <= 0.0f)
		{
			return lightSample;
		}

		if (matEval.edf.isValid)
		{
			const float totArea = meshLightAttributes.totalArea;

			// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
			const float factor = (matEval.edf.mode == 0) ? matEval.opacity : matEval.opacity / totArea;

			lightSample.pdf             = lightSample.distance * lightSample.distance / (totArea * matEval.edf.cos); // Solid angle measure.
			lightSample.radiance        = matEval.edf.intensity * matEval.edf.edf * factor;
			lightSample.radianceOverPdf = lightSample.radiance / lightSample.pdf;
			lightSample.isValid         = true;
		}

		return lightSample;
	}

	__forceinline__ __device__ LightSample sampleEnvironment(const LightData& light, RayWorkItem& prd, LaunchParams& params)
	{
		EnvLightAttributesData attrib  = *reinterpret_cast<EnvLightAttributesData*>(light.attributes);
		auto                   texture = attrib.texture;

		unsigned width  = texture->dimension.x;
		unsigned height = texture->dimension.y;

		LightSample lightSample;

		// importance sample an envmap pixel using an alias map
		const float3       sample = rng3(prd.seed);
		const unsigned int size   = width * height;
		const auto         idx    = math::min<unsigned>((unsigned int)(sample.x * (float)size), size - 1);
		unsigned int       envIdx;
		float              sampleY = sample.y;
		if (sampleY < attrib.aliasMap[idx].q)
		{
			envIdx = idx;
			sampleY /= attrib.aliasMap[idx].q;
		}
		else
		{
			envIdx  = attrib.aliasMap[idx].alias;
			sampleY = (sampleY - attrib.aliasMap[idx].q) / (1.0f - attrib.aliasMap[idx].q);
		}

		const unsigned int py = envIdx / width;
		const unsigned int px = envIdx % width;
		lightSample.pdf       = attrib.aliasMap[envIdx].pdf;

		const float u = (float)(px + sampleY) / (float)width;
		//const float phi = (M_PI_2)*(1.0f-u);

		//const float phi = (float)M_PI  -u * (float)(2.0 * M_PI);
		const float phi = u * (float)(2.0 * M_PI) - (float)M_PI;
		float       sinPhi, cosPhi;
		sincosf(phi > float(-M_PI) ? phi : (phi + (float)(2.0 * M_PI)), &sinPhi, &cosPhi);
		const float stepTheta = (float)M_PI / (float)height;
		const float theta0    = (float)(py) * stepTheta;
		const float cosTheta  = cosf(theta0) * (1.0f - sample.z) + cosf(theta0 + stepTheta) * sample.z;
		const float theta     = acosf(cosTheta);
		const float sinTheta  = sinf(theta);
		const float v         = theta * (float)(1.0 / M_PI);

		float x = cosPhi * sinTheta;
		float y = sinPhi * sinTheta;
		float z = -cosTheta;

		math::vec3f dir{x, y, z};
		// Now rotate that normalized object space direction into world space. 
		lightSample.direction = math::transformNormal3F(attrib.transformation, dir);
		lightSample.distance  = params.settings.renderer.maxClamp; // Environment light.
		lightSample.position  = prd.hitProperties.position + lightSample.direction * lightSample.distance;
		lightSample.normal    = -lightSample.direction;
		// Get the emission from the spherical environment texture.
		lightSample.radiance = tex2D<float4>(texture->texObj, u, v);
		lightSample.radiance *= attrib.scaleLuminosity;
		// For simplicity we pretend that we perfectly importance-sampled the actual texture-filtered environment map
		// and not the Gaussian-smoothed one used to actually generate the CDFs and uniform sampling in the texel.
		// (Note that this does not contain the light.emission which just modulates the texture.)
		lightSample.radianceOverPdf = lightSample.radiance;
		if (DENOMINATOR_EPSILON < lightSample.pdf)
		{
			lightSample.radianceOverPdf = lightSample.radianceOverPdf / lightSample.pdf;
		}

		lightSample.isValid = true;
		return lightSample;
	}

	__forceinline__ __device__ LightSample sampleLight(RayWorkItem& prd, LaunchParams& params)
	{
		LightSample lightSample;
		if (const int& numLights = params.numberOfLights; numLights > 0)
		{
			//Randomly Selecting a Light
			//TODO, I think here we can do some better selection by giving more importance to lights with greater power

			const int indexLight = (1 < numLights) ? gdt::clamp(static_cast<int>(floorf(rng(prd.seed) * numLights)), 0, numLights - 1) : 0;

			const LightData& light = *(params.lights[indexLight]);

			const LightType& typeLight = light.type;

			switch (typeLight)
			{
			case L_MESH:
				{
					lightSample = sampleMeshLight(light, prd, params);
				}
				break;
			case L_ENV:
				{
					lightSample = sampleEnvironment(light, prd, params);
				}
				break;
			default:
				{
					lightSample.isValid = false;
					return lightSample;
				}
			}

			lightSample.typeLight = typeLight;

			lightSample.pdf             = lightSample.pdf * (1.0f / numLights);
			lightSample.radianceOverPdf = lightSample.radianceOverPdf * (float)numLights;

			if (lightSample.isValid && lightSample.pdf > 0.0f)
			{
				return lightSample;
			}
		}
		lightSample.isValid = false;
		lightSample.pdf     = 0.0f;
		return lightSample;
	}

	__forceinline__ __device__ void setAuxiliaryRenderPassData(const RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, const LaunchParams* params)
	{
		//Auxiliary Data
		if (prd.depth == 0)
		{
			const math::vec3f colorsTrueNormal  = 0.5f * (prd.hitProperties.trueNormal + 1.0f);
			const math::vec3f colorsUv          = prd.hitProperties.uv;
			const math::vec3f colorsOrientation = prd.hitProperties.isFrontFace ? math::vec3f(0.0f, 0.0f, 1.0f) : math::vec3f(1.0f, 0.0f, 0.0f);
			const math::vec3f colorsTangent     = 0.5f * (prd.hitProperties.tangent + 1.0f);
			if (matEval.aux.isValid)
			{
				const math::vec3f colorsBounceDiffuse = matEval.aux.albedo;
				const math::vec3f colorsShadingNormal = 0.5f * (matEval.aux.normal + 1.0f);
				if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
				{
					nanCheckAddAtomic(colorsBounceDiffuse, params->frameBuffer.albedoAccumulator[prd.originPixel]);
					nanCheckAddAtomic(colorsShadingNormal, params->frameBuffer.normalAccumulator[prd.originPixel]);
				}
				else
				{
					nanCheckAdd(colorsBounceDiffuse, params->frameBuffer.albedoAccumulator[prd.originPixel]);
					nanCheckAdd(colorsShadingNormal, params->frameBuffer.normalAccumulator[prd.originPixel]);
				}
			}
			if (params->settings.renderer.adaptiveSamplingSettings.active && params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
			{
				nanCheckAddAtomic(colorsTrueNormal, params->frameBuffer.trueNormal[prd.originPixel]);
				nanCheckAddAtomic(colorsTangent, params->frameBuffer.tangent[prd.originPixel]);
				nanCheckAddAtomic(colorsOrientation, params->frameBuffer.orientation[prd.originPixel]);
				nanCheckAddAtomic(colorsUv, params->frameBuffer.uv[prd.originPixel]);
			}
			else
			{
				nanCheckAdd(colorsTrueNormal, params->frameBuffer.trueNormal[prd.originPixel]);
				nanCheckAdd(colorsTangent, params->frameBuffer.tangent[prd.originPixel]);
				nanCheckAdd(colorsOrientation, params->frameBuffer.orientation[prd.originPixel]);
				nanCheckAdd(colorsUv, params->frameBuffer.uv[prd.originPixel]);
			}
		}
	}

	__forceinline__ __device__ void auxiliaryNetworkInference(
		const LaunchParams* params,
		const int&          originPixel, const int& depth, const int& shadeQueueIndex,
		const float&        samplingFraction,
		const math::vec3f&  sample, const float&     pdf,
		const math::vec3f&  bsdfSample, const float& bsdfProb, const float sampleProb
	)
	{
		if (!neuralNetworkActive(params) || depth != params->debugData->debugDepth)
		{
			return;
		}


		math::vec3f*   debugBuffer      = params->networkInterface->debugBuffers->inferenceDebugBuffer;
		InferenceData* inferenceQueries = params->networkInterface->inferenceData;

		switch (params->settings.renderer.displayBuffer)
		{
		case(FB_NETWORK_INFERENCE_STATE_POSITION):
			{
				debugBuffer[originPixel] = inferenceQueries->getStatePosition(shadeQueueIndex);
			}
			break;
		case(FB_NETWORK_INFERENCE_STATE_NORMAL):
			{
				debugBuffer[originPixel] = (inferenceQueries->getStateNormal(shadeQueueIndex) + 1.0f) * 0.5f;
			}
			break;
		case(FB_NETWORK_INFERENCE_OUTGOING_DIRECTION):
			{
				debugBuffer[originPixel] = (inferenceQueries->getStateDirection(shadeQueueIndex) + 1.0f) * 0.5f;
			}
			break;
		case(FB_NETWORK_INFERENCE_MEAN):
			{
				debugBuffer[originPixel] = (inferenceQueries->getMean(shadeQueueIndex) + 1.0f) * 0.5f;
			}
			break;
		case(FB_NETWORK_INFERENCE_SAMPLE):
			{
				if (sample == math::vec3f(0.0f))
				{
					debugBuffer[originPixel] = math::vec3f(0.0f);
				}
				else
				{
					debugBuffer[originPixel] = (sample + 1.0f) * 0.5f;
				}
			}
			break;
		case(FB_NETWORK_INFERENCE_SAMPLE_DEBUG):
			{
				if (sample == math::vec3f(0.0f))
				{
					debugBuffer[originPixel] = math::vec3f(0.0f);
					return;
				}
				const math::vec3f mean          = inferenceQueries->getMean(shadeQueueIndex);
				const float       cosSampleMean = dot(mean, sample);
				const float       value         = (cosSampleMean + 1.0f) * 0.5f;
				//const math::vec3f color = floatToScientificRGB(value);
				//debugBuffer[originPixel] = color;

				math::vec3f cosineColor  = RED * value + GREEN * (1.0f - value);
				cosineColor.z            = pdf;
				debugBuffer[originPixel] = cosineColor;
			}
			break;
		case(FB_NETWORK_INFERENCE_CONCENTRATION):
			{
				debugBuffer[originPixel] = floatToScientificRGB(fminf(1.0f, inferenceQueries->getConcentration(shadeQueueIndex)));
			}
			break;
		case(FB_NETWORK_INFERENCE_ANISOTROPY):
			{
				debugBuffer[originPixel] = floatToScientificRGB(fminf(1.0f, inferenceQueries->getAnisotropy(shadeQueueIndex)));
			}
			break;
		case(FB_NETWORK_INFERENCE_SAMPLING_FRACTION):
			{
				debugBuffer[originPixel] = math::vec3f(floatToScientificRGB(samplingFraction));
			}
			break;
		case(FB_NETWORK_INFERENCE_PDF):
			{
				if (pdf == 0.0f)
				{
					debugBuffer[originPixel] = math::vec3f(0.0f);
				}
				else
				{
					debugBuffer[originPixel] = math::vec3f(floatToScientificRGB(fminf(1.0f, pdf)));
				}
			}
			break;
		}
	}

	__forceinline__ __device__ float neuralSamplingFraction(const LaunchParams& params, const int& shadeQueueIndex)
	{
		//if(params.settings.neural.type == network::NT_SAC)
		//{
		//	samplingFraction = params.settings.neural.sac.neuralSampleFraction;
		//}
		//else
		//{
		//}
		const float samplingFraction = params.networkInterface->inferenceData->getSamplingFraction(shadeQueueIndex);

		return samplingFraction;
	}

	__forceinline__ __device__ void correctBsdfSampling(const LaunchParams& params, mdl::MaterialEvaluation* matEval, const RayWorkItem& prd, const math::vec4f& neuralSample, float samplingFraction, const bool& doNeuralSample, const int& shadeQueueIndex)
	{
		math::vec3f neuralDirection = {0.0f, 0.0f, 0.0f};
		float       neuralPdf       = 0.0f;
		math::vec3f bsdfDir         = matEval->bsdfSample.nextDirection;
		float       bsdfPdf         = matEval->bsdfSample.bsdfPdf;

		if (
			!(matEval->bsdfSample.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR) // && // we don't use neural sampling for specular events
			// in case the neural network wants to absorb the ray, which is equivalent to a sampling fraction of 0
			//((doNeuralSample && matEval->neuralBsdfEvaluation.eventType != mi::neuraylib::BSDF_EVENT_ABSORB) || !doNeuralSample)
		)
		{
			math::vec3f                    bsdf;
			float                          bsdfProb;
			math::vec3f                    wi;
			mi::neuraylib::Bsdf_event_type eventType;
			if (doNeuralSample)
			{
				bsdf            = matEval->neuralBsdfEvaluation.bsdf;
				wi              = {neuralSample.x, neuralSample.y, neuralSample.z};
				bsdfProb        = matEval->neuralBsdfEvaluation.bsdfPdf;
				neuralPdf       = neuralSample.w;
				eventType       = matEval->neuralBsdfEvaluation.eventType;
				neuralDirection = wi;
			}
			else
			{
				bsdf      = matEval->bsdfSample.bsdf;
				wi        = matEval->bsdfSample.nextDirection;
				bsdfProb  = matEval->bsdfSample.bsdfPdf;
				neuralPdf = params.networkInterface->inferenceData->evaluate(shadeQueueIndex, matEval->bsdfSample.nextDirection);
				eventType = matEval->bsdfSample.eventType;
			}

			matEval->bsdfSample.eventType = eventType;
			if (eventType == mi::neuraylib::BSDF_EVENT_ABSORB)
			{
				return;
			}

			matEval->bsdfSample.isValid       = true;
			matEval->bsdfSample.nextDirection = wi;
			matEval->bsdfSample.bsdf          = bsdf;
			matEval->bsdfSample.bsdfPdf       = bsdfProb;
			matEval->bsdfSample.pdf           = bsdfProb * (1.0f - samplingFraction) + neuralPdf * samplingFraction;
			matEval->bsdfSample.bsdfOverPdf   = bsdf / (matEval->bsdfSample.pdf);

			if (
				math::isNan(matEval->bsdfSample.bsdfOverPdf) ||
				matEval->bsdfSample.pdf <= 0.0f
			)
			{
				matEval->bsdfSample.eventType = mi::neuraylib::BSDF_EVENT_ABSORB;
			}
		}
		auxiliaryNetworkInference(&params, prd.originPixel, prd.depth, shadeQueueIndex, samplingFraction, neuralDirection, neuralPdf, bsdfDir, bsdfPdf, matEval->bsdfSample.pdf);

		if (params.debugData->debugPixel == prd.originPixel)
		{
			params.debugData->bounceData[prd.depth].isNeuralSample   = doNeuralSample;
			params.debugData->bounceData[prd.depth].wi               = matEval->bsdfSample.nextDirection;
			params.debugData->bounceData[prd.depth].wiPdf            = matEval->bsdfSample.pdf;
			params.debugData->bounceData[prd.depth].neuralSample     = neuralDirection;
			params.debugData->bounceData[prd.depth].neuralSamplePdf  = neuralPdf;
			params.debugData->bounceData[prd.depth].samplingFraction = samplingFraction;
			params.debugData->bounceData[prd.depth].bsdf             = matEval->bsdfSample.bsdf;
			params.debugData->bounceData[prd.depth].bsdfPdf          = bsdfPdf;
			params.debugData->bounceData[prd.depth].bsdfSample       = bsdfDir;
			const float* mixtureParameters                           = nullptr;
			const float* mixtureWeights                              = nullptr;
			params.networkInterface->inferenceData->getSampleMixtureParameters(shadeQueueIndex, mixtureParameters, mixtureWeights);
			const int paramCount = distribution::Mixture::getDistributionParametersCount(params.settings.neural.distributionType);
			params.debugData->copyMixtureData(mixtureWeights, mixtureParameters, prd.depth, params.settings.neural.mixtureSize, paramCount);
		}
	}

	__forceinline__ __device__ void correctLightSample(const LaunchParams& params, mdl::MaterialEvaluation* matEval, const math::vec3f& lightDirection, const int& shadeQueueIndex, const RayWorkItem& prd)
	{
		if (matEval->bsdfEvaluation.isValid)
		{
			if (
				math::isZero(matEval->bsdfEvaluation.bsdf) ||
				matEval->bsdfEvaluation.bsdfPdf == 0.0f ||
				matEval->bsdfEvaluation.eventType == mi::neuraylib::BSDF_EVENT_ABSORB
			)
			{
				matEval->bsdfEvaluation.isValid = false;
				return;
			}
			const float samplingFraction = neuralSamplingFraction(params, shadeQueueIndex);
			const float neuralPdf        = params.networkInterface->inferenceData->evaluate(shadeQueueIndex, lightDirection);
			if (isnan(samplingFraction) || isnan(neuralPdf) || samplingFraction < 0.0f || neuralPdf < 0.0f || isinf(samplingFraction) || isinf(neuralPdf))
			{
				//printf("Invalid neural in CORRECT LIGHT SAMPLE pdf: %f, sampling fraction: %f\n", neuralPdf, samplingFraction);
				return;
			}
			matEval->bsdfEvaluation.pdf = matEval->bsdfEvaluation.bsdfPdf * (1.0f - samplingFraction) + neuralPdf * samplingFraction;
			if (params.debugData->debugPixel == prd.originPixel)
			{
				params.debugData->bounceData[prd.depth].lsNeuralPdf = neuralPdf;
				params.debugData->bounceData[prd.depth].lsBsdfPdf   = matEval->bsdfEvaluation.bsdfPdf;
				params.debugData->bounceData[prd.depth].lsDoNeural  = true;
			}
		}
	}

	__forceinline__ __device__ void evaluateMaterialAndSampleLight(
		mdl::MaterialEvaluation* matEval, LightSample* lightSample, LaunchParams& params,
		RayWorkItem&             prd, int              shadeQueueIndex)
	{
		mdl::MdlRequest request;
		prd.depth == 0 ? request.auxiliary = true : request.auxiliary = false;
		request.ior                        = true;
		request.outgoingDirection          = -prd.direction;
		request.surroundingIor             = prd.mediumIor;
		request.opacity                    = false;
		request.seed                       = &prd.seed;
		request.hitProperties              = &prd.hitProperties;
		request.edf                        = prd.hitProperties.hasEdf;

		bool       isDirectOnly  = isDirectLighting(&params, prd.originPixel);
		bool       doNEE         = isMIS(&params, prd.originPixel);
		const bool doSampleLight = isDirectOnly || doNEE;
		const bool doSampleBsdf  = !isDirectOnly;

		if (doSampleLight)
		{
			*lightSample = sampleLight(prd, params);
			if (lightSample->isValid)
			{
				request.bsdfEvaluation = true;
				request.toSampledLight = lightSample->direction;
			}
		}

		math::vec4f neuralSample              = math::vec4f(0.0f);
		bool        isNeuralSamplingActivated = neuralSamplingActivated(&params, prd.depth, prd.originPixel);
		bool        doNeuralSample            = false;
		float       samplingFraction          = 0.0f;
		if (doSampleBsdf)
		{
			request.bsdfSample = true;
			if (isNeuralSamplingActivated)
			{
				samplingFraction = neuralSamplingFraction(params, shadeQueueIndex);
				if (isnan(samplingFraction) || isinf(samplingFraction) || samplingFraction < 0.0f || samplingFraction > 1.0f)
				{
					//printf("Invalid sampling fraction %f\n", samplingFraction);
					samplingFraction = 0.0f;
				}
				else
				{
					if (const float uniformSample = rng(prd.seed); uniformSample > samplingFraction)
					{
						doNeuralSample = false;
					}
					else
					{
						doNeuralSample               = true;
						request.evalOnNeuralSampling = true;
						neuralSample                 = params.networkInterface->inferenceData->sample(shadeQueueIndex, prd.seed);
						request.toNeuralSample       = {neuralSample.x, neuralSample.y, neuralSample.z};
						if (utl::isNan(request.toNeuralSample) || utl::isInf(request.toNeuralSample) || math::length(request.toNeuralSample) <= 0.0f || neuralSample.w < 0.0f)
						{
							//printf("Invalid neural sample %f %f %f Prob %f\n", neuralSample.x, neuralSample.y, neuralSample.z, neuralSample.w);
							doNeuralSample               = false;
							request.evalOnNeuralSampling = false;
							samplingFraction             = 0.0f;
						}
					}
				}
			}
		}

		evaluateMaterial(prd.hitProperties.programCall, &request, matEval);

		assert(!(utl::isNan(matEval->bsdfSample.nextDirection) && matEval->bsdfSample.eventType != 0));


		if (isNeuralSamplingActivated)
		{
			if (doSampleBsdf)
			{
				correctBsdfSampling(params, matEval, prd, neuralSample, samplingFraction, doNeuralSample, shadeQueueIndex);
			}
			if (doSampleLight)
			{
				correctLightSample(params, matEval, lightSample->direction, shadeQueueIndex, prd);
			}
		}
	}

	__forceinline__ __device__ ShadowWorkItem nextEventEstimation(
		const mdl::MaterialEvaluation& matEval,
		const LightSample&             lightSample,
		const RayWorkItem&             prd,
		const LaunchParams&            params,
		BounceData*                    networkBounceData)
	{
		ShadowWorkItem swi;
		swi.distance = -1; //invalid

		if (
			(isMIS(&params, prd.originPixel) || isDirectLighting(&params, prd.originPixel))
			&& lightSample.isValid
			&& matEval.bsdfEvaluation.isValid
			&& !(matEval.bsdfEvaluation.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR)
		)
		{
			float weightMis = 1.0f;
			if ((lightSample.typeLight == L_MESH || lightSample.typeLight == L_ENV) && !isDirectLighting(&params, prd.originPixel))
			{
				weightMis = utl::heuristic(lightSample.pdf, matEval.bsdfEvaluation.pdf);
			}
			swi.radiance    = prd.throughput * weightMis * matEval.bsdfEvaluation.bsdf * lightSample.radianceOverPdf;
			swi.direction   = lightSample.direction;
			swi.distance    = lightSample.distance - params.settings.renderer.minClamp;
			swi.depth       = prd.depth + 1;
			swi.originPixel = prd.originPixel;
			swi.origin      = prd.hitProperties.position;
			swi.seed        = prd.seed;
			swi.mediumIor   = prd.mediumIor;

			if (DEBUG_PRINT_INFO && math::length(swi.radiance) > 1000)
			{
				printf(
					   "Radiance Light Sample over 1000:"
					   "\t pixel %i\n"
					   "\t depth %i\n"
					   "\t weightMis %f\n"
					   "\t bsdf [%f, %f, %f]\n"
					   "\t bsdf prob %f\n"
					   "\t radianceOverPdf [%f, %f, %f]\n"
					   "\t lightSample.pdf %f\n"
					   "\t distance %f\n"
					   "\t throughput [%f %f %f]\n"
					   "\t total Radiance  [%f %f %f]\n"
					   , prd.originPixel
					   , prd.depth
					   , weightMis
					   , matEval.bsdfEvaluation.bsdf.x, matEval.bsdfEvaluation.bsdf.y, matEval.bsdfEvaluation.bsdf.z
					   , matEval.bsdfEvaluation.bsdfPdf
					   , lightSample.radianceOverPdf.x, lightSample.radianceOverPdf.y, lightSample.radianceOverPdf.z
					   , lightSample.pdf
					   , lightSample.distance
					   , prd.throughput.x, prd.throughput.y, prd.throughput.z
					   , swi.radiance.x, swi.radiance.y, swi.radiance.z
					  );
			}

			/*printf("Next Event mis Weight %f, sample Prob %f, contribution [%f %f %f]\n",
				weightMis, lightSample.pdf, swi.radiance.x, swi.radiance.y, swi.radiance.z
			);*/

			if (networkBounceData)
			{
				networkBounceData->lightSample.bsdf       = matEval.bsdfEvaluation.bsdf;
				networkBounceData->lightSample.bsdfProb   = matEval.bsdfEvaluation.bsdfPdf;
				networkBounceData->lightSample.wi         = lightSample.direction;
				networkBounceData->lightSample.Li         = lightSample.radiance;
				networkBounceData->lightSample.LiOverProb = lightSample.radianceOverPdf; // This and the next might be adjusted
				networkBounceData->lightSample.wiProb     = lightSample.pdf;
				networkBounceData->lightSample.misWeight  = weightMis;
				networkBounceData->lightSample.valid      = false;
			}

			if (params.debugData->debugPixel == prd.originPixel)
			{
				params.debugData->bounceData[prd.depth].lsWeightMis  = weightMis;
				params.debugData->bounceData[prd.depth].lsPdf        = lightSample.pdf;
				params.debugData->bounceData[prd.depth].lsBsdfMisPdf = matEval.bsdfEvaluation.bsdfPdf;
				params.debugData->bounceData[prd.depth].lsBsdf       = matEval.bsdfEvaluation.bsdf;
				params.debugData->bounceData[prd.depth].lsWi         = lightSample.direction;
				params.debugData->bounceData[prd.depth].lsLiOverPdf  = lightSample.radianceOverPdf;
				params.debugData->bounceData[prd.depth].isLsSample   = true;
			}
		}

		return swi;
	}

	__forceinline__ __device__ void evaluateEmission(mdl::MaterialEvaluation& matEval, RayWorkItem& prd, const LaunchParams& params, BounceData* networkBounceData)
	{
		if (params.debugData->debugPixel == prd.originPixel)
		{
			params.debugData->bounceData[prd.depth].isSeSample = false;
		}
		if (matEval.edf.isValid)
		{
			const MeshLightAttributesData* attributes = reinterpret_cast<MeshLightAttributesData*>(prd.hitProperties.lightData->attributes);
			const float                    area       = attributes->totalArea;
			// We compute the solid angle measure of selecting this light to compare with the pdf of the bsdf which is over directions.
			float misWeight   = 1.0f;
			bool  computedMis = false;
			if (
				prd.depth > 0 &&
				isMIS(&params, prd.originPixel) &&
				!(prd.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR))
			{
				matEval.edf.pdf = prd.hitDistance * prd.hitDistance / (area * matEval.edf.cos * (float)params.numberOfLights);
				if (isinf(matEval.edf.pdf) || isnan(matEval.edf.pdf) || matEval.edf.pdf <= 0.0f)
				{
					matEval.edf.pdf = 0.0f;
				}
				misWeight   = utl::heuristic(prd.pdf, matEval.edf.pdf);
				computedMis = true;
			}
			// Power (flux) [W] divided by light area gives radiant exitance [W/m^2].
			const float       factor          = (matEval.edf.mode == 0) ? 1.0f : 1.0f / area;
			const math::vec3f emittedRadiance = matEval.edf.intensity * matEval.edf.edf * factor;
			math::vec3f       radiance        = prd.throughput * emittedRadiance * misWeight;
			prd.radiance += radiance;

			if (DEBUG_PRINT_INFO && math::length(radiance) > 2000)
			{
				printf(
					   "Implicit Light Sample over 1000:"
					   "\t pixel %i\n"
					   "\t depth %i\n"
					   "\t weightMis %f\n"
					   "\t throughput [%f %f %f]\n"
					   "\t emittedRadiance [%f %f %f]\n"
					   "\t matEval ed pdf %f\n"
					   "\t matEval ed cos %f\n"
					   "\t hitDistance %f\n"
					   "\t area %f\n"
					   "\t numberOfLights %i\n"
					   "\t prd pdf %f\n"
					   "\t radiance [%f %f %f]\n"
					   , prd.originPixel
					   , prd.depth
					   , misWeight
					   , prd.throughput.x, prd.throughput.y, prd.throughput.z
					   , emittedRadiance.x, emittedRadiance.y, emittedRadiance.z
					   , matEval.edf.pdf
					   , matEval.edf.cos
					   , prd.hitDistance
					   , area
					   , params.numberOfLights
					   , prd.pdf
					   , radiance.x, radiance.y, radiance.z
					  );
			}

			/*math::vec3f weightedContribution = prd.throughput * misWeight * emittedRadiance;
			printf("Surface Event mis Weight %f, sample Prob %f, contribution [%f %f %f]\n",
				misWeight, prd.pdf, weightedContribution.x, weightedContribution.y, weightedContribution.z
			);*/

			if (networkBounceData)
			{
				networkBounceData->surfaceEmission.Le        = emittedRadiance;
				networkBounceData->surfaceEmission.misWeight = misWeight;
			}

			if (params.debugData->debugPixel == prd.originPixel)
			{
				params.debugData->bounceData[prd.depth].seWeightMis   = misWeight;
				params.debugData->bounceData[prd.depth].rayPdf        = prd.pdf;
				params.debugData->bounceData[prd.depth].sePdf         = matEval.edf.pdf;
				params.debugData->bounceData[prd.depth].seEdf         = matEval.edf.edf;
				params.debugData->bounceData[prd.depth].seIntensity   = matEval.edf.intensity;
				params.debugData->bounceData[prd.depth].isMisComputed = computedMis;
				params.debugData->bounceData[prd.depth].seEventType   = prd.eventType;
				params.debugData->bounceData[prd.depth].isSeSample    = true;
				params.debugData->bounceData[prd.depth].cond1         = prd.depth > 0;
				params.debugData->bounceData[prd.depth].cond2         = isMIS(&params, prd.originPixel);
				params.debugData->bounceData[prd.depth].cond3         = !(prd.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR);
			}
		}
	}

	__forceinline__ __device__ bool russianRoulette(RayWorkItem& prd, const LaunchParams& params, float& continuationProbability)
	{
		continuationProbability = 1.0f;
		if (params.settings.renderer.useRussianRoulette && 2 <= prd.depth) // Start termination after a minimum number of bounces.
		{
			continuationProbability = fminf(1.0f, fmaxf(fmaxf(prd.throughput.x, prd.throughput.y), prd.throughput.z));

			if (continuationProbability == 0.0f || continuationProbability < rng(prd.seed)) // Paths with lower probability to continue are terminated earlier.
			{
				return false;
			}
		}
		return true;
	}

	__forceinline__ __device__ bool bsdfSample(RayWorkItem& prd, const mdl::MaterialEvaluation& matEval, const LaunchParams& params, const float& continuationProbability, BounceData* networkBounceData)
	{
		const bool doNextBounce = !isDirectLighting(&params, prd.originPixel) && prd.depth + 1 < params.settings.renderer.maxBounces;
		if (doNextBounce && (matEval.bsdfSample.eventType != mi::neuraylib::BSDF_EVENT_ABSORB))
		{
			prd.direction = matEval.bsdfSample.nextDirection; // Continuation direction.
			prd.throughput *= (matEval.bsdfSample.bsdfOverPdf / continuationProbability);
			prd.pdf       = matEval.bsdfSample.pdf;
			prd.eventType = matEval.bsdfSample.eventType;

			if (!matEval.isThinWalled && (prd.eventType & mi::neuraylib::BSDF_EVENT_TRANSMISSION) != 0)
			{
				if (prd.hitProperties.isFrontFace) // Entered a volume. 
				{
					prd.mediumIor = matEval.ior;
				}
				else // if !isFrontFace. Left a volume.
				{
					prd.mediumIor = 1.0f;
				}
			}
			{
				const math::vec3f evalBsdfOverPdf = (matEval.bsdfSample.bsdfEvaluation.bsdf / matEval.bsdfSample.bsdfEvaluation.bsdfPdf);
				const float       diff            = math::length(evalBsdfOverPdf - matEval.bsdfSample.bsdfOverPdf);
				bool              sign            = diff < 0.0f;
				const math::vec3f debugColor      = diff < 0.0f ? math::vec3f(0.0f, 0.0f, -1.0f * diff) : math::vec3f(diff, 0.0f, 0.0f);
				addDebug01(debugColor, prd.originPixel, &params);
			}
			if (neuralNetworkActive(&params))
			{
				networkBounceData->bsdfSample.bsdf         = matEval.bsdfSample.bsdf;
				networkBounceData->bsdfSample.bsdfProb     = matEval.bsdfSample.bsdfPdf;
				networkBounceData->bsdfSample.bsdfOverProb = matEval.bsdfSample.bsdfOverPdf;
				networkBounceData->bsdfSample.wi           = matEval.bsdfSample.nextDirection;
				networkBounceData->bsdfSample.wiProb       = matEval.bsdfSample.pdf;
				networkBounceData->bsdfSample.isSpecular   = (matEval.bsdfSample.eventType & mi::neuraylib::BSDF_EVENT_SPECULAR);
			}

			return true;
		}

		return false;
	}

	__forceinline__ __device__ void nextWork(const TraceWorkItem& twi, const ShadowWorkItem& swi, const LaunchParams& params)
	{
		if (twi.extendRay)
		{
			params.queues.radianceTraceQueue->Push(twi);
		}
		else if (twi.radiance != math::vec3f(0.0f))
		{
			AccumulationWorkItem awi{};
			awi.radiance    = twi.radiance;
			awi.originPixel = twi.originPixel;
			awi.depth       = twi.depth;
			params.queues.accumulationQueue->Push(awi);
		}
		if (swi.distance > 0.0f)
		{
			params.queues.shadowQueue->Push(swi);
		}
	}

	__forceinline__ __device__ void setGBuffer(const RayWorkItem& prd, const LaunchParams& params)
	{
		if (prd.depth == 0)
		{
			params.frameBuffer.gBufferHistory[prd.originPixel].recordId(prd.hitProperties.instanceId);
			const bool smoothGBuffer = false;
			if (smoothGBuffer)
			{
				const float value                           = params.frameBuffer.gBuffer[prd.originPixel] * ((float)params.frameBuffer.samples[prd.originPixel] - 1.0f) + (float)prd.hitProperties.instanceId;
				params.frameBuffer.gBuffer[prd.originPixel] = value / (float)params.frameBuffer.samples[prd.originPixel];
			}
			else
			{
				params.frameBuffer.gBuffer[prd.originPixel] = (float)params.frameBuffer.gBufferHistory[prd.originPixel].mostFrequent;
			}
		}
	}

	__forceinline__ __device__ void shade(LaunchParams* params, RayWorkItem& prd, ShadowWorkItem& swi, TraceWorkItem& twi, const int& shadeQueueIndex = 0)
	{
		prd.hitProperties.calculate(params, prd.direction);
		BounceData* networkBounceData = nullptr;
		if (neuralNetworkActive(params))
		{
			networkBounceData                                        = &params->networkInterface->getAndReset(prd.originPixel, prd.depth);
			params->networkInterface->maxPathLength[prd.originPixel] = prd.depth;
			networkBounceData->hit.position                          = prd.hitProperties.position;
			networkBounceData->hit.normal                            = prd.hitProperties.shadingNormal;
			networkBounceData->hit.matId                             = prd.hitProperties.programCall;
			networkBounceData->hit.triangleId                        = prd.hitProperties.triangleId;
			networkBounceData->hit.instanceId                        = prd.hitProperties.instanceId;
			networkBounceData->wo                                    = -prd.direction;

			//networkBounceData->bsdfSample.valid = false;
			//networkBounceData->emission.valid = false;
			//networkBounceData->lightSample.valid = false;
		}


		setGBuffer(prd, *params);

		mdl::MaterialEvaluation matEval{};
		twi.extendRay = false;
		LightSample lightSample{};
		bool        extend = false;
		float       continuationProbability;
		const bool  isDirectOnly = isDirectLighting(params, prd.originPixel);
		if (prd.hitProperties.hasMaterial)
		{
			evaluateMaterialAndSampleLight(&matEval, &lightSample, *params, prd, shadeQueueIndex);
			if (!isDirectOnly)
			{
				evaluateEmission(matEval, prd, *params, networkBounceData);
			}

			if (prd.depth + 1 < params->settings.renderer.maxBounces)
			{
				swi = nextEventEstimation(matEval, lightSample, prd, *params, networkBounceData);

				if (!isDirectOnly)
				{
					prd.pdf = 0.0f;

					extend = russianRoulette(prd, *params, continuationProbability);
					if (extend)
					{
						extend = bsdfSample(prd, matEval, *params, continuationProbability, networkBounceData);
					}
				}
			}
		}
		setAuxiliaryRenderPassData(prd, matEval, params);

		if (params->debugData->debugPixel == prd.originPixel)
		{
			params->debugData->actualDepth                                   = prd.depth;
			params->debugData->bounceData[prd.depth].position                = prd.hitProperties.position;
			params->debugData->bounceData[prd.depth].throughput              = prd.throughput;
			params->debugData->bounceData[prd.depth].extend                  = extend;
			params->debugData->bounceData[prd.depth].continuationProbability = continuationProbability;
			if (extend)
			{
				params->debugData->bounceData[prd.depth].wi          = prd.direction;
				params->debugData->bounceData[prd.depth].wiPdf       = prd.pdf;
				params->debugData->bounceData[prd.depth].bsdfOverPdf = matEval.bsdfSample.bsdfOverPdf;
				params->debugData->bounceData[prd.depth].bsdf        = matEval.bsdfSample.bsdf;
				params->debugData->bounceData[prd.depth].bsdfPdf     = matEval.bsdfSample.bsdfPdf;
				params->debugData->bounceData[prd.depth].eventType   = matEval.bsdfSample.eventType;
			}
			else
			{
				params->debugData->bounceData[prd.depth].wi          = {0.0f, 0.0f, 0.0f};
				params->debugData->bounceData[prd.depth].bsdfOverPdf = {0.0f, 0.0f, 0.0f};
				params->debugData->bounceData[prd.depth].bsdf        = {0.0f, 0.0f, 0.0f};
				params->debugData->bounceData[prd.depth].bsdfPdf     = 0.0f;
				params->debugData->bounceData[prd.depth].wiPdf       = 0.0f;
			}
			if (!neuralSamplingActivated(params, prd.depth, prd.originPixel))
			{
				params->debugData->bounceData[prd.depth].neuralActive     = false;
				params->debugData->bounceData[prd.depth].isNeuralSample   = false;
				params->debugData->bounceData[prd.depth].samplingFraction = 0.0f;
				params->debugData->bounceData[prd.depth].neuralSample     = 0.0f;
				params->debugData->bounceData[prd.depth].neuralSamplePdf  = 0.0f;
			}
			else
			{
				params->debugData->bounceData[prd.depth].neuralActive = true;
			}
			params->debugData->bounceData[prd.depth].shadingNormal = prd.hitProperties.shadingNormal;
			params->debugData->bounceData[prd.depth].trueNormal    = prd.hitProperties.trueNormal;
			params->debugData->bounceData[prd.depth].tangent       = prd.hitProperties.tangent;
			params->debugData->bounceData[prd.depth].bitangent     = prd.hitProperties.bitangent;
			params->debugData->bounceData[prd.depth].uv            = {prd.hitProperties.uv.x, prd.hitProperties.uv.y};
			params->debugData->bounceData[prd.depth].depth         = prd.depth;

			if (isDirectLighting(params, prd.originPixel) || isMIS(params, prd.originPixel))
			{
				params->debugData->bounceData[prd.depth].isLsSample = true;
			}
		}

		twi.seed        = prd.seed;
		twi.originPixel = prd.originPixel;
		twi.origin      = prd.hitProperties.position;
		twi.direction   = prd.direction;
		twi.radiance    = prd.radiance;
		twi.throughput  = prd.throughput;
		twi.mediumIor   = prd.mediumIor;
		twi.eventType   = prd.eventType;
		twi.pdf         = prd.pdf;
		twi.extendRay   = extend;
		twi.depth       = prd.depth + 1;
	}

	__forceinline__ __device__ bool transparentAnyHit(
		HitProperties&      hitProperties,
		const math::vec3f&  direction,
		const math::vec3f&  mediumIor,
		unsigned            seed,
		const LaunchParams* params
	)
	{
		hitProperties.calculate(params, direction);
		mdl::MdlRequest request;
		if (hitProperties.hasOpacity)
		{
			mdl::MaterialEvaluation matEval;
			if (hitProperties.argBlock == nullptr)
			{
				optixIgnoreIntersection();
				return false;
			}
			request.edf               = false;
			request.outgoingDirection = -direction;
			request.surroundingIor    = mediumIor;
			request.opacity           = true;
			request.seed              = &seed;
			request.hitProperties     = &hitProperties;

			evaluateMaterial(hitProperties.programCall, &request, &matEval);

			// Stochastic alpha test to get an alpha blend effect.
			// No need to calculate an expensive random number if the test is going to fail anyway.
			if (matEval.opacity < 1.0f && matEval.opacity <= rng(seed))
			{
				optixIgnoreIntersection();
				return false;
			}
			return true;
		}
		return false;
	}

	__forceinline__ __device__ bool transparentAnyHit(RayWorkItem* prd, const LaunchParams* params)
	{
		return transparentAnyHit(
								 prd->hitProperties,
								 prd->direction,
								 prd->mediumIor,
								 prd->seed,
								 params
								);
	}

	__forceinline__ __device__ int getAdaptiveSampleCount(const int& fbIndex, const LaunchParams* params)
	{
		int samplesPerLaunch = 1;
		if (params->settings.renderer.adaptiveSamplingSettings.active)
		{
			if (params->settings.renderer.adaptiveSamplingSettings.minAdaptiveSamples <= params->settings.renderer.iteration)
			{
				samplesPerLaunch = params->frameBuffer.noiseBuffer[fbIndex].adaptiveSamples;
				if (samplesPerLaunch <= 0) //direct miss
				{
					return 0;
				}
			}
		}
		return samplesPerLaunch;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////// Integrator Kernel Launchers //////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	__forceinline__ __device__ void resetQueues(const LaunchParams* params)
	{
		params->queues.radianceTraceQueue->Reset();
		params->queues.shadeQueue->Reset();
		params->queues.shadowQueue->Reset();
		params->queues.escapedQueue->Reset();
		params->queues.accumulationQueue->Reset();
		if (neuralNetworkActive(params))
		{
			params->networkInterface->inferenceData->reset();
		}
	}

	__forceinline__ __device__ void fillCounters(const LaunchParams* params)
	{
		params->queues.queueCounters->traceQueueCounter        = params->queues.radianceTraceQueue->getCounter();
		params->queues.queueCounters->shadeQueueCounter        = params->queues.shadeQueue->getCounter();
		params->queues.queueCounters->shadowQueueCounter       = params->queues.shadowQueue->getCounter();
		params->queues.queueCounters->escapedQueueCounter      = params->queues.escapedQueue->getCounter();
		params->queues.queueCounters->accumulationQueueCounter = params->queues.accumulationQueue->getCounter();
	}

	__forceinline__ __device__ void wfInitRayEntry(const int id, const LaunchParams* params)
	{
		if (id == 0)
		{
			resetQueues(params);

			if (params->settings.renderer.iteration <= 0)
			{
				fillCounters(params);
			}
		}


		cleanFrameBuffer(id, params);
		const int samplesPerLaunch = getAdaptiveSampleCount(id, params);
		if (samplesPerLaunch == 0) return;
		TraceWorkItem twi;
		for (int i = 0; i < samplesPerLaunch; i++)
		{
			if (neuralNetworkActive(params))
			{
				params->networkInterface->reset(id);
			}

			params->debugData->reset();
			twi.seed = tea<4>(id + i, params->settings.renderer.iteration + *params->frameID);
			generateCameraRay(id, params, twi);

			params->queues.radianceTraceQueue->Push(twi);
		}
	}

	__forceinline__ __device__ void handleShading(int queueWorkId, LaunchParams& params)
	{
		if (queueWorkId == 0)
		{
			params.queues.radianceTraceQueue->Reset();
		}

		if (params.queues.shadeQueue->Size() <= queueWorkId)
			return;

		RayWorkItem    prd = (*params.queues.shadeQueue)[queueWorkId];
		ShadowWorkItem swi;
		TraceWorkItem  twi;
		shade(&params, prd, swi, twi, queueWorkId);
		nextWork(twi, swi, params);
	}

	__forceinline__ __device__ void wfAccumulateEntry(const int queueWorkId, const LaunchParams* params)
	{
		/*if (queueWorkId == 0)
		{
			params->shadeQueue->Reset();
			params->shadowQueue->Reset();
			params->escapedQueue->Reset();
		}*/
		if (queueWorkId >= params->queues.accumulationQueue->Size())
		{
			return;
		}
		const AccumulationWorkItem awi = (*params->queues.accumulationQueue)[queueWorkId];
		accumulateRay(awi, params);
	}

	__forceinline__ __device__ void wfEscapedEntry(const int id, const LaunchParams* params)
	{
		if (id >= params->queues.escapedQueue->Size())
			return;

		EscapedWorkItem ewi = (*params->queues.escapedQueue)[id];

		const math::vec3f stepRadiance = missShader(ewi, params);

		AccumulationWorkItem awi;
		awi.originPixel = ewi.originPixel;
		awi.radiance    = ewi.radiance;
		awi.depth       = ewi.depth;
		params->queues.accumulationQueue->Push(awi);
	}

	#ifdef ARCHITECTURE_OPTIX

	__forceinline__ __device__ void optixHitProperties(RayWorkItem* prd)
	{
		const float2 baricenter2D    = optixGetTriangleBarycentrics();
		prd->hitDistance             = optixGetRayTmax();
		const auto        baricenter = math::vec3f(1.0f - baricenter2D.x - baricenter2D.y, baricenter2D.x, baricenter2D.y);
		const math::vec3f position   = prd->hitProperties.position + prd->direction * prd->hitDistance;
		prd->hitProperties.init(optixGetInstanceId(), optixGetPrimitiveIndex(), baricenter, position);
	}

	template <typename T>
	__forceinline__ __device__ bool trace(math::vec3f& origin, math::vec3f& direction, const float distance, T* rd, const int sbtIndex, LaunchParams& params, const OptixRayFlags flags = OPTIX_RAY_FLAG_NONE)
	{
		math::vec2ui payload = splitPointer(rd);

		optixTrace(
				   params.topObject,
				   origin,
				   direction, // origin, direction
				   params.settings.renderer.minClamp,
				   distance,
				   0.0f, // tmin, tmax, time
				   static_cast<OptixVisibilityMask>(0xFF),
				   flags,    //OPTIX_RAY_FLAG_NONE,
				   sbtIndex, //SBT Offset
				   0,        // SBT stride
				   0,        // missSBTIndex
				   payload.x,
				   payload.y);

		if (payload.x == 0)
		{
			return false;
		}
		return true;
	}

	__forceinline__ __device__ void elaborateShadowTrace(ShadowWorkItem& swi, LaunchParams& params, const ArchitectureType architecture = A_WAVEFRONT_CUDA_SHADE)
	{
		bool hit = trace(swi.origin, swi.direction, swi.distance, &swi, 1, params, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
		if (hit == false)
		{
			AccumulationWorkItem awi;
			awi.radiance    = swi.radiance;
			awi.depth       = swi.depth;
			awi.originPixel = swi.originPixel;
			if (architecture == A_FULL_OPTIX)
			{
				accumulateRay(awi, &params);
			}
			else
			{
				params.queues.accumulationQueue->Push(awi);
				if (neuralNetworkActive(&params))
				{
					params.networkInterface->validateLightSample(swi.originPixel, swi.depth - 1); // stored at previous depth, where it was sampled
				}
			}
		}
	}

	__forceinline__ __device__ void elaborateRadianceTrace(TraceWorkItem& twi, LaunchParams& params, ArchitectureType architecture = A_WAVEFRONT_CUDA_SHADE)
	{
		RayWorkItem prd{};
		prd.seed                   = twi.seed;
		prd.originPixel            = twi.originPixel;
		prd.depth                  = twi.depth;
		prd.hitProperties.position = twi.origin;
		prd.direction              = twi.direction;
		prd.radiance               = twi.radiance;
		prd.throughput             = twi.throughput;
		prd.mediumIor              = twi.mediumIor;
		prd.eventType              = twi.eventType;
		prd.pdf                    = twi.pdf;

		bool hit = trace(twi.origin, twi.direction, params.settings.renderer.maxClamp, &prd, 0, params);

		if (architecture == A_FULL_OPTIX)
		{
			if (hit)
			{
				ShadowWorkItem swi;
				shade(&params, prd, swi, twi);
				if (swi.distance > 0.0f)
				{
					elaborateShadowTrace(swi, params, A_FULL_OPTIX);
				}
				if (!twi.extendRay)
				{
					AccumulationWorkItem awi;
					awi.radiance    = prd.radiance;
					awi.depth       = prd.depth;
					awi.originPixel = prd.originPixel;
					accumulateRay(awi, &params);
				}
			}
			else
			{
				EscapedWorkItem ewi{};
				ewi.seed        = prd.seed;
				ewi.originPixel = prd.originPixel;
				ewi.depth       = prd.depth;
				ewi.direction   = prd.direction;
				ewi.radiance    = prd.radiance;
				ewi.throughput  = prd.throughput;
				ewi.eventType   = prd.eventType;
				ewi.pdf         = prd.pdf;
				missShader(ewi, &params);
				AccumulationWorkItem awi;
				awi.radiance    = ewi.radiance;
				awi.depth       = ewi.depth;
				awi.originPixel = ewi.originPixel;
				accumulateRay(awi, &params);
				twi.extendRay = false;
			}
		}
		else
		{
			if (hit)
			{
				int shadeQueueIndex = params.queues.shadeQueue->Push(prd);
				if (neuralSamplingActivated(&params, prd.depth, 0))
				{
					prd.hitProperties.calculateForInferenceQuery(&params);
					BounceData queryData;

					queryData.hit.position   = prd.hitProperties.position;
					queryData.wo             = -prd.direction;
					queryData.hit.normal     = prd.hitProperties.shadingNormal;
					queryData.hit.instanceId = prd.hitProperties.instanceId;
					queryData.hit.triangleId = prd.hitProperties.triangleId;
					queryData.hit.matId      = prd.hitProperties.programCall;
					params.networkInterface->inferenceData->registerQuery(shadeQueueIndex, queryData);
				}
			}
			else
			{
				EscapedWorkItem ewi{};
				ewi.seed        = prd.seed;
				ewi.originPixel = prd.originPixel;
				ewi.depth       = prd.depth;
				ewi.direction   = prd.direction;
				ewi.radiance    = prd.radiance;
				ewi.throughput  = prd.throughput;
				ewi.eventType   = prd.eventType;
				ewi.pdf         = prd.pdf;
				params.queues.escapedQueue->Push(ewi);
			}
		}
	}

	__forceinline__ __device__ void wfTraceRadianceEntry(const int queueWorkId, LaunchParams& params)
	{
		if (queueWorkId == 0)
		{
			params.queues.shadeQueue->Reset();
			if (neuralNetworkActive(&params))
			{
				params.networkInterface->inferenceData->reset();
			}
		}

		int radianceTraceQueueSize = params.queues.radianceTraceQueue->Size();
		if (radianceTraceQueueSize <= queueWorkId)
			return;

		TraceWorkItem twi = (*params.queues.radianceTraceQueue)[queueWorkId];
		// Shadow Trace
		const int  maxTraceQueueSize = params.frameBuffer.frameSize.x * params.frameBuffer.frameSize.y;
		const bool isLongPath        = (float)radianceTraceQueueSize <= params.settings.wavefront.longPathPercentage * (float)maxTraceQueueSize;
		if (!(params.settings.wavefront.useLongPathKernel && isLongPath))
		{
			elaborateRadianceTrace(twi, params);
		}
		else
		{
			int remainingBounces = params.settings.renderer.maxBounces - twi.depth;
			for (int i = 0; i < remainingBounces; i++)
			{
				elaborateRadianceTrace(twi, params, A_FULL_OPTIX);
				if (!twi.extendRay)
				{
					break;
				}
			}
		}
	}

	__forceinline__ __device__ void wfTraceShadowEntry(const int queueWorkId, LaunchParams& params)
	{
		if (params.queues.shadowQueue->Size() <= queueWorkId)
			return;

		ShadowWorkItem swi = (*params.queues.shadowQueue)[queueWorkId];
		// Shadow Trace

		elaborateShadowTrace(swi, params);
	}

	#endif
}

#endif
