#pragma once
#ifndef DEBUG_DOWNLOAD_DATA_H
#define DEBUG_DOWNLOAD_DATA_H
#include <vector>

#include "cuda_runtime.h"
#include "core/math.h"
#include "NeuralNetworks/Config/DistributionConfig.h"

namespace vtx
{
	struct DebugBounceData
	{
		math::vec3f accumulatedRadiance;
		math::vec3f position;
		math::vec3f trueNormal;
		math::vec3f shadingNormal;
		math::vec3f tangent;
		math::vec3f bitangent;
		math::vec2f uv;
		int depth;

		// Surface Emission

		bool isSeSample;
		float seWeightMis;
		float rayPdf;
		float sePdf;
		math::vec3f seEdf;
		math::vec3f seIntensity;
		int seEventType;
		bool isMisComputed;
		bool cond1;
		bool cond2;
		bool cond3;

		//Bsdf Sample Data
		bool extend;
		math::vec3f throughput;
		math::vec3f wi;
		float wiPdf;
		math::vec3f bsdfOverPdf;
		math::vec3f bsdf;
		float bsdfPdf;
		math::vec3f bsdfSample;
		int eventType;

		bool neuralActive;
		bool isNeuralSample;
		float samplingFraction;
		math::vec3f neuralSample;
		float neuralSamplePdf;
		float* mixtureWeigths;
		float* mixtureParameters;

		//Light Sample Data
		float	lsNeuralPdf;
		float	lsBsdfPdf;
		bool	lsDoNeural;

		float       lsWeightMis;
		float       lsPdf;
		float       lsBsdfMisPdf;
		math::vec3f lsBsdf;
		math::vec3f lsWi;
		math::vec3f lsLiOverPdf;
		bool        isLsSample;
		float         continuationProbability;


		__forceinline__ __device__ void reset()
		{
			isSeSample = false;
			extend = false;
			isLsSample = false;
			isNeuralSample = false;
			accumulatedRadiance = math::vec3f(0.0f);
		}

	};

	struct DebugData
	{
		DebugBounceData* bounceData;
		int maxDepth;
		int actualDepth;
		int debugPixel;
		int debugDepth;

		__host__ static void prepare(
			const int maxDepth,
			const int debugPixel,
			const int debugDepth,
			const int mixtureSize,
			const network::config::DistributionType& type
		);

		__host__ static std::vector<DebugBounceData> getFromDevice();

		__forceinline__ __device__ void reset()
		{
			actualDepth = -1;
			for (int i = 0; i < maxDepth; i++)
			{
				bounceData[i].reset();
			}
		}

		__forceinline__ __device__ void copyMixtureData(
			const float* weigthsSrc,
			const float* paramsSrc,
			const int depth,
			const int mixtureSize, 
			const int mixtureParamCount)
		{
			float* weigthsDst = bounceData[depth].mixtureWeigths;
			float* paramsDst = bounceData[depth].mixtureParameters;
			for (int i = 0; i < mixtureSize; i++)
			{
				for (int j = 0; j < mixtureParamCount; j++)
				{
					paramsDst[i * mixtureParamCount + j] = paramsSrc[i * mixtureParamCount + j];
				}
				weigthsDst[i] = weigthsSrc[i];
			}
		}
	};

	}
#endif
