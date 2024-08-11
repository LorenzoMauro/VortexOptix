#pragma once
#ifndef NEW_NETWORK_INTERFACE
#define NEW_NETWORK_INTERFACE

#define CUDA_INTERFACE
#include "NeuralNetworks/Distributions/Mixture.h"
#include "NetworkInterfaceStructs.h"
#include "NeuralNetworks/Config/NetworkSettings.h"
#include "Device/DevicePrograms/nvccUtils.h"
#include "Device/DevicePrograms/ToneMapper.h"

namespace vtx
{
//#define PRINT_CONDITION(condition) printf("Invalid condition file %s line %d condition %s\n", __FILE__, __LINE__, #condition)
#define PRINT_CONDITION(condition)

#define printInvalidCondition(condition) if (condition) {PRINT_CONDITION(condition); return;}
	struct Samples
	{
		math::vec3f* position = nullptr;
		math::vec3f* wo = nullptr;
		math::vec3f* normal = nullptr;
		float* instanceId = nullptr;
		float* triangleId = nullptr;
		float* matId = nullptr;
		math::vec3f* Lo = nullptr;
		math::vec3f* wi = nullptr;
		math::vec3f* Li = nullptr;
		math::vec3f* bsdf = nullptr;
		float* bsdfProb = nullptr;
		float* wiProb = nullptr;
		int* validSamples = nullptr;
		int maxAllowedBounces = 0;

		cudaFunction static bool isNotValid(const float value)
		{
			return (isnan(value) || isinf(value));
		}

		cudaFunction static bool isNotValid(const math::vec3f& value)
		{
			return (math::isNan(value) || math::isInf(value));
		}

		cudaFunction void registerBsdfSample(const int index, const BounceData& data)
		{
			validSamples[index] = 0;
			printInvalidCondition(isNotValid(data.hit.position));
			printInvalidCondition(isNotValid(data.wo));
			printInvalidCondition(isNotValid(data.hit.normal));
			printInvalidCondition(isNotValid((float)data.hit.instanceId));
			printInvalidCondition(isNotValid((float)data.hit.triangleId));
			printInvalidCondition(isNotValid((float)data.hit.matId));
			printInvalidCondition(isNotValid(data.bsdfSample.Lo));
			printInvalidCondition(isNotValid(data.bsdfSample.wi));
			printInvalidCondition(isNotValid(data.bsdfSample.Li));
			printInvalidCondition(isNotValid(data.bsdfSample.bsdf));
			printInvalidCondition(isNotValid(data.bsdfSample.bsdfProb));
			printInvalidCondition(gdt::isEqual(data.bsdfSample.bsdfProb ,0.0f));
			printInvalidCondition(gdt::isEqual(data.bsdfSample.wiProb ,0.0f));
			printInvalidCondition(gdt::isEqual(math::length(data.hit.normal) ,0.0f));
			printInvalidCondition(gdt::isEqual(math::length(data.wo) ,0.0f));

			position[index] = data.hit.position;
			wo[index] = data.wo;
			normal[index] = data.hit.normal;
			instanceId[index] = (float)data.hit.instanceId;
			triangleId[index] = (float)data.hit.triangleId;
			matId[index] = (float)data.hit.matId;
			Lo[index] = data.bsdfSample.Li * data.bsdfSample.bsdf;
			wi[index] = data.bsdfSample.wi;
			Li[index] = data.bsdfSample.Li;
			bsdf[index] = data.bsdfSample.bsdf;
			bsdfProb[index] = data.bsdfSample.bsdfProb;
			wiProb[index] = data.bsdfSample.wiProb;
			validSamples[index] = 1;
		}

		cudaFunction void registerLightSample(const int index, const BounceData& data)
		{
			if (!data.lightSample.valid)
			{
				return;
			}
			printInvalidCondition(isNotValid(data.hit.position));
			printInvalidCondition(isNotValid(data.wo));
			printInvalidCondition(isNotValid(data.hit.normal));
			printInvalidCondition(isNotValid((float)data.hit.instanceId));
			printInvalidCondition(isNotValid((float)data.hit.triangleId));
			printInvalidCondition(isNotValid((float)data.hit.matId));
			printInvalidCondition(isNotValid(data.lightSample.Lo));
			printInvalidCondition(isNotValid(data.lightSample.wi));
			printInvalidCondition(isNotValid(data.lightSample.Li));
			printInvalidCondition(isNotValid(data.lightSample.bsdf));
			printInvalidCondition(isNotValid(data.lightSample.bsdfProb));
			printInvalidCondition(gdt::isEqual(data.lightSample.bsdfProb, 0.0f));
			printInvalidCondition(gdt::isEqual(data.lightSample.wiProb, 0.0f));
			printInvalidCondition(gdt::isEqual(math::length(data.hit.normal), 0.0f));
			printInvalidCondition(gdt::isEqual(math::length(data.wo), 0.0f));

			position[index] = data.hit.position;
			wo[index] = data.wo;
			normal[index] = data.hit.normal;
			instanceId[index] = (float)data.hit.instanceId;
			triangleId[index] = (float)data.hit.triangleId;
			matId[index] = (float)data.hit.matId;
			Lo[index] = data.lightSample.Li * data.lightSample.bsdf;
			wi[index] = data.lightSample.wi;
			Li[index] = data.lightSample.Li;
			bsdf[index] = data.lightSample.bsdf;
			bsdfProb[index] = data.lightSample.bsdfProb;
			wiProb[index] = data.lightSample.wiProb;
			validSamples[index] = 1;
		}

		cudaFunction int getIndex(const int pixelId, const int bounce, bool isLightSample)
		{
			//return cuAtomicAdd(size, 1);
			int id = pixelId * maxAllowedBounces * 2 + bounce;
			if (isLightSample)
			{
				id++;
			}
			return id;
		}

		cudaFunction void registerSample(const int pixelId, const int maxBounce, const int bounce, const BounceData& p, const network::config::NetworkSettings& netSettings)
		{
			const int bsdfSampleIndex = getIndex(pixelId, bounce, false);
			const int lightSampleIndex = getIndex(pixelId, bounce, true);
			if (
				(netSettings.trainingBatchGenerationSettings.skipSpecular == false || !p.bsdfSample.isSpecular) &&
				bounce < maxBounce &&
				(netSettings.trainingBatchGenerationSettings.limitToFirstBounce == false || bounce == 0)
				)
			{
				if (
					(!netSettings.trainingBatchGenerationSettings.onlyNonZero || !math::isZero(p.bsdfSample.Lo))
					&& bounce != maxBounce
					)
				{
					registerBsdfSample(bsdfSampleIndex, p);
				}
				if (
					(!netSettings.trainingBatchGenerationSettings.onlyNonZero || !math::isZero(p.lightSample.Lo))
					&& netSettings.trainingBatchGenerationSettings.trainOnLightSample
					)
				{
					registerLightSample(lightSampleIndex, p);
				}
				
			}
		}

		cudaFunction void resetPixel(const int pixelId)
		{
			for (int i = 0; i < maxAllowedBounces * 2; i++)
			{
				validSamples[pixelId * maxAllowedBounces * 2 + i] = 0;
			}
		}
	};

	struct TrainingData
	{
		math::vec3f* position = nullptr;
		math::vec3f* wo = nullptr;
		math::vec3f* normal = nullptr;
		float* instanceId = nullptr;
		float* triangleId = nullptr;
		float* matId = nullptr;
		math::vec3f* Lo = nullptr;
		math::vec3f* wi = nullptr;
		math::vec3f* Li = nullptr;
		math::vec3f* bsdf = nullptr;
		float* bsdfProb = nullptr;
		float* wiProb = nullptr;

		int nAlloc;
		int* size = nullptr;

		cudaFunction void reset()
		{
			*size = 0;
		}

		cudaFunction void buildTrainingData(const int id, const Samples* samples, bool toneMapRadiance)
		{
			if(samples->validSamples[id]!=1)
			{
				return;
			}
			const int index = cuAtomicAdd(size, 1);
			position[index] = samples->position[id];
			wo[index] = samples->wo[id];
			normal[index] = samples->normal[id];
			instanceId[index] = samples->instanceId[id];
			triangleId[index] = samples->triangleId[id];
			matId[index] = samples->matId[id];
			Lo[index] = samples->Lo[id];
			wi[index] = samples->wi[id];
			Li[index] = samples->Li[id];
			if (toneMapRadiance)
			{
				Lo[index] = toneMap({}, Lo[index]);
				Li[index] = toneMap({}, Li[index]);
			}
			bsdf[index] = samples->bsdf[id];
			bsdfProb[index] = samples->bsdfProb[id];
			wiProb[index] = samples->wiProb[id];
		}
	};

	struct InferenceData
	{
		math::vec3f* position = nullptr;
		math::vec3f* wo = nullptr;
		math::vec3f* normal = nullptr;
		float* instanceId = nullptr;
		float* triangleId = nullptr;
		float* matId = nullptr;

		float* distributionParameters = nullptr;
		float* mixtureWeights = nullptr;
		float* samplingFractionArray = nullptr;

		int* size = nullptr;

		int                        nAlloc;
		network::config::DistributionType  distributionType;
		int                        mixtureSize;

		cudaFunction void registerQuery(const int queryId, const BounceData& data)
		{
			cuAtomicAdd(size, 1);
			position[queryId] = data.hit.position;
			wo[queryId] = data.wo;
			normal[queryId] = data.hit.normal;
			instanceId[queryId] = (float)data.hit.instanceId;
			triangleId[queryId] = (float)data.hit.triangleId;
			matId[queryId] = (float)data.hit.matId;
		}

		cudaFunction void reset()
		{
			*size = 0;
		}

		cudaFunction float evaluate(const int& idx, const math::vec3f& action)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			const float  sampleProb = distribution::Mixture::evaluate(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType, action);
			return sampleProb;
		}

		cudaFunction math::vec4f sample(const int& idx, unsigned& seed) const
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			math::vec3f  sample = distribution::Mixture::sample(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType, seed);
			const float  sampleProb = distribution::Mixture::evaluate(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType, sample);
			return { sample, sampleProb };
		}

		cudaFunction void getSampleMixtureParameters(const int& idx, const float*& sampleMixtureParams, const float*& sampleMixtureWeights) const
		{
			const int mixtureParamsOffset = idx * mixtureSize * distribution::Mixture::getDistributionParametersCount(distributionType);
			const int mixtureWeightsOffset = idx * mixtureSize;
			sampleMixtureParams = distributionParameters + mixtureParamsOffset;
			sampleMixtureWeights = mixtureWeights + mixtureWeightsOffset;
		}

		cudaFunction math::vec3f& getStatePosition(const int& idx)
		{
			return position[idx];
		}

		cudaFunction math::vec3f& getStateDirection(const int& idx)
		{
			return wo[idx];
		}

		cudaFunction math::vec3f& getStateNormal(const int& idx)
		{
			return normal[idx];
		}

		cudaFunction math::vec3f getMean(const int& idx)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			return distribution::Mixture::getAvarageAxis(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType);
		}

		cudaFunction float getConcentration(const int& idx)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			return distribution::Mixture::getAvarageConcentration(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType);
		}

		cudaFunction float& getSamplingFraction(const int& idx)
		{
			return samplingFractionArray[idx];
		}

		cudaFunction float getAnisotropy(const int& idx)
		{
			const float* sampleMixtureParams = nullptr;
			const float* sampleMixtureWeights = nullptr;
			getSampleMixtureParameters(idx, sampleMixtureParams, sampleMixtureWeights);
			return distribution::Mixture::getAverageAnisotropy(sampleMixtureParams, sampleMixtureWeights, mixtureSize, distributionType);
		}
	};

	struct NetworkInterface
	{
		BounceData* path = nullptr;
		int* maxPathLength = nullptr;
		int maxAllowedPathLength = 0;

		Samples* samples = nullptr;
		TrainingData* trainingData = nullptr;
		InferenceData* inferenceData = nullptr;

		NetworkInterfaceDebugBuffers* debugBuffers = nullptr;

		cudaFunction BounceData& get(unsigned pixel, unsigned bounce)
		{
			return path[pixel*maxAllowedPathLength + bounce];
		}

		cudaFunction BounceData& getAndReset(unsigned pixel, unsigned bounce)
		{
			BounceData& d = get(pixel, bounce);
			//d.reset();
			//if(bounce < maxAllowedPathLength) get(pixel, bounce + 1).reset();
			return d;
		}

		cudaFunction void record(
			unsigned pixel
			, unsigned bounce
			, BounceData& data
		)
		{
			BounceData& d = get(pixel, bounce);
			d = data;
			maxPathLength[pixel] = bounce;
		}

		cudaFunction void validateLightSample(unsigned pixel, unsigned bounce)
		{
			BounceData& d = get(pixel, bounce);
			d.lightSample.valid = true;
		}

		cudaFunction __host__ void static bounceComputation(const int bounce, const int maxBounce, BounceData& p, const math::vec3f& nextPLo, const network::config::NetworkSettings& netSettings)
		{
			if (bounce != maxBounce)
			{
				p.bsdfSample.Li = nextPLo;
				if (netSettings.trainingBatchGenerationSettings.weightByPdf)
				{
					p.bsdfSample.Lo = p.bsdfSample.bsdfOverProb * p.bsdfSample.Li;
				}
				else
				{
					p.bsdfSample.Lo = p.bsdfSample.bsdf * p.bsdfSample.Li;
				}
			}

			// Light Sample Contribution
			p.lightSample.Lo = 0.0f;
			if (p.lightSample.valid && netSettings.trainingBatchGenerationSettings.useLightSample)
			{
				if (netSettings.trainingBatchGenerationSettings.weightByPdf)
				{
					p.lightSample.Lo = p.lightSample.bsdf * p.lightSample.LiOverProb;
				}
				else
				{
					p.lightSample.Lo = p.lightSample.bsdf * p.lightSample.Li;
				}
				if (netSettings.trainingBatchGenerationSettings.weightByMis)
				{
					p.lightSample.Lo *= p.lightSample.misWeight;
				}
			}

			p.Lo += p.bsdfSample.Lo;
			p.Lo += p.lightSample.Lo;

			if (netSettings.trainingBatchGenerationSettings.weightByMis)
			{
				p.Lo += p.surfaceEmission.Le * p.surfaceEmission.misWeight;
			}
			else
			{
				p.Lo += p.surfaceEmission.Le;
			}
		}

		cudaFunction void finalizePath(unsigned pixel, const network::config::NetworkSettings& netSettings, bool registerTrainingSample = true)
		{
			//samples->resetPixel(pixel);
			const int maxBounce = maxPathLength[pixel];
			if(maxBounce == -1)
			{

				return;
			}


			math::vec3f nextPLo = 0.0f;
			for(int bounce = maxBounce; bounce >= 0; bounce--)
			{
				BounceData& p = get(pixel, bounce);

				// Bsdf Sample Contribution
				bounceComputation(bounce, maxBounce, p, nextPLo, netSettings);
				nextPLo = p.Lo;

				if(registerTrainingSample)
				{
					samples->registerSample(pixel, maxBounce, bounce, p, netSettings);
				}
				
			}
			debugBuffers->filmBuffer[pixel] += math::vec4f{ nextPLo, 1.0f };
		}

		cudaFunction __host__ static void finalizePathStatic(const network::config::NetworkSettings& netSettings, int maxBounce, BounceData* data)
		{
			if (maxBounce == -1)
			{
				return;
			}

			math::vec3f nextPLo = 0.0f;
			for (int bounce = maxBounce-1; bounce >= 0; bounce--)
			{
				BounceData& p = data[bounce];

				// Bsdf Sample Contribution
				bounceComputation(bounce, maxBounce, p, nextPLo, netSettings);
				nextPLo = p.Lo;
			}
		}

		cudaFunction void reset(unsigned pixel)
		{
			maxPathLength[pixel] = -1;
			for(int bounce = 0; bounce < maxAllowedPathLength; ++bounce)
			{
				get(pixel, bounce).reset();
			}
		}
	};
}
#endif