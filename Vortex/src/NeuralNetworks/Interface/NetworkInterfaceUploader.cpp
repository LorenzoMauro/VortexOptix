#include "NetworkInterfaceUploader.h"
#include "NetworkInterface.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"

namespace vtx
{
	TrainingData* uploadTrainingData(int maxTrainingData)
	{
		TrainingData data;
		device::TrainingDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.trainingDataBuffers;
		data.position = buffers.positionBuffer.alloc<math::vec3f>(maxTrainingData);
		data.wo = buffers.woBuffer.alloc<math::vec3f>(maxTrainingData);
		data.normal = buffers.normalBuffer.alloc<math::vec3f>(maxTrainingData);
		data.instanceId = buffers.instanceIdBuffer.alloc<float>(maxTrainingData);
		data.triangleId = buffers.triangleIdBuffer.alloc<float>(maxTrainingData);
		data.matId = buffers.matIdBuffer.alloc<float>(maxTrainingData);
		data.Lo = buffers.LoBuffer.alloc<math::vec3f>(maxTrainingData);
		data.wi = buffers.wiBuffer.alloc<math::vec3f>(maxTrainingData);
		data.Li = buffers.LiBuffer.alloc<math::vec3f>(maxTrainingData);
		data.bsdf = buffers.bsdfBuffer.alloc<math::vec3f>(maxTrainingData);
		data.bsdfProb = buffers.bsdfProbBuffer.alloc<float>(maxTrainingData);
		data.wiProb = buffers.wiProbBuffer.alloc<float>(maxTrainingData);
		data.size = buffers.sizeBuffer.upload(0);
		data.nAlloc = maxTrainingData;
		return buffers.structBuffer.upload(data);
	}

	TrainingData* getPreviousTrainingData()
	{
		return onDeviceData->networkInterfaceData.resourceBuffers.trainingDataBuffers.structBuffer.castedPointer<TrainingData>();
	}

	Samples* uploadSamples(int maxTrainingData, int maxBounces)
	{
		Samples data;
		device::SamplesBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.samplesBuffers;
		data.position = buffers.positionBuffer.alloc<math::vec3f>(maxTrainingData);
		data.wo = buffers.woBuffer.alloc<math::vec3f>(maxTrainingData);
		data.normal = buffers.normalBuffer.alloc<math::vec3f>(maxTrainingData);
		data.instanceId = buffers.instanceIdBuffer.alloc<float>(maxTrainingData);
		data.triangleId = buffers.triangleIdBuffer.alloc<float>(maxTrainingData);
		data.matId = buffers.matIdBuffer.alloc<float>(maxTrainingData);
		data.Lo = buffers.LoBuffer.alloc<math::vec3f>(maxTrainingData);
		data.wi = buffers.wiBuffer.alloc<math::vec3f>(maxTrainingData);
		data.Li = buffers.LiBuffer.alloc<math::vec3f>(maxTrainingData);
		data.bsdf = buffers.bsdfBuffer.alloc<math::vec3f>(maxTrainingData);
		data.bsdfProb = buffers.bsdfProbBuffer.alloc<float>(maxTrainingData);
		data.wiProb = buffers.wiProbBuffer.alloc<float>(maxTrainingData);
		const std::vector<int> validSamples(maxTrainingData, 0);
		data.validSamples = buffers.validSamplesBuffer.upload(validSamples);
		data.maxAllowedBounces = maxBounces;
		return buffers.structBuffer.upload(data);
	}

	Samples* getPreviousSamples()
	{
		return onDeviceData->networkInterfaceData.resourceBuffers.samplesBuffers.structBuffer.castedPointer<Samples>();
	}

	InferenceData* uploadInferenceData(const int& numberOfPixels, const network::config::DistributionType& type, const int& mixtureSize)
	{
		InferenceData                 data;
		device::InferenceDataBuffers& buffers = onDeviceData->networkInterfaceData.resourceBuffers.inferenceDataBuffers;

		data.position = buffers.positionBuffer.alloc<math::vec3f>(numberOfPixels);
		data.wo = buffers.woBuffer.alloc<math::vec3f>(numberOfPixels);
		data.normal = buffers.normalBuffer.alloc<math::vec3f>(numberOfPixels);
		data.instanceId = buffers.instanceIdBuffer.alloc<float>(numberOfPixels);
		data.triangleId = buffers.triangleIdBuffer.alloc<float>(numberOfPixels);
		data.matId = buffers.matIdBuffer.alloc<float>(numberOfPixels);
		data.samplingFractionArray = buffers.samplingFractionArrayBuffer.alloc<float>(numberOfPixels);
		data.distributionParameters = buffers.distributionParametersBuffer.alloc<float>(numberOfPixels * mixtureSize * distribution::Mixture::getDistributionParametersCount(type));
		data.mixtureWeights = buffers.mixtureWeightsBuffer.alloc<float>(numberOfPixels * mixtureSize);
		data.size = buffers.sizeBuffer.upload(0);
		data.nAlloc = numberOfPixels;
		data.distributionType = type;
		data.mixtureSize = mixtureSize;

		return buffers.structBuffer.upload(data);
	}

	InferenceData* getPreviousInferenceData()
	{
		return onDeviceData->networkInterfaceData.resourceBuffers.inferenceDataBuffers.structBuffer.castedPointer<InferenceData>();
	}

	static int prev_numberOfPixels = 0;
	static int prev_maxBounce = 0;
	static int prev_maxTrainingDataSize = 0;
	static network::config::DistributionType prev_type = network::config::D_COUNT;
	static int prev_mixtureSize = 0;
	static bool wasAllocated = false;

	NetworkInterfaceDebugBuffers* uploadNetworkInterfaceDebugBuffers(const int& numberOfPixels)
	{
		NetworkInterfaceDebugBuffers data;
		auto& buffers = onDeviceData->networkInterfaceData.resourceBuffers.debugBuffers;
		data.inferenceDebugBuffer = buffers.inferenceDebugBuffer.alloc<math::vec3f>(numberOfPixels);
		data.filmBuffer = buffers.filmBuffer.alloc<math::vec4f>(numberOfPixels);
		return buffers.structBuffer.upload(data);
	}

	NetworkInterfaceDebugBuffers* getPreviousNetworkInterfaceDebugBuffers()
	{
		return onDeviceData->networkInterfaceData.resourceBuffers.debugBuffers.structBuffer.castedPointer<NetworkInterfaceDebugBuffers>();
	}

	NetworkInterface* uploadNetworkInterface(const int& numberOfPixels, int maxBounce, int maxTrainingDataSize, const network::config::DistributionType& type, const int& mixtureSize)
	{
		NetworkInterface data;
		data.trainingData = (maxTrainingDataSize != prev_maxTrainingDataSize) ? uploadTrainingData(maxTrainingDataSize) : getPreviousTrainingData();
		data.samples = (maxTrainingDataSize != prev_maxTrainingDataSize) ? uploadSamples(maxTrainingDataSize, maxBounce) : getPreviousSamples();
		data.inferenceData = (numberOfPixels != prev_numberOfPixels || type != prev_type || mixtureSize != prev_mixtureSize) ? uploadInferenceData(numberOfPixels, type, mixtureSize) : getPreviousInferenceData();
		data.debugBuffers = (numberOfPixels != prev_numberOfPixels ) ? uploadNetworkInterfaceDebugBuffers(numberOfPixels) : getPreviousNetworkInterfaceDebugBuffers();

		auto& buffers = onDeviceData->networkInterfaceData.resourceBuffers;
		if (maxBounce != prev_maxBounce || numberOfPixels != prev_numberOfPixels)
		{
			data.path = buffers.bounceDataBuffer.alloc<BounceData>(numberOfPixels * maxBounce);
		}
		else
		{
			data.path = buffers.bounceDataBuffer.castedPointer<BounceData>();
		}

		if (numberOfPixels != prev_numberOfPixels)
		{
			data.maxPathLength = buffers.maxPathLengthBuffer.alloc<int>(numberOfPixels);
		}
		else
		{
			data.maxPathLength = buffers.maxPathLengthBuffer.castedPointer<int>();
		}
		data.maxAllowedPathLength = maxBounce;

		prev_numberOfPixels = numberOfPixels;
		prev_maxBounce = maxBounce;
		prev_maxTrainingDataSize = maxTrainingDataSize;
		prev_type = type;
		prev_mixtureSize = mixtureSize;
		wasAllocated = true;
		return onDeviceData->networkInterfaceData.resourceBuffers.structBuffer.upload(data);
	}

	bool needNetworkInterfaceReallocation(const int& numberOfPixels, int maxBounce, int maxTrainingDataSize, const network::config::DistributionType& type, const int& mixtureSize)
	{
		const bool needReallocation = !wasAllocated || numberOfPixels != prev_numberOfPixels || maxBounce != prev_maxBounce || maxTrainingDataSize != prev_maxTrainingDataSize || type != prev_type || mixtureSize != prev_mixtureSize;
		return needReallocation;
	}

	void resetNetworkInterfaceAllocation()
	{
		if (auto& netInterface = onDeviceData->launchParamsData.getHostImage().networkInterface; netInterface != nullptr)
		{
			// Free memory if we are not using neural network
			onDeviceData->networkInterfaceData.freeResourceBuffer();
			onDeviceData->launchParamsData.editableHostImage().networkInterface = nullptr;
		}
		prev_numberOfPixels = 0;
		prev_maxBounce = 0;
		prev_maxTrainingDataSize = 0;
		prev_type = network::config::D_COUNT;
		prev_mixtureSize = 0;
		wasAllocated = false;
	}
}
