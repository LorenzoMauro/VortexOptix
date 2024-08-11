#include "DebugDownloadData.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "NeuralNetworks/Distributions/Mixture.h"

namespace vtx
{
	static int prevMaxDepth = 0;
	static int prevDebugPixel = -1;
	static int prevMixtureSize = 0;
	static int prevDebugDepth = -1;
	static network::config::DistributionType prevType = network::config::DistributionType::D_COUNT;
	static bool wasAllocated = false;

	__host__ void vtx::DebugData::prepare(
		const int maxDepth,
		const int debugPixel,
		const int debugDepth,
		const int mixtureSize,
		const network::config::DistributionType& type
	)
	{
		if(
			!wasAllocated ||
			debugPixel != prevDebugPixel ||
			debugDepth != prevDebugDepth ||
			maxDepth != prevMaxDepth ||
			mixtureSize != prevMixtureSize ||
			type != prevType
		)
		{
			
			auto& buffers = onDeviceData->debugData.resourceBuffers;
			auto& debugData = onDeviceData->debugData.editableHostImage();

			debugData.debugPixel = debugPixel;
			debugData.debugDepth = debugDepth;

			if(
				maxDepth != prevMaxDepth ||
				mixtureSize != prevMixtureSize ||
				type != prevType)
			{
				// Allocate memory for bounce data
				debugData.maxDepth = maxDepth;
				std::vector<DebugBounceData> bounceDataVec(maxDepth);

				//Allocate memory for mixtures
				const int mixtureWeightsOffset = mixtureSize;
				const int mixtureParametersOffset = mixtureSize * distribution::Mixture::getDistributionParametersCount(type);
				auto* mixtureWeights = buffers.mixtureWeightsBuffers.alloc<float>(maxDepth * mixtureWeightsOffset);
				auto* mixtureParameters = buffers.mixtureParamsBuffers.alloc<float>(maxDepth * mixtureParametersOffset);
				for (int i = 0; i < maxDepth; i++)
				{
					bounceDataVec[i].mixtureWeigths = mixtureWeights + mixtureWeightsOffset * i;
					bounceDataVec[i].mixtureParameters = mixtureParameters + mixtureParametersOffset * i;
				}

				debugData.bounceData = buffers.bounceDataBuffers.upload(bounceDataVec);
			}
			else
			{
				debugData.bounceData = buffers.bounceDataBuffers.castedPointer<DebugBounceData>();
			}
			

			prevMaxDepth = maxDepth;
			prevDebugPixel = debugPixel;
			prevMixtureSize = mixtureSize;
			prevType = type;
			wasAllocated = true;
			prevDebugDepth = debugDepth;
		}
	}
	__host__ std::vector<DebugBounceData> DebugData::getFromDevice()
	{
		DebugData debugData;
		auto& oddDebugData = onDeviceData->debugData;

		oddDebugData.imageBuffer.download(&debugData);
		if (debugData.actualDepth == -1)
		{
			return std::vector<DebugBounceData>();
		}
		std::vector<DebugBounceData> bounceDataVec(debugData.actualDepth+1);
		cuMemcpy((CUdeviceptr)bounceDataVec.data(), oddDebugData.resourceBuffers.bounceDataBuffers.dPointer(), sizeof(DebugBounceData) * (debugData.actualDepth+1));
		//oddDebugData.resourceBuffers.bounceDataBuffers.download(bounceDataVec.data());

		return bounceDataVec;
	}
}

