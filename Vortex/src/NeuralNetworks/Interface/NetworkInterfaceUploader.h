#pragma once
#include <vector>

#include "NetworkInterfaceStructs.h"
#include "NeuralNetworks/Config/DistributionConfig.h"

namespace vtx
{
	struct NetworkInterface;
	struct InferenceData;
	struct TrainingData;

	TrainingData* uploadTrainingData(int maxTrainingData);

	TrainingData* getPreviousTrainingData();

	InferenceData* uploadInferenceData(const int& numberOfPixels, const network::config::DistributionType& type, const int& mixtureSize);

	InferenceData* getPreviousInferenceData();

	NetworkInterface* uploadNetworkInterface(const int& numberOfPixels, int maxBounce, int maxTrainingDataSize,const network::config::DistributionType& type,const int& mixtureSize);

	bool needNetworkInterfaceReallocation(const int& numberOfPixels, int maxBounce, int maxTrainingDataSize, const network::config::DistributionType& type, const int& mixtureSize);

	void resetNetworkInterfaceAllocation();
}
