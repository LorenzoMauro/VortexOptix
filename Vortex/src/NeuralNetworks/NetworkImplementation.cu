#include "NetworkImplementation.h"
#include "NeuralNetwork.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Device/Wrappers/KernelLaunch.h"
#include "Device/Wrappers/KernelTimings.h"
#include "Interface/NetworkInterface.h"

namespace vtx::network
{
	void Network::reset()
	{
		settings.doTraining = true;
		impl->reset();

		const LaunchParams* deviceParams = onDeviceData->launchParamsData.getDeviceImage();
		const LaunchParams& hostParams = onDeviceData->launchParamsData.getHostImage();
		if(hostParams.networkInterface == nullptr)
		{
			return;
		}
		const math::vec2ui screenSize = hostParams.frameBuffer.frameSize;
		const int               nPixels = screenSize.x * screenSize.y;
		gpuParallelFor(eventNames[N_FILL_PATH],
			nPixels,
			[deviceParams] __device__(const int id)
		{
			deviceParams->networkInterface->reset(id);
			deviceParams->networkInterface->samples->resetPixel(id);
			if (id == 0)
			{
				deviceParams->networkInterface->trainingData->reset();
			}
		});
	}

	void NetworkImplementation::prepareDataset()
	{
		const LaunchParams* deviceParams = onDeviceData->launchParamsData.getDeviceImage();
		const LaunchParams& hostParams = onDeviceData->launchParamsData.getHostImage();
		const math::vec2ui screenSize = hostParams.frameBuffer.frameSize;
		const int maxBounces = hostParams.settings.renderer.maxBounces;
		const int               nPixels = screenSize.x * screenSize.y;

		gpuParallelFor(eventNames[N_FILL_PATH],
			nPixels,
			[deviceParams] __device__(const int id)
		{
			deviceParams->networkInterface->finalizePath(id, deviceParams->settings.neural, true);
			if (id == 0)
			{
				deviceParams->networkInterface->trainingData->reset();
			}
		});

		const int               maxDatasetSize = maxBounces * nPixels * 2;
		bool doToneMap = settings->toneMapRadiance;
		gpuParallelFor(eventNames[N_PREPARE_DATASET],
			maxDatasetSize,
			[deviceParams, doToneMap] __device__(const int id)
		{
			const Samples* samples = deviceParams->networkInterface->samples;
			TrainingData* trainingData = deviceParams->networkInterface->trainingData;

			trainingData->buildTrainingData(id, samples, doToneMap);
		});

	}
}

