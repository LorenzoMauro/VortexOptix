#include "NeuralNetwork.h"

#include "Core/Options.h"
#include "Device/DevicePrograms/LaunchParams.h" 
#include "Device/UploadCode/UploadBuffers.h"

#include "Device/UploadCode/DeviceDataCoordinator.h"
//#include "Networks/Sac.h"
//#include "Networks/Npg.h"
#include "Device/Wrappers/KernelTimings.h"
#include "Networks/PGNet.h"


namespace vtx::network
{
	Network::Network()
    {
        settings = getOptions()->networkSettings;
        initNetworks();
    }

    Network::~Network() = default;

    void Network::initNetworks()
    {
        impl = std::make_unique<PGNet>(&settings);
        isInitialized = true;
    }

    void Network::train()
    {
        if (settings.active && settings.doTraining && onDeviceData->launchParamsData.getHostImage().networkInterface != nullptr)
        {
            const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_TRAIN]);
            cudaEventRecord(events.first);
            impl->train();
            CUDA_SYNC_CHECK();
            cudaEventRecord(events.second);
        }
    }

    bool Network::doInference()
    {
        return onDeviceData->launchParamsData.getHostImage().settings.renderer.iteration >= settings.inferenceIterationStart && settings.active && settings.doInference && onDeviceData->launchParamsData.getHostImage().networkInterface!=nullptr;
    }

	config::NetworkSettings& Network::getNeuralNetSettings()
    {
        return settings;
    }

    void Network::inference(const int& depth)
    {
        if(doInference())
        {
            const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[N_INFER]);
            cudaEventRecord(events.first);
            impl->inference(depth);
            CUDA_SYNC_CHECK();
            cudaEventRecord(events.second);
        }
    }

    GraphsData& Network::getGraphs()
    {
    	return impl->getGraphs();
	}
}

