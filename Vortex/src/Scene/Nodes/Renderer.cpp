#include "Renderer.h"
#include "Device/CUDAChecks.h"
#include "Core/Options.h"
#include <cudaGL.h>
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/UploadBuffers.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Scene/Traversal.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Core/Application.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Device/DevicePrograms/CudaKernels.h"
#include "Device/Wrappers/KernelTimings.h"
#include "Scene/Scene.h"
#include "Scene/Utility/Operations.h"
#include "Scene/Graph.h"
#include "Scene/HostVisitor.h"

namespace vtx::graph
{
	Renderer::Renderer() :
		Node(NT_RENDERER),
		width(getOptions()->width),
		height(getOptions()->height),
		threadData(&Renderer::render, this),
		settings(),
		waveFrontIntegrator(&settings)
	{
		camera    = ops::standardCamera();
		sceneRoot = ops::createNode<Group>();
		settings  = getOptions()->rendererSettings;

		//drawFrameBuffer.setSize(width, height);
		//drawFrameBuffer.bind();
		//glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT);
		//drawFrameBuffer.unbind();
		//
		//displayFrameBuffer.setSize(width, height);
		//displayFrameBuffer.bind();
		//glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		//glClear(GL_COLOR_BUFFER_BIT);
		//displayFrameBuffer.unbind();

		setWindow(Application::get()->glfwWindow);
	}


	std::vector<std::shared_ptr<Node>> Renderer::getChildren() const
	{
		std::vector<std::shared_ptr<Node>> children;
		camera ? children.push_back(camera) : void();
		sceneRoot ? children.push_back(sceneRoot) : void();
		environmentLight ? children.push_back(environmentLight) : void();
		return children;
	}

	void Renderer::accept(NodeVisitor& visitor)
	{
		visitor.visit(as<Renderer>());
	}

	vtxID Renderer::getInstanceIdOnClick(const int pixelID)
	{
		const gBufferHistory* gBufferPtr      = onDeviceData->frameBufferData.resourceBuffers.gBufferData.castedPointer<gBufferHistory>();
		const gBufferHistory* pixelGBufferPtr = gBufferPtr + pixelID;
		const auto*           pixelDataPtr    = reinterpret_cast<const vtxID*>(pixelGBufferPtr);

		vtxID hostValue;
		cudaMemcpy(&hostValue, pixelDataPtr, sizeof(vtxID), cudaMemcpyDeviceToHost);
		return graph::Scene::getSim()->UIDfromTID(NT_INSTANCE,hostValue);
	}

	bool Renderer::isReady(const bool setBusy)
	{
		std::unique_lock<std::mutex> lock(threadData.renderMutex, std::defer_lock);
		if (lock.try_lock())
		{
			return true;
			//if (!threadData.renderThreadBusy) {
			//	if (setBusy) {
			//		threadData.renderThreadBusy = true;
			//	}
			//	return true;
			//}
		}
		return false;
	}

	void Renderer::threadedRender()
	{
		if (isReady())
		{
			threadData.rendererBusy = true;
			threadData.renderCondition.notify_one();
		}
	}

	void Renderer::restart()
	{
		settings.iteration = 0;
		settings.isUpdated = true;
	}

	void Renderer::prepare()
	{
	}

	void Renderer::render()
	{
		onDeviceData->sync();
		onDeviceData->incrementFrameIteration();
		if (!settings.accumulate)
		{
			settings.iteration = 0;
		}
		if (settings.iteration == 0)
		{
			resetKernelStats();
			timer.reset();
			threadData.bufferCount = 0;
		}
		if (settings.iteration > settings.maxSamples)
		{
			return;
		}
		const LaunchParams& launchParams       = onDeviceData->launchParamsData.getHostImage();
		const LaunchParams* launchParamsDevice = onDeviceData->launchParamsData.getDeviceImage();
		const size_t        launchParamsSize   = onDeviceData->launchParamsData.imageBuffer.bytesSize();
		const int           fbWidth            = launchParams.frameBuffer.frameSize.x;
		const int           fbHeight           = launchParams.frameBuffer.frameSize.y;

		if (fbWidth == 0 || fbHeight == 0)
		{
			return;
		}

		{
			//timer.reset();
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_NOISE_COMPUTATION]);
			cudaEventRecord(events.first);

			if (
				settings.adaptiveSamplingSettings.active &&
				settings.adaptiveSamplingSettings.minAdaptiveSamples <= settings.iteration
				)
			{
				noiseComputation(launchParamsDevice, settings.adaptiveSamplingSettings.noiseKernelSize, settings.adaptiveSamplingSettings.albedoNormalNoiseInfluence);
			}
			cudaEventRecord(events.second);
		}

		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_TRACE]);
			cudaEventRecord(events.first);

			if (waveFrontIntegrator.settings.active && launchParams.queues.queueCounters != nullptr)
			{
				waveFrontIntegrator.render();
			}
			else
			{
				const optix::State&            state    = *(optix::getState());
				const OptixPipeline&           pipeline = optix::getRenderingPipeline()->getPipeline();
				const OptixShaderBindingTable& sbt      = optix::getRenderingPipeline()->getSbt();

				const auto result = optixLaunch( /*! pipeline we're launching launch: */
					pipeline, state.stream,
					/*! parameters and SBT */
					(CUdeviceptr)launchParamsDevice,
					launchParamsSize,
					&sbt,
					/*! dimensions of the launch: */
					launchParams.frameBuffer.frameSize.x,
					launchParams.frameBuffer.frameSize.y,
					1
				);
				OPTIX_CHECK(result);
			}
			cudaEventRecord(events.second);
		}

		{
			const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_POSTPROCESSING]);
			cudaEventRecord(events.first);

			toneMapRadianceKernel(launchParamsDevice, fbWidth, fbHeight, eventNames[R_TONE_MAP_RADIANCE]);
			math::vec3f* beauty                 = nullptr;
			const bool   isFireflyRemovalActive = settings.fireflySettings.active && settings.iteration > settings.fireflySettings.start;
			if (isFireflyRemovalActive)
			{
				removeFireflies(launchParamsDevice, settings.fireflySettings.kernelSize, settings.fireflySettings.threshold, fbWidth, fbHeight);
				beauty = onDeviceData->frameBufferData.resourceBuffers.fireflyRemoval.castedPointer<math::vec3f>();
			}

			if (settings.denoiserSettings.active && settings.iteration > settings.denoiserSettings.denoiserStart)
			{
				CUDABuffer& denoiserRadianceInput = isFireflyRemovalActive ? onDeviceData->frameBufferData.resourceBuffers.fireflyRemoval : onDeviceData->frameBufferData.resourceBuffers.hdriRadiance;
				CUDABuffer& albedoBuffer          = onDeviceData->frameBufferData.resourceBuffers.albedoNormalized;
				CUDABuffer& normalBuffer          = onDeviceData->frameBufferData.resourceBuffers.normalNormalized;
				optix::getState()->denoiser.setInputs(denoiserRadianceInput, albedoBuffer, normalBuffer);
				beauty = optix::getState()->denoiser.denoise(settings.denoiserSettings.denoiserBlend);
			}
			threadData.outputBufferBusy = true;
			switchOutput(launchParamsDevice, fbWidth, fbHeight, beauty);

			// Selection edge
			{
				const std::set<vtxID> uidSelectedInstances = Scene::get()->getSelectedInstancesIds();
				if (!uidSelectedInstances.empty())
				{
					auto*              gBuffer           = onDeviceData->frameBufferData.resourceBuffers.gBuffer.castedPointer<float>();
					auto*              outputBuffer      = onDeviceData->frameBufferData.resourceBuffers.cudaOutputBuffer.castedPointer<math::vec4f>();
					CUDABuffer         edgeMapBuffer     = onDeviceData->frameBufferData.resourceBuffers.edgeMapBuffer;
					CUDABuffer         selectedIdsBuffer = onDeviceData->frameBufferData.resourceBuffers.selectedIdsBuffer;
					std::vector<float> typeIds;
					typeIds.reserve(uidSelectedInstances.size());
					for (const vtxID uid : uidSelectedInstances)
					{
						typeIds.push_back((float)graph::Scene::getSim()->TIDfromUID(uid));
					}
					cuda::overlaySelectionEdge(gBuffer, outputBuffer, fbWidth, fbHeight, typeIds, 0.2f, 1.35f, &edgeMapBuffer, &selectedIdsBuffer);
				}
			}
			threadData.outputBufferBusy = false;
			threadData.bufferCount += 1;
			cudaEventRecord(events.second);
		}

		copyToGl();

		settings.iteration++;
		settings.isUpdated = true;
	}

	void Renderer::resize(const int _width, const int _height)
	{
		if (isSizeLocked)
		{
			return;
		}
		if (_width <= 0 || _height <= 0)
		{
			return;
		}
		if (width == _width && height == _height)
		{
			camera->resize(width, height);
			return;
		}
		width   = _width;
		height  = _height;
		resized = true;
		camera->resize(width, height);
	}

	void Renderer::copyToGl()
	{
		// Update the GL buffer here
		//CUDA_SYNC_CHECK();
		const std::pair<cudaEvent_t, cudaEvent_t> events = GetProfilerEvents(eventNames[R_DISPLAY]);
		cudaEventRecord(events.first);
		const LaunchParams& launchParams = onDeviceData->launchParamsData.getHostImage();

		const CUdeviceptr& buffer = launchParams.frameBuffer.outputBuffer;
		const int&         width  = launchParams.frameBuffer.frameSize.x;
		const int&         height = launchParams.frameBuffer.frameSize.y;
		const InteropUsage usage  = settings.runOnSeparateThread ? InteropUsage::MultiThreaded : InteropUsage::SingleThreaded;
		interopDraw.prepare(width, height, 4, usage);
		interopDraw.copyToGlBuffer(buffer, width, height);
		std::swap(interopDraw, interopDisplay);

		CUDA_SYNC_CHECK();
		cudaEventRecord(events.second);
	}

	void Renderer::setWindow(GLFWwindow* window)
	{
		// Create a shared OpenGL context for the separate thread
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		sharedContext = glfwCreateWindow(1, 1, "Shared Context", nullptr, window);
	}

	GlFrameBuffer Renderer::getFrame()
	{
		return interopDisplay.glFrameBuffer;
	}

	void Statistics::update(const std::shared_ptr<graph::Renderer>& renderNode)
	{
		const CudaEventTimes cuTimes        = getCudaEventTimes();
		const int            actualLaunches = getLaunches();
		totTimeSeconds                      = (cuTimes.trace + cuTimes.noiseComputation + cuTimes.postProcessing + cuTimes.display) / 1000.0f;
		samplesPerPixel                     = renderNode->settings.iteration;
		sppPerSecond                        = (float)(renderNode->width * renderNode->height * actualLaunches) / totTimeSeconds;
		frameTime                           = totTimeSeconds / (float)actualLaunches;
		fps                                 = 1.0f / frameTime;
		totTimeInternal                     = renderNode->timer.elapsedMillis() / 1000.0f;
		internalFps                         = (float)renderNode->settings.iteration / totTimeInternal;
		const float factor                  = 1.0f / (float)actualLaunches;
		rendererNoise                       = factor * cuTimes.noiseComputation;
		rendererTrace                       = factor * cuTimes.trace;
		rendererPost                        = factor * cuTimes.postProcessing;
		rendererDisplay                     = factor * cuTimes.display;
		waveFrontGenerateRay                = factor * cuTimes.genCameraRay;
		waveFrontTrace                      = factor * cuTimes.traceRadianceRay;
		waveFrontShade                      = factor * cuTimes.shadeRay;
		waveFrontShadow                     = factor * cuTimes.shadowRay;
		waveFrontEscaped                    = factor * cuTimes.handleEscapedRay;
		waveFrontAccumulate                 = factor * cuTimes.accumulateRay;
		waveFrontReset                      = factor * cuTimes.reset;
		waveFrontFetchQueueSize             = factor * cuTimes.fetchQueueSize;
		neuralShuffleDataset                = factor * (cuTimes.nnPrepareDataset + cuTimes.nnFillPath);
		neuralNetworkTrain                  = factor * cuTimes.nnTrain;
		neuralNetworkInfer                  = factor * cuTimes.nnInfer;
	}
}
