#include "Experiment.h"

#include <queue>
#include <unordered_set>

#include "Config/NetSettingGenerators.h"
#include "Core/Application.h"
#include "Device/OptixWrapper.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Gui/Windows/ExperimentsWindow.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Device/DevicePrograms/CudaKernels.h"
#include "Serialization/Serializer.h"


namespace vtx
{
	void Experiment::constructName(const int experimentNumber)
	{
		if (name == "Unnamed")
		{
			name = "Experiment_" + std::to_string(experimentNumber);
		}
		else
		{
			std::vector<std::string> splitName         = utl::splitString(name, "_");
			const std::string&       seedExperiment    = splitName[0];
			const std::string&       samplingTechnique = rendererSettings.samplingTechnique == S_MIS ? "MIS" : "BSDF";
			if (splitName.size() > 1)
			{
				name = seedExperiment + "_" + samplingTechnique + "_" + splitName.back() + "_" + std::to_string(experimentNumber);
			}
			else
			{
				name = seedExperiment + "_" + samplingTechnique + "_" + std::to_string(experimentNumber);
			}
		}
	}

	std::string Experiment::getStringHashKey()
	{
		std::stringstream hash;
		hash << stringHash(rendererSettings.samplingTechnique);
		if (networkSettings.active)
		{
			hash << network::config::getNetworkSettingHash(networkSettings);
		}
		return hash.str();
	}

	void ExperimentsManager::loadGroundTruth(const std::string& filePath)
	{
		Image image;
		image.load(filePath);
		float* hostImage = image.getData();
		width            = image.getWidth();
		height           = image.getHeight();

		groundTruthBuffer.resize(image.getWidth() * image.getHeight() * image.getChannels() * sizeof(float));
		groundTruthBuffer.upload(hostImage, image.getWidth() * image.getHeight() * image.getChannels());
		groundTruth = groundTruthBuffer.castedPointer<math::vec3f>();

		isGroundTruthReady = true;
	}

	void ExperimentsManager::cleanExperiments()
	{
		for (auto& exp : experiments)
		{
			experimentSet.erase(exp.getStringHashKey());
		}
		bsdfBestMapeScore     = FLT_MAX;
		misBestMapeScore      = FLT_MAX;
		bsdfExperimentMinHeap = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, MinHeapComparator>();
		misExperimentMinHeap  = std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, MinHeapComparator>();
		experimentSet.clear();
		experiments.clear();
		currentExperiment = 0;
	}

	void ExperimentsManager::saveGroundTruth(const std::string& filePath)
	{
		std::vector<math::vec3f> hostImage(width * height);
		math::vec3f*             hostImagePtr = hostImage.data();
		groundTruthBuffer.download(hostImagePtr);
		Image image;
		image.load((float*)hostImagePtr, width, height, 3);
		image.save(filePath);
	}

	std::vector<Experiment> ExperimentsManager::generateExperimentNeighbors(const Experiment& experiment)
	{
		std::vector<Experiment>                             newExperiments;
		network::config::NetworkSettings                    networkSettings = experiment.networkSettings;
		const std::vector<network::config::NetworkSettings> variations      = network::config::generateNetworkSettingNeighbors(networkSettings);
		for (const auto& var : variations)
		{
			Experiment newExperiment                         = experiment;
			newExperiment.networkSettings                    = var;
			newExperiment.rendererSettings.samplingTechnique = S_BSDF;
			newExperiments.push_back(newExperiment);
			newExperiment.rendererSettings.samplingTechnique = S_MIS;
			newExperiments.push_back(newExperiment);
		}
		return newExperiments;
	}

	std::tuple<Experiment, Experiment, Experiment> ExperimentsManager::startingConfigExperiments(const std::shared_ptr<graph::Renderer>& renderer)
	{
		Experiment groundTruthExp;
		groundTruthExp.rendererSettings                                = renderer->settings;
		groundTruthExp.rendererSettings.samplingTechnique              = S_MIS;
		groundTruthExp.rendererSettings.maxSamples                     = gtSamples;
		groundTruthExp.rendererSettings.denoiserSettings.active        = true;
		groundTruthExp.rendererSettings.denoiserSettings.denoiserBlend = 0.0f;
		groundTruthExp.rendererSettings.useRussianRoulette             = false;

		groundTruthExp.wavefrontSettings                   = renderer->waveFrontIntegrator.settings;
		groundTruthExp.wavefrontSettings.active            = true;
		groundTruthExp.wavefrontSettings.fitWavefront      = false;
		groundTruthExp.wavefrontSettings.optixShade        = false;
		groundTruthExp.wavefrontSettings.parallelShade     = false;
		groundTruthExp.wavefrontSettings.useLongPathKernel = false;

		groundTruthExp.networkSettings        = network::config::getBestGuess();
		groundTruthExp.networkSettings.active = false;

		Experiment bsdfExperiment                                      = groundTruthExp;
		bsdfExperiment.rendererSettings.samplingTechnique              = S_BSDF;
		bsdfExperiment.rendererSettings.maxSamples                     = testSamples;
		bsdfExperiment.rendererSettings.denoiserSettings.active        = false;
		groundTruthExp.rendererSettings.denoiserSettings.denoiserBlend = 0.0f;
		bsdfExperiment.rendererSettings.fireflySettings.active         = false;

		Experiment misExperiment                                       = groundTruthExp;
		misExperiment.rendererSettings.maxSamples                      = testSamples;
		misExperiment.rendererSettings.denoiserSettings.active         = false;
		groundTruthExp.rendererSettings.denoiserSettings.denoiserBlend = 0.0f;
		misExperiment.rendererSettings.fireflySettings.active          = false;

		return {groundTruthExp, misExperiment, bsdfExperiment};
	}

	void ExperimentsManager::setupNewExperiment(const Experiment& experiment, const std::shared_ptr<graph::Renderer>& renderer)
	{
		const auto tmpDisplayBuffer                                     = renderer->settings.displayBuffer;
		renderer->settings                                              = experiment.rendererSettings;
		renderer->settings.displayBuffer                                = tmpDisplayBuffer;
		renderer->settings.isUpdated                                    = true;
		renderer->waveFrontIntegrator.settings                          = experiment.wavefrontSettings;
		renderer->waveFrontIntegrator.settings.isUpdated                = true;
		renderer->waveFrontIntegrator.network.settings                  = experiment.networkSettings;
		renderer->waveFrontIntegrator.network.settings.isUpdated        = true;
		renderer->waveFrontIntegrator.network.settings.maxTrainingSteps = testSamples * 0.5f;
		renderer->waveFrontIntegrator.network.reset();
		renderer->restart();
	}

	void ExperimentsManager::generateGroundTruth(const Experiment& gtExperiment, Application* app, const std::shared_ptr<graph::Renderer>& renderer)
	{
		VTX_WARN("GROUND TRUTH GENERATION");
		setupNewExperiment(gtExperiment, renderer);
		for (int i = 0; i < renderer->settings.maxSamples; i++)
		{
			app->batchExperimentAppLoopBody(i, renderer);
		}
		{
			CUDABuffer&  gtBuffer  = optix::getState()->denoiser.output;
			const size_t imageSize = (size_t)width * (size_t)height * sizeof(math::vec3f);
			groundTruthBuffer.resize(imageSize);
			toneMapBuffer(gtBuffer, groundTruthBuffer, width, height, onDeviceData->launchParamsData.getHostImage().settings.renderer.toneMapperSettings);
			groundTruth     = groundTruthBuffer.castedPointer<math::vec3f>();
			groundTruthHost = std::vector<float>((size_t)width * (size_t)height * 3);
			groundTruthBuffer.download(groundTruthHost.data());
			isGroundTruthReady    = true;
			currentExperimentStep = 0;
			Image(groundTruthBuffer, width, height, 3).save(getImageSavePath("groundTruth.png"));
		}
	}

	std::tuple<bool, bool> ExperimentsManager::performExperiment(Experiment& experiment, Application* app, const std::shared_ptr<graph::Renderer>& renderer, const int maxRuns)
	{
		experiment.constructName(experiments.size());
		experiment.generatedByBatchExperiments = true;
		experiment.displayExperiment           = true;

		bool success = false;

		const std::string&              name     = experiment.name;
		int                             runCount = 0;
		std::vector<std::vector<float>> mapeRuns(experiment.rendererSettings.maxSamples);
		std::vector<std::vector<float>> mseRuns(experiment.rendererSettings.maxSamples);

		int tryCount = 0;
		while (tryCount < maxRuns)
		{
			try
			{
				VTX_INFO("EXPERIMENT {} Run {}", experiment.name, runCount);
				experiment.mape.clear();
				experiment.mse.clear();
				setupNewExperiment(experiment, renderer);

				for (int i = 0; i < renderer->settings.maxSamples; i++)
				{
					app->batchExperimentAppLoopBody(i, renderer);
					Errors errors;
					if (experiment.rendererSettings.denoiserSettings.active)
					{
						toneMapBuffer(optix::getState()->denoiser.output, rendererImageBufferToneMapped, width, height, onDeviceData->launchParamsData.getHostImage().settings.renderer.toneMapperSettings);
						errors = cuda::computeErrors(groundTruthBuffer, rendererImageBufferToneMapped, errorMapsBuffer, width, height);
					}
					else
					{
						errors = cuda::computeErrors(groundTruthBuffer, onDeviceData->frameBufferData.resourceBuffers.tmRadiance, errorMapsBuffer, width, height);
					}
					experiment.mape.push_back(errors.mape);
					experiment.mse.push_back(errors.mse);
					experiment.mapeMap = errors.dMapeMap;
					experiment.mseMap  = errors.dMseMap;
					mapeRuns[i].push_back(errors.mape);
					mseRuns[i].push_back(errors.mse);
					if (glfwWindowShouldClose(app->glfwWindow))
						return {false, false};
				}
				runCount++;
			}
			catch (std::exception& e)
			{
				VTX_ERROR("Standard exception in experiment {}: {}", name, e.what());
			}
			catch (...)
			{
				VTX_ERROR("Unknown exception in experiment {}", name);
			}

			// Save images regardless of success
			{
				std::string runCountStr = "_RUN_" + std::to_string(runCount);
				if (renderer->settings.denoiserSettings.active)
				{
					Image(rendererImageBufferToneMapped, width, height, 3).save(getImageSavePath(experiment.name + runCountStr));
				}
				else
				{
					Image(onDeviceData->frameBufferData.resourceBuffers.tmRadiance, width, height, 3).save(getImageSavePath(experiment.name + runCountStr));
				}

				PreAllocatedCudaBuffer pre(width * height * sizeof(float), (void*)experiment.mseMap);
				CUDABuffer             displayBuffer;
				vtx::cuda::copyRtoRGBA(*(CUDABuffer*)(&pre), displayBuffer, width, height);
				Image(displayBuffer, width, height, 4).save(utl::splitString(getImageSavePath(experiment.name + "_MSE_MAP" + runCountStr), ".")[0] + ".hdr");

				pre = PreAllocatedCudaBuffer(width * height * sizeof(float), (void*)experiment.mapeMap);
				vtx::cuda::copyRtoRGBA(*(reinterpret_cast<CUDABuffer*>(&pre)), displayBuffer, width, height);
				Image(displayBuffer, width, height, 4).save(utl::splitString(getImageSavePath(experiment.name + "_MAPE_MAP" + runCountStr), ".")[0] + ".hdr");
				displayBuffer.free();
			}
			tryCount++;
		}

		success        = runCount > 0;
		bool breakLoop = false;
		bool isMis     = experiment.rendererSettings.samplingTechnique == S_MIS;
		if (success)
		{
			experiment.mape.clear();
			experiment.mse.clear();

			// Average the results
			{
				VTX_INFO("Averaging results");
				float avgMape  = 0.f;
				float avgMse   = 0.f;
				int   avgCount = 0;
				for (int i = 0; i < mapeRuns.size(); i++)
				{
					float sumMape = 0.f;
					float sumMse  = 0.f;
					int   count   = 0.f;
					for (int j = 0; j < mapeRuns[i].size(); j++)
					{
						sumMape += mapeRuns[i][j];
						sumMse += mseRuns[i][j];
						count++;
						avgMape += mapeRuns[i][j];
						avgMse += mseRuns[i][j];
						avgCount++;
					}
					if (count == 0)
					{
						experiment.completed         = false;
						experiment.displayExperiment = false;
						VTX_WARN("Experiment {} failed", experiment.name);
						return {false, false};
					}
					experiment.mape.push_back(sumMape / (float)count);
					experiment.mse.push_back(sumMse / (float)count);
				}
				avgMape /= (float)avgCount;
				avgMse /= (float)avgCount;
				VTX_INFO("Averaged MAPE: {}", avgMape);
				VTX_INFO("Averaged MSE: {}", avgMse);
				if (isnan(avgMape) || isnan(avgMse) || isinf(avgMape) || isinf(avgMse))
				{
					experiment.completed         = false;
					experiment.displayExperiment = false;
					VTX_WARN("Experiment {} failed", experiment.name);
					return {false, false};
				}
				experiment.averageMape = avgMape;
				experiment.averageMse  = avgMse;
			}


			experiment.statistics = renderer->statistics;
			experimentSet.insert(experiment.getStringHashKey());

			if (experiment.networkSettings.active)
			{
				const float mapeScore           = experiment.mape.back();
				const float mseScore            = experiment.mse.back();
				auto&       experimentMinHeap   = isMis ? misExperimentMinHeap : bsdfExperimentMinHeap;
				float&      bestMapeScore       = isMis ? misBestMapeScore : bsdfBestMapeScore;
				float&      bestMseScore        = isMis ? misBestMseScore : bsdfBestMseScore;
				int&        bestExperimentIndex = isMis ? misBestExperimentIndex : bsdfBestExperimentIndex;

				experimentMinHeap.push({mapeScore, experiments.size() - 1});
				experiment.completed  = true;
				const bool isBestMape = mapeScore < bestMapeScore;
				const bool isBestMse  = mseScore < bestMseScore;
				if (isBestMse)
				{
					experiment.displayExperiment = true;
					bestMseScore                 = mseScore;
					bestExperimentIndex          = experiments.size() - 1;
					breakLoop                    = true;
				}
				if (isBestMape)
				{
					experiment.displayExperiment = true;
					bestMapeScore                = mapeScore;
				}
				if (!isBestMape && !isBestMse)
				{
					experiment.displayExperiment = false;
				}

				VTX_INFO("\t Experiment {} finished with MAPE {} vs {} and MSE {} vs {}", experiment.name, mapeScore, bestMapeScore, mseScore, bestMseScore);
				if (isBestMse)
				{
					VTX_INFO("\t\tNew best MSE! Breaking and Searching from this!");
				}
				if (isBestMape)
				{
					VTX_INFO("\t\tNew best MAPE!");
				}
			}
			else
			{
				// It's a gt experiment
				experiment.displayExperiment = true;
			}
		}
		else
		{
			VTX_ERROR("Experiment {} failed", experiment.name);
			experiment.completed         = false;
			experiment.displayExperiment = false;
		}
		vtx::serializer::serializeBatchExperiments(saveFilePath);
		vtx::serializer::serializeBatchExperiments(utl::splitString(saveFilePath, ".")[0] += ".xml");
		return {breakLoop, isMis};
	}

	void ExperimentsManager::refillExperimentQueue(bool isMis)
	{
		bool firstRun = true;
		while ((firstRun || experimentQueue.empty()) && (!misExperimentMinHeap.empty() || !bsdfExperimentMinHeap.empty()))
		{
			if (isMis)
			{
				VTX_INFO("Attempting Refilling experiment queue from Mis heap");
			}
			else
			{
				VTX_INFO("Attempting Refilling experiment queue from BSDF heap");
			}
			firstRun                             = false;
			auto&     firstTryExperimentMinHeap  = isMis ? misExperimentMinHeap : bsdfExperimentMinHeap;
			auto&     secondTryExperimentMinHeap = isMis ? bsdfExperimentMinHeap : misExperimentMinHeap;
			const int bestIndex                  = firstTryExperimentMinHeap.empty() ? secondTryExperimentMinHeap.empty() ? -1 : secondTryExperimentMinHeap.top().second : firstTryExperimentMinHeap.top().second;
			if (bestIndex == -1)
			{
				VTX_WARN("No more experiments to run");
				break;
			}
			Experiment best       = experiments[bestIndex];
			auto       variations = generateExperimentNeighbors(best);
			VTX_INFO("Adding {} variations to the queue front", variations.size());
			for (auto& neighbor : variations)
			{
				std::string hashKey = neighbor.getStringHashKey();
				if (experimentSet.count(hashKey) != 0)
				{
					VTX_WARN("SKIPPING BECAUSE ALREADY IN SET");
					continue;
				}
				experimentQueue.push_front(neighbor);
			}
		}
	}


	void addVariations(Experiment exp, const network::config::NetworkSettings& netSett, const std::string& name, std::deque<Experiment>& queue)
	{
		bool doIncoming = true;

		exp.networkSettings                     = netSett;
		exp.rendererSettings.useRussianRoulette = false;

		queue.push_back(exp);
		Experiment* current                         = &queue.back();
		current->rendererSettings.samplingTechnique = S_MIS;
		current->name                               = name + "-outgoing-";

		if (doIncoming)
		{
			queue.push_back(exp);
			current                                     = &queue.back();
			current->rendererSettings.samplingTechnique = S_MIS;
			current->name                               = name + "-incoming-";
			current->networkSettings.learnInputRadiance = true;
		}


		queue.push_back(exp);
		current                                     = &queue.back();
		current->rendererSettings.samplingTechnique = S_BSDF;
		current->name                               = name + "-outgoing-";
		//if(current->networkSettings.distributionType==network::config::D_NASG_AXIS_ANGLE)
		//{
		//	current->networkSettings.trainingBatchGenerationSettings.onlyNonZero = false;
		//	current->networkSettings.trainingBatchGenerationSettings.skipIncomplete = true;
		//}

		if (doIncoming)
		{
			queue.push_back(exp);
			current                                     = &queue.back();
			current->rendererSettings.samplingTechnique = S_BSDF;
			current->name                               = name + "-incoming-";
			current->networkSettings.learnInputRadiance = true;
			//if (current->networkSettings.distributionType == network::config::D_NASG_AXIS_ANGLE)
			//{
			//	current->networkSettings.trainingBatchGenerationSettings.onlyNonZero = false;
			//	current->networkSettings.trainingBatchGenerationSettings.skipIncomplete = true;
			//}
		}
	}

	void ExperimentsManager::BatchExperimentRun()
	{
		VTX_INFO("BATCH EXPERIMENT RUN");
		Application*                             app      = Application::get();
		const graph::Scene*                      scene    = graph::Scene::get();
		const std::shared_ptr<graph::Renderer>&  renderer = scene->renderer;
		const std::shared_ptr<ExperimentsWindow> ew       = app->windowManager->getWindow<ExperimentsWindow>();

		renderer->isSizeLocked = false;
		renderer->resize(width, height);
		renderer->isSizeLocked = true;

		renderer->camera->lockCamera = false;
		renderer->camera->resize(width, height);
		renderer->camera->lockCamera = true;

		rendererImageBufferToneMapped.resize((size_t)width * (size_t)height * sizeof(math::vec3f));

		// for each Scene
		{
			auto [gtExperiment, misExperiment, bsdfExperiment] = startingConfigExperiments(renderer);
			if (!(isGroundTruthReady && groundTruthBuffer.bytesSize() != 0))
			{
				generateGroundTruth(gtExperiment, app, renderer);
			}
			else
			{
				groundTruth = groundTruthBuffer.castedPointer<math::vec3f>();
			}
			// we start by performing the mis and bsdf without network experiments


			misExperiment.name = "GT";
			experimentQueue.push_back(misExperiment);

			bsdfExperiment.name = "GT";
			experimentQueue.push_back(bsdfExperiment);

			int tryCount = 3;
			if (doAblation)
			{
				tryCount                                                 = 3;
				auto                                          bestGuess  = network::config::getBestGuess();
				std::vector<network::config::NetworkSettings> variations = network::config::ablationVariations(bestGuess);
				variations.push_back(bestGuess);

				Experiment baseExp                          = misExperiment;
				baseExp.name                                = "ablation";
				baseExp.rendererSettings.useRussianRoulette = false;

				std::set<std::string> localExperimentSet;
				for (int i = variations.size() - 1; i >= 0; i--)
				{
					Experiment localExp = baseExp;
					if (i == variations.size() - 1)
					{
						localExp.name = "REFERENCE";
					}
					localExp.networkSettings                    = variations[i];
					localExp.rendererSettings.samplingTechnique = S_MIS;
					std::string hashKey                         = localExp.getStringHashKey();
					if (localExperimentSet.count(hashKey) != 0)
					{
						VTX_WARN("SKIPPING BECAUSE ALREADY IN SET");
						continue;
					}
					experimentQueue.push_back(localExp);
					localExperimentSet.insert(hashKey);

					localExp.rendererSettings.samplingTechnique = S_BSDF;
					hashKey                                     = localExp.getStringHashKey();
					if (localExperimentSet.count(hashKey) != 0)
					{
						VTX_WARN("SKIPPING BECAUSE ALREADY IN SET");
						continue;
					}
					experimentQueue.push_back(localExp);
					localExperimentSet.insert(hashKey);
				}

				VTX_INFO("ABALATION EXPERIMENTS: {} from {} variations", experimentQueue.size(), 2*(variations.size()+1));
			}
			else
			{
				addVariations(misExperiment, network::config::getBestGuess(), "BestGuess", experimentQueue);
				addVariations(misExperiment, network::config::getSOTA(), "SOTA-NPM", experimentQueue);
				addVariations(misExperiment, network::config::getNasgSOTA(), "SOTA-NASG", experimentQueue);
			}


			const int queueSize = experimentQueue.size();
			for (int i = 0; i < queueSize; i++)
			{
				Experiment experiment = experimentQueue.front();
				experimentQueue.pop_front();
				if (std::string hashKey = experiment.getStringHashKey(); experimentSet.count(hashKey) != 0)
				{
					continue;
				}
				experiments.push_back(experiment);
				performExperiment(experiments.back(), app, renderer, tryCount);
				experiments.back().displayExperiment = true;
			}

			if (stopAfterPlanned)
			{
				// exit with success
				for (auto& exp : experiments)
				{
					VTX_INFO("Experiment: {}", exp.name);
					VTX_INFO("\tMSE: {}", exp.averageMse);
					VTX_INFO("\tMAPE: {}", exp.averageMape);
				}
				glfwSetWindowShouldClose(app->glfwWindow, GLFW_TRUE);
				return;
			}
			// at this point the queue are empty we need to refill them
			// we refill them based on the best bsdf only for now hence the false
			// by doing this I think I ensure I do all the pre thought tests and then start to
			// explore from the best mis experiment
			refillExperimentQueue(true);

			while (!experimentQueue.empty() && !glfwWindowShouldClose(app->glfwWindow))
			{
				const int queueSize = experimentQueue.size();
				for (int i = 0; i < queueSize; i++)
				{
					experiments.push_back(experimentQueue.front());
					experimentQueue.pop_front();
					auto [breakLoop, isBestMis] = performExperiment(experiments.back(), app, renderer, tryCount);
					if (breakLoop)
					{
						refillExperimentQueue(isBestMis);
						break;
					}
				}
			}
		}
		renderer->isSizeLocked       = false;
		renderer->camera->lockCamera = false;
	}

	std::string ExperimentsManager::getImageSavePath(std::string experimentName)
	{
		std::string savePath = utl::getFolder(saveFilePath) + "/Images/Experiment_" + experimentName + ".png";
		return utl::absolutePath(savePath);
	}

	GlFrameBuffer ExperimentsManager::getGroundTruthGlBuffer()
	{
		if (!isGroundTruthReady || groundTruth == nullptr || groundTruthBuffer.bytesSize() == 0)
		{
			VTX_WARN("Ground truth not ready");
			return GlFrameBuffer();
		}
		if (
			gtImageInterop.cuArray == nullptr ||
			gtImageInterop.cuGraphicResource == nullptr ||
			width != (int)gtImageInterop.glFrameBuffer.width ||
			height != (int)gtImageInterop.glFrameBuffer.height
		)
		{
			CUDABuffer rgbaGT;

			gtImageInterop.prepare(width, height, 3, InteropUsage::SingleThreaded);
			auto cuImage = onDeviceData->frameBufferData.resourceBuffers.tmRadiance.dPointer();
			gtImageInterop.copyToGlBuffer(cuImage, width, height);
		}

		return gtImageInterop.glFrameBuffer;
	}
}
