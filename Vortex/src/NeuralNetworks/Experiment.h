#pragma once
#include <map>
#include <memory>
#include <queue>
#include <unordered_set>

#include "Config/NetworkSettings.h"
#include "Core/Application.h"
#include "Core/Image.h"
#include "Core/Math.h"
#include "Device/UploadCode/CUDABuffer.h"
#include "Scene/Nodes/RendererSettings.h"
#include "Scene/Nodes/Statistics.h"
#include "Device/CudaFunctions/ErrorTypes.h"


namespace vtx
{
	struct Errors;

	namespace graph
	{
		class Renderer;
		struct Statistics;
	}

	namespace network
	{
		class Network;
	}
}


namespace vtx
{
	enum ExperimentStages
	{
		STAGE_NONE,
		STAGE_REFERENCE_GENERATION,
		STAGE_MAPE_COMPUTATION,

		STAGE_END
	};

	static std::map<ExperimentStages, std::string> experimentStageNames =
	{
		{STAGE_NONE, "Not performing Any Experiment"},
		{STAGE_REFERENCE_GENERATION, "Generating Ground truth"},
		{STAGE_MAPE_COMPUTATION, "Computing Mape"},
		{STAGE_END, "End"}
	};

	struct MinHeapComparator
	{
		bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const
		{
			return a.first > b.first; // For min heap
		}
	};

	struct Experiment
	{
		RendererSettings                 rendererSettings;
		network::config::NetworkSettings networkSettings;
		WavefrontSettings                wavefrontSettings;
		std::vector<float>               mape        = std::vector<float>(0);
		std::vector<float>               mse         = std::vector<float>(0);
		float*                           mseMap      = nullptr;
		float*                           mapeMap     = nullptr;
		std::string                      name        = "Unnamed";
		float                            averageMape = FLT_MAX;
		float                            averageMse  = FLT_MAX;
		graph::Statistics                statistics;
		bool                             generatedByBatchExperiments = false;
		bool                             completed                   = false;

		void        constructName(const int experimentNumber);
		std::string getStringHashKey();

		bool storeExperiment   = false;
		bool displayExperiment = true;
	};

	class ExperimentsManager
	{
	public:
		void loadGroundTruth(const std::string& filePath);

		void cleanExperiments();

		void saveGroundTruth(const std::string& filePath);

		static std::vector<Experiment>                 generateExperimentNeighbors(const Experiment& experiment);
		std::tuple<Experiment, Experiment, Experiment> startingConfigExperiments(
			const std::shared_ptr<graph::Renderer>& renderer);
		void setupNewExperiment(const Experiment& experiment, const std::shared_ptr<graph::Renderer>& renderer);
		void generateGroundTruth(const Experiment&                       gtExperiment, Application* app,
								 const std::shared_ptr<graph::Renderer>& renderer);
		std::tuple<bool, bool> performExperiment(Experiment&                             experiment, Application* app,
												 const std::shared_ptr<graph::Renderer>& renderer, const int      maxRuns);

		void refillExperimentQueue(bool isMis);
		void BatchExperimentRun();

		std::string getImageSavePath(std::string experimentName);

		GlFrameBuffer getGroundTruthGlBuffer();

	public:
		std::vector<Experiment> experiments;

		int width  = 800;
		int height = 800;

		bool         isGroundTruthReady = false;
		math::vec3f* groundTruth        = nullptr;
		CUDABuffer   groundTruthBuffer;

		CUDABuffer rendererImageBufferToneMapped;

		CUDABuffer errorMapsBuffer;

		int  gtSamples        = 5000;
		int  testSamples      = 400;
		bool stopAfterPlanned = false;
		bool doAblation       = false;
		// Manual Testing
		int              currentExperiment     = 0;
		int              currentExperimentStep = 0;
		ExperimentStages stage                 = STAGE_NONE;

		// Batch Testing
		std::string                     saveFilePath;
		std::unordered_set<std::string> experimentSet;
		std::deque<Experiment>          experimentQueue;

		// BSDF ONLY EXPERIMENTS
		std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, MinHeapComparator>
		bsdfExperimentMinHeap;
		float bsdfBestMapeScore = FLT_MAX;
		float bsdfBestMseScore  = FLT_MAX;
		int   bsdfBestExperimentIndex;

		// MIS EXPERIMENTS
		std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, MinHeapComparator>
		misExperimentMinHeap;
		float misBestMapeScore = FLT_MAX;
		float misBestMseScore  = FLT_MAX;
		int   misBestExperimentIndex;

		std::vector<float> groundTruthHost;

		InteropWrapper gtImageInterop;
	};
}
