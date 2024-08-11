#include "Options.h"

#include "Utils.h"
#include "Device/DevicePrograms/LaunchParams.h"
#include "NeuralNetworks/Experiment.h"

namespace vtx {

	static Options options;

	void startOptions()
	{

		options.width = 2100;
		options.height = 900;
		options.windowName = "Vortex";
		options.executablePath = utl::absolutePath(utl::getExecutablePath()) + '/';
		options.dataFolder = options.executablePath + "assets/";
		options.imGuiIniFile = options.dataFolder + "ImGui.ini";
		options.dllPath = options.executablePath;
		options.clearColor[4] = (0.45f, 0.55f, 0.60f, 1.00f);
#ifdef NDEBUG
		options.isDebug = false;
#else
		options.isDebug = true;
#endif
		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Rendering Settings /////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////

		options.rendererSettings.iteration = -1;
		options.rendererSettings.maxBounces = 10;
		options.rendererSettings.maxSamples = 100000;
		options.rendererSettings.accumulate = true;
		options.rendererSettings.samplingTechnique = S_MIS;
		options.rendererSettings.displayBuffer = FB_NETWORK_DEBUG_PATHS;
		options.rendererSettings.minClamp = 0.0001f;
		options.rendererSettings.maxClamp = 1000.0f;
		options.rendererSettings.useRussianRoulette = true;
		options.rendererSettings.runOnSeparateThread = false;
		options.rendererSettings.isUpdated = true;

		options.rendererSettings.adaptiveSamplingSettings.active = false;
		options.rendererSettings.adaptiveSamplingSettings.noiseKernelSize = 3;
		options.rendererSettings.adaptiveSamplingSettings.minAdaptiveSamples = 100;
		options.rendererSettings.adaptiveSamplingSettings.minPixelSamples = 1;
		options.rendererSettings.adaptiveSamplingSettings.maxPixelSamples = 1;
		options.rendererSettings.adaptiveSamplingSettings.albedoNormalNoiseInfluence = 1.0f;
		options.rendererSettings.adaptiveSamplingSettings.noiseCutOff = 0.1f;
		options.rendererSettings.adaptiveSamplingSettings.isUpdated = true;

		options.rendererSettings.fireflySettings.kernelSize = 3;
		options.rendererSettings.fireflySettings.threshold = 2.0f;
		options.rendererSettings.fireflySettings.active = false;
		options.rendererSettings.fireflySettings.start = 200;
		options.rendererSettings.fireflySettings.isUpdated = true;

		options.rendererSettings.denoiserSettings.active = false;
		options.rendererSettings.denoiserSettings.denoiserStart = -5;
		options.rendererSettings.denoiserSettings.denoiserBlend = 0.1f;
		options.rendererSettings.denoiserSettings.isUpdated = true;

		options.rendererSettings.toneMapperSettings.whitePoint = { 1.0f, 1.0f, 1.0f };
		options.rendererSettings.toneMapperSettings.invWhitePoint = { 1.0f, 1.0f, 1.0f };
		options.rendererSettings.toneMapperSettings.colorBalance = { 1.0f, 1.0f, 1.0f };
		options.rendererSettings.toneMapperSettings.burnHighlights = 0.0f;
		options.rendererSettings.toneMapperSettings.crushBlacks = 1.0f;
		options.rendererSettings.toneMapperSettings.saturation = 1.0f;
		options.rendererSettings.toneMapperSettings.gamma = 2.2f;
		options.rendererSettings.toneMapperSettings.invGamma = 1.0f / 2.2f;
		options.rendererSettings.toneMapperSettings.isUpdated = true;

		options.wavefrontSettings.active = true;
		options.wavefrontSettings.fitWavefront = false;
		options.wavefrontSettings.optixShade = false;
		options.wavefrontSettings.parallelShade = false;
		options.wavefrontSettings.longPathPercentage = 0.25f;
		options.wavefrontSettings.useLongPathKernel = false;
		options.wavefrontSettings.isUpdated  = true;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Neural Network Options /////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.networkSettings.active = true; // 10;//
		options.networkSettings.batchSize = 256000; // 256 * 50;// 100000; // 10;//
		options.networkSettings.doInference = true;
		options.networkSettings.doTraining = true;
		options.networkSettings.maxTrainingSteps = 1000;
		options.networkSettings.inferenceIterationStart = 1;
		options.networkSettings.clearOnInferenceStart = true;

		options.networkSettings.trainingBatchGenerationSettings.onlyNonZero = false;
		options.networkSettings.trainingBatchGenerationSettings.limitToFirstBounce = false;
		options.networkSettings.trainingBatchGenerationSettings.weightByMis = true;
		options.networkSettings.trainingBatchGenerationSettings.weightByPdf = true;
		options.networkSettings.trainingBatchGenerationSettings.useLightSample = true;
		options.networkSettings.trainingBatchGenerationSettings.trainOnLightSample = true;

		options.networkSettings.inputSettings.position.otype = network::config::EncodingType::Frequency;

		options.networkSettings.mainNetSettings.hiddenDim = 64;
		options.networkSettings.mainNetSettings.numHiddenLayers = 4;
		options.networkSettings.distributionType = network::config::D_NASG_TRIG;
		options.networkSettings.mixtureSize = 1;

		options.networkSettings.learningRate = 0.01f;
		options.networkSettings.blendFactor = 0.8f;
		options.networkSettings.constantBlendFactor = false;
		options.networkSettings.samplingFractionBlend = false;
		options.networkSettings.lossType = network::config::L_KL_DIV_MC_ESTIMATION;
		options.networkSettings.lossReduction = network::config::MEAN;

		options.networkSettings = ExperimentsManager::getBestGuess();
		options.networkSettings.isUpdated = true;
		options.networkSettings.active    = false;


		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Optix Options //////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.OptixVersion = OPTIX_VERSION;
		options.deviceID = 0;
		options.maxDcDepth = 2;
		options.maxTraversableGraphDepth = 2;
		options.LaunchParamName = "optixLaunchParams";
		options.enableCache = true;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// MDL Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.mdlSearchPath = { options.dataFolder, "./", options.dataFolder + "mdl/"};
		options.numTextureSpaces = 1; //should be set to 1 for performance reasons If you do not use the hair BSDF.
		options.numTextureResults = 16;
		options.enable_derivatives = false;
		options.mdlOptLevel = "2";
		options.mdlCallType = MDL_CUDA;

		////////////////////////////////////////////////////////////////////////////////////
		/////////////////// Gui Options ////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////
		options.nodeWidth = 300;
		options.fontPath = options.dataFolder + "fonts/LucidaGrande.ttf";
	}
	Options* getOptions()
	{
		if(!options.initialized)
		{
			startOptions();
			options.initialized = true;
		}
		return &options;
	}

}
