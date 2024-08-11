#include "NetSettingGenerators.h"
#include <algorithm>
#include <random>
#include <sstream>

#include "NetworkSettings.h"


static std::random_device rd;      // Obtain a random number from hardware
static std::mt19937       g(rd()); // Seed the generator

namespace vtx::network::config
{
	template <typename S, typename T>
	void mutate(S& settings, T& option, const std::vector<T>& options, std::vector<S>& variations)
	{
		T originValue = option;
		for (const auto& x : options)
		{
			if (originValue == x)
			{
				continue;
			}
			option = x;
			variations.push_back(settings);
		}
		option = originValue;
	}

	std::vector<EncodingConfig> mutateEncodingConfig(EncodingConfig& original)
	{
		std::vector<EncodingConfig> variations;
		mutate(original, original.otype, {EncodingType::Frequency, EncodingType::Grid, EncodingType::Identity, EncodingType::OneBlob, EncodingType::SphericalHarmonics, EncodingType::TriangleWave}, variations);
		switch (original.otype)
		{
		case EncodingType::Frequency:
			mutate(original, original.frequencyEncoding.n_frequencies, {6, 12, 24}, variations);
			break;
		case EncodingType::Grid:
			mutate(original, original.gridEncoding.type, {GridType::Hash, GridType::Tiled}, variations);
			mutate(original, original.gridEncoding.interpolation, {InterpolationType::Nearest, InterpolationType::Linear, InterpolationType::Smoothstep}, variations);
			mutate(original, original.gridEncoding.n_levels, {8, 16, 32}, variations);
			mutate(original, original.gridEncoding.n_features_per_level, {1, 2, 4, 8}, variations);
			mutate(original, original.gridEncoding.log2_hashmap_size, {14, 19, 24}, variations);
			mutate(original, original.gridEncoding.base_resolution, {8, 16, 32}, variations);
			mutate(original, original.gridEncoding.per_level_scale, {1.0, 2.0, 4.0}, variations);
			break;
		case EncodingType::Identity:
			break;
		case EncodingType::OneBlob:
			mutate(original, original.oneBlobEncoding.n_bins, {8, 16, 32}, variations);
			break;
		case EncodingType::SphericalHarmonics:
			mutate(original, original.sphericalHarmonicsEncoding.degree, {2, 4, 8}, variations);
			break;
		case EncodingType::TriangleWave:
			mutate(original, original.triangleWaveEncoding.n_frequencies, {6, 12, 24}, variations);
			break;
		default: ;
		}
		return variations;
	}

	std::vector<NetworkSettings> generateNetworkSettingNeighbors(NetworkSettings& original)
	{
		std::vector<NetworkSettings> variations;

		mutate(original, original.mainNetSettings.hiddenDim, {16, 32, 64, 128}, variations);
		mutate(original, original.mainNetSettings.numHiddenLayers, {2, 3, 4, 5, 6, 7, 8}, variations);

		// MAIN NETWORK SETTINGS
		mutate(original, original.inputSettings.normalizePosition, {true, false}, variations);
		mutate(original, original.inputSettings.position, mutateEncodingConfig(original.inputSettings.position), variations);
		mutate(original, original.inputSettings.wo, mutateEncodingConfig(original.inputSettings.wo), variations);
		mutate(original, original.inputSettings.normal, mutateEncodingConfig(original.inputSettings.normal), variations);

		// DISTRIBUTION SETTINGS
		mutate(original, original.distributionType, {D_SPHERICAL_GAUSSIAN, D_NASG_TRIG, D_NASG_ANGLE, D_NASG_AXIS_ANGLE}, variations);
		mutate(original, original.mixtureSize, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, variations);

		// BATCH GENERATION
		mutate(original, original.batchSize, {32000, 64000, 128000, 256000, 512000}, variations);
		mutate(original, original.trainingBatchGenerationSettings.limitToFirstBounce, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.onlyNonZero, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.weightByMis, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.weightByPdf, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.useLightSample, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.trainOnLightSample, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.skipSpecular, {true, false}, variations);
		mutate(original, original.trainingBatchGenerationSettings.isUpdated, {true, false}, variations);

		// LOSS SETTINGS
		mutate(original, original.lossType, {L_KL_DIV, L_KL_DIV_MC_ESTIMATION, L_KL_DIV_MC_ESTIMATION_NORMALIZED, L_MSE, L_PEARSON_DIV_MC_ESTIMATION}, variations);
		mutate(original, original.constantBlendFactor, {true, false}, variations);
		if (original.constantBlendFactor)
		{
			mutate(original, original.blendFactor, {1.0f, 0.99f, 0.9f, 0.8f, 0.5f}, variations);
		}
		mutate(original, original.samplingFractionBlend, {true, false}, variations);
		if (original.samplingFractionBlend)
		{
			mutate(original, original.fractionBlendTrainPercentage, {0.1f, 0.15f, 0.2f, 0.3f}, variations);
		}
		mutate(original, original.clampSamplingFraction, {true, false}, variations);
		if (original.clampSamplingFraction)
		{
			mutate(original, original.sfClampValue, {0.1f, 0.5f, 0.75f, 0.9f}, variations);
		}
		mutate(original, original.targetScale, {1.0f, 2.0f, 0.5f, 0.1f}, variations);
		mutate(original, original.scaleBySampleProb, {true, false}, variations);
		mutate(original, original.scaleLossBlendedQ, {true, false}, variations);
		mutate(original, original.clampBsdfProb, {true, false}, variations);
		mutate(original, original.learnInputRadiance, {true, false}, variations);
		mutate(original, original.lossClamp, {0.0f, 400.0f, 800.0f}, variations);
		mutate(original, original.toneMapRadiance, {true, false}, variations);

		// ENTROPY LOSS MUTATION
		mutate(original, original.useEntropyLoss, {true, false}, variations);
		if (original.useEntropyLoss)
		{
			mutate(original, original.entropyWeight, {0.001f, 0.01f, 0.1f, 1.0f, 10.0f}, variations);
		}

		// OPTIMIZER SETTINGS
		mutate(original, original.learningRate, {0.001f, 0.001f, 0.005f, 0.01f, 0.02f, 0.1f}, variations);
		mutate(original, original.adamEps, {5, 8, 10, 15, 20}, variations);
		mutate(original, original.schedulerGamma, {1.0f, 0.9f, 0.8f, 0.5f, 0.3f, 0.1f}, variations);
		if (original.schedulerGamma < 1.0f)
		{
			mutate(original, original.schedulerStep, {25, 50, 100, 200, 400, 600, 800, 1000, 1500}, variations);
		}
		mutate(original, original.emaUpdate, {true, false}, variations);
		if (original.emaUpdate)
		{
			mutate(original, original.emaDecay, {0.99f, 0.9f, 0.8f, 0.7f, 0.6f, 0.4f, 0.2f}, variations);
		}
		mutate(original, original.l2WeightExp, {0, 3, 6, 9}, variations);

		// AUXILIARY NETWORK MUTATION
		mutate(original, original.useAuxiliaryNetwork, {true, false}, variations);
		if (original.useAuxiliaryNetwork)
		{
			mutate(original, original.auxiliaryNetSettings.numHiddenLayers, {2, 3, 4, 5, 8}, variations);
			mutate(original, original.auxiliaryNetSettings.hiddenDim, {32, 64, 128}, variations);
			mutate(original, original.auxiliaryInputSettings.wi, mutateEncodingConfig(original.auxiliaryInputSettings.wi), variations);
			mutate(original, original.totAuxInputSize, {16, 32, 64}, variations);
			mutate(original, original.inRadianceLossFactor, {0.01f, 0.1f, 1.0f, 10.f}, variations);
			mutate(original, original.outRadianceLossFactor, {0.01f, 0.1f, 1.0f, 10.f}, variations);
			mutate(original, original.throughputLossFactor, {0.01f, 0.1f, 1.0f, 10.f}, variations);
			mutate(original, original.auxiliaryWeight, {0.001f, 0.01f, 0.1f, 1.0f}, variations);
			mutate(original, original.radianceTargetScaleFactor, {1.0f, 10.f}, variations);
		}

		// ADDITIONAL INPUTS
		mutate(original, original.useInstanceId, {true, false}, variations);
		if (original.useInstanceId)
		{
			mutate(original, original.instanceIdEncodingConfig, mutateEncodingConfig(original.instanceIdEncodingConfig), variations);
		}
		mutate(original, original.useMaterialId, {true, false}, variations);
		if (original.useMaterialId)
		{
			mutate(original, original.materialIdEncodingConfig, mutateEncodingConfig(original.materialIdEncodingConfig), variations);
		}
		mutate(original, original.useTriangleId, {true, false}, variations);
		if (original.useTriangleId)
		{
			mutate(original, original.triangleIdEncodingConfig, mutateEncodingConfig(original.triangleIdEncodingConfig), variations);
		}

		std::shuffle(variations.begin(), variations.end(), g);
		return variations;
	}


	std::vector<NetworkSettings> ablationVariations(NetworkSettings& original)
	{
		std::vector<NetworkSettings> variations;

		bool all = false;
		mutate(original, original.distributionType, {D_NASG_TRIG_NORMALIZED}, variations);
		if (all)
		{
			mutate(original, original.distributionType, {D_NASG_TRIG_NORMALIZED, D_NASG_AXIS_ANGLE, D_SPHERICAL_GAUSSIAN, D_NASG_TRIG, D_NASG_ANGLE}, variations);
			mutate(original, original.mainNetSettings.hiddenDim, {32, 64, 128}, variations);
			mutate(original, original.mainNetSettings.numHiddenLayers, {2, 4, 6}, variations);

			// MAIN NETWORK SETTINGS
			// mutate(original, original.inputSettings.normalizePosition, { true, false }, variations);
			mutate(original, original.inputSettings.position, mutateEncodingConfig(original.inputSettings.position), variations);
			mutate(original, original.inputSettings.wo, mutateEncodingConfig(original.inputSettings.wo), variations);
			mutate(original, original.inputSettings.normal, mutateEncodingConfig(original.inputSettings.normal), variations);

			// DISTRIBUTION SETTINGS
			mutate(original, original.mixtureSize, {1, 2, 4, 6, 8, 10}, variations);

			// BATCH GENERATION
			mutate(original, original.batchSize, {32000, 64000, 128000, 256000, 512000}, variations);
			// mutate(original, original.trainingBatchGenerationSettings.limitToFirstBounce, { true, false }, variations);
			mutate(original, original.trainingBatchGenerationSettings.onlyNonZero, {true, false}, variations);
			mutate(original, original.trainingBatchGenerationSettings.weightByMis, {true, false}, variations);
			// mutate(original, original.trainingBatchGenerationSettings.weightByPdf, { true, false }, variations);
			mutate(original, original.trainingBatchGenerationSettings.useLightSample, {true, false}, variations);
			mutate(original, original.trainingBatchGenerationSettings.trainOnLightSample, {true, false}, variations);
			mutate(original, original.trainingBatchGenerationSettings.skipSpecular, {true, false}, variations);
			// mutate(original, original.trainingBatchGenerationSettings.isUpdated, { true, false }, variations);

			// LOSS SETTINGS
			// mutate(original, original.lossType, {L_KL_DIV, L_KL_DIV_MC_ESTIMATION, L_KL_DIV_MC_ESTIMATION_NORMALIZED, L_MSE, L_PEARSON_DIV_MC_ESTIMATION}, variations);
			// mutate(original, original.constantBlendFactor, {true, false}, variations);
			// if (original.constantBlendFactor)
			// {
			// 	mutate(original, original.blendFactor, {1.0f, 0.99f, 0.9f, 0.8f, 0.5f}, variations);
			// }
			// mutate(original, original.samplingFractionBlend, {true, false}, variations);
			// if (original.samplingFractionBlend)
			// {
			// 	mutate(original, original.fractionBlendTrainPercentage, {0.1f, 0.15f, 0.2f, 0.3f}, variations);
			// }
			// mutate(original, original.clampSamplingFraction, {true, false}, variations);
			// if (original.clampSamplingFraction)
			// {
			// 	mutate(original, original.sfClampValue, {0.1f, 0.5f, 0.75f, 0.9f}, variations);
			// }
			// mutate(original, original.targetScale, {1.0f, 2.0f, 0.5f, 0.1f}, variations);
			// mutate(original, original.scaleBySampleProb, {true, false}, variations);
			// mutate(original, original.scaleLossBlendedQ, {true, false}, variations);
			// mutate(original, original.clampBsdfProb, {true, false}, variations);
			// mutate(original, original.learnInputRadiance, {true, false}, variations);
			// mutate(original, original.lossClamp, {0.0f, 400.0f, 800.0f}, variations);
			mutate(original, original.toneMapRadiance, {true, false}, variations);

			// ENTROPY LOSS MUTATION
			mutate(original, original.useEntropyLoss, {true, false}, variations);
			const auto tmp          = original.useEntropyLoss;
			original.useEntropyLoss = true;
			mutate(original, original.entropyWeight, {0.001f, 0.01f, 0.1f, 1.0f}, variations);
			original.useEntropyLoss = tmp;

			// OPTIMIZER SETTINGS
			mutate(original, original.learningRate, {0.001f, 0.005f, 0.01f, 0.05f, 0.1f, 0.5f}, variations);
			mutate(original, original.adamEps, {5, 8, 10, 15, 20}, variations);
			mutate(original, original.schedulerGamma, {1.0f, 0.9f, 0.8f, 0.5f, 0.3f, 0.1f}, variations);
			if (original.schedulerGamma < 1.0f)
			{
				mutate(original, original.schedulerStep, {50, 100, 200, 400, 800}, variations);
			}
			mutate(original, original.emaUpdate, {true, false}, variations);
			const auto tmp2    = original.emaUpdate;
			original.emaUpdate = true;
			mutate(original, original.emaDecay, {0.99f, 0.9f, 0.8f, 0.7f, 0.6f, 0.4f, 0.2f}, variations);
			original.emaUpdate = tmp2;
			mutate(original, original.l2WeightExp, {0, 3, 6, 9}, variations);

			// AUXILIARY NETWORK MUTATION
			//mutate(original, original.useAuxiliaryNetwork, {true, false}, variations);
			//if (original.useAuxiliaryNetwork)
			//{
			//	mutate(original, original.auxiliaryNetSettings.numHiddenLayers, {2, 3, 4, 5, 8}, variations);
			//	mutate(original, original.auxiliaryNetSettings.hiddenDim, {32, 64, 128}, variations);
			//	mutate(original, original.auxiliaryInputSettings.wi, mutateEncodingConfig(original.auxiliaryInputSettings.wi), variations);
			//	mutate(original, original.totAuxInputSize, {16, 32, 64}, variations);
			//	mutate(original, original.inRadianceLossFactor, {0.01f, 0.1f, 1.0f, 10.f}, variations);
			//	mutate(original, original.outRadianceLossFactor, {0.01f, 0.1f, 1.0f, 10.f}, variations);
			//	mutate(original, original.throughputLossFactor, {0.01f, 0.1f, 1.0f, 10.f}, variations);
			//	mutate(original, original.auxiliaryWeight, {0.001f, 0.01f, 0.1f, 1.0f}, variations);
			//	mutate(original, original.radianceTargetScaleFactor, {1.0f, 10.f}, variations);
			//}

			// ADDITIONAL INPUTS
			mutate(original, original.useInstanceId, {true, false}, variations);
			//if (original.useInstanceId)
			//{
			//	mutate(original, original.instanceIdEncodingConfig, mutateEncodingConfig(original.instanceIdEncodingConfig), variations);
			//}
			mutate(original, original.useMaterialId, {true, false}, variations);
			//if (original.useMaterialId)
			//{
			//	mutate(original, original.materialIdEncodingConfig, mutateEncodingConfig(original.materialIdEncodingConfig), variations);
			//}
			//mutate(original, original.useTriangleId, {true, false}, variations);
			//if (original.useTriangleId)
			//{
			//	mutate(original, original.triangleIdEncodingConfig, mutateEncodingConfig(original.triangleIdEncodingConfig), variations);
			//}

			//std::shuffle(variations.begin(), variations.end(), g);
		}

		return variations;
	}

	NetworkSettings getBestGuess()
	{
		NetworkSettings bestGuess = getSOTA();

		bestGuess.active                  = true;
		bestGuess.doTraining              = true;
		bestGuess.doInference             = true;
		bestGuess.plotGraphs              = true;
		bestGuess.isUpdated               = true;
		bestGuess.maxTrainingSteps        = 1000;
		bestGuess.inferenceIterationStart = 10;
		bestGuess.clearOnInferenceStart   = false;


		bestGuess.mainNetSettings.hiddenDim       = 64;
		bestGuess.mainNetSettings.numHiddenLayers = 2; // <- 3;// 5;

		{
			bestGuess.inputSettings.normalizePosition = true;
			bestGuess.inputSettings.position.otype    = EncodingType::Grid;

			bestGuess.inputSettings.position.oneBlobEncoding.n_bins = 16;

			bestGuess.inputSettings.position.gridEncoding.type                 = GridType::Hash;
			bestGuess.inputSettings.position.gridEncoding.n_levels             = 8;
			bestGuess.inputSettings.position.gridEncoding.n_features_per_level = 4;
			bestGuess.inputSettings.position.gridEncoding.log2_hashmap_size    = 19;
			bestGuess.inputSettings.position.gridEncoding.base_resolution      = 8;
			bestGuess.inputSettings.position.gridEncoding.per_level_scale      = 2.0f;
			bestGuess.inputSettings.position.gridEncoding.interpolation        = InterpolationType::Linear;
		}
		{
			bestGuess.inputSettings.normal.otype = EncodingType::SphericalHarmonics;
		}
		{
			bestGuess.inputSettings.wo.otype = EncodingType::SphericalHarmonics;
		}


		bestGuess.distributionType = D_NASG_AXIS_ANGLE;
		bestGuess.mixtureSize      = 8; // 7;// 5;// 3;


		bestGuess.useTriangleId = false;
		bestGuess.useInstanceId = false;
		bestGuess.useMaterialId = false;

		//AUXILIARY NETWORK SETTINGS

		bestGuess.useAuxiliaryNetwork = false;

		// TRAINING SETTINGS
		bestGuess.batchSize                                          = 512000; // 262144;
		bestGuess.trainingBatchGenerationSettings.onlyNonZero        = false;
		bestGuess.trainingBatchGenerationSettings.weightByMis        = true;
		bestGuess.trainingBatchGenerationSettings.weightByPdf        = true;
		bestGuess.trainingBatchGenerationSettings.useLightSample     = true;
		bestGuess.trainingBatchGenerationSettings.trainOnLightSample = false;
		bestGuess.trainingBatchGenerationSettings.limitToFirstBounce = false;
		bestGuess.trainingBatchGenerationSettings.skipSpecular       = true;
		bestGuess.trainingBatchGenerationSettings.skipIncomplete     = false;
		bestGuess.lossClamp                                          = 0.0f;
		bestGuess.toneMapRadiance                                    = false;
		bestGuess.learnInputRadiance                                 = false;
		bestGuess.clampBsdfProb                                      = false;
		bestGuess.scaleLossBlendedQ                                  = false;
		bestGuess.blendFactor                                        = 1.0f;
		bestGuess.constantBlendFactor                                = false;
		bestGuess.samplingFractionBlend                              = false;
		bestGuess.fractionBlendTrainPercentage                       = 0.1f;
		bestGuess.constantSamplingFraction                           = false;
		bestGuess.clampSamplingFraction                              = true;
		bestGuess.sfClampValue                                       = 0.95f;
		bestGuess.lossType                                           = L_KL_DIV_MC_ESTIMATION;
		bestGuess.lossReduction                                      = MEAN;

		bestGuess.useEntropyLoss = true;
		bestGuess.entropyWeight  = 0.01f;
		bestGuess.targetEntropy  = 3.0f;

		bestGuess.learningRate   = 0.02f; //<- 0.01f;// 0.005f;
		bestGuess.adamEps        = 15;
		bestGuess.schedulerGamma = 0.5f; // 0.9f;
		bestGuess.schedulerStep  = 150;  // <- 150;
		bestGuess.emaUpdate      = true;
		bestGuess.emaDecay       = 0.3f;
		bestGuess.l2WeightExp    = 0;


		return bestGuess;
	}

	NetworkSettings getWhishfull()
	{
		NetworkSettings whishfull = getBestGuess();

		whishfull.useTriangleId = false;
		whishfull.useInstanceId = false;
		whishfull.useMaterialId = true;

		// TRAINING SETTINGS
		whishfull.trainingBatchGenerationSettings.onlyNonZero = true;

		whishfull.useEntropyLoss = true;
		whishfull.entropyWeight  = 0.01f;
		whishfull.targetEntropy  = 3.0f;

		//AUXILIARY NETWORK SETTINGS

		// TRAINING SETTINGS
		return whishfull;
	}

	NetworkSettings getSOTA()
	{
		NetworkSettings settings;
		settings.active                  = true;
		settings.doTraining              = true;
		settings.doInference             = true;
		settings.plotGraphs              = true;
		settings.isUpdated               = true;
		settings.maxTrainingSteps        = 1000;
		settings.inferenceIterationStart = 1;
		settings.clearOnInferenceStart   = false;

		settings.useTriangleId = false;
		settings.useInstanceId = false;
		settings.useMaterialId = false;

		settings.mainNetSettings.hiddenDim       = 64;
		settings.mainNetSettings.numHiddenLayers = 3;

		{
			settings.inputSettings.normalizePosition                          = true;
			settings.inputSettings.position.otype                             = EncodingType::Grid;
			settings.inputSettings.position.gridEncoding.n_levels             = 8;
			settings.inputSettings.position.gridEncoding.n_features_per_level = 4;
			settings.inputSettings.position.gridEncoding.log2_hashmap_size    = 19;
			settings.inputSettings.position.gridEncoding.base_resolution      = 8;
			settings.inputSettings.position.gridEncoding.per_level_scale      = 2.0f;
			settings.inputSettings.position.gridEncoding.interpolation        = InterpolationType::Linear;
		}
		{
			settings.inputSettings.normal.otype = EncodingType::SphericalHarmonics;
		}
		{
			settings.inputSettings.wo.otype = EncodingType::SphericalHarmonics;
		}


		settings.distributionType = D_SPHERICAL_GAUSSIAN;
		settings.mixtureSize      = 8; // 5;// 3;

		//AUXILIARY NETWORK SETTINGS

		settings.useAuxiliaryNetwork = false;

		// TRAINING SETTINGS
		settings.learnInputRadiance                                 = false;
		settings.lossClamp                                          = 0.0f;
		settings.toneMapRadiance                                    = false;
		settings.learningRate                                       = 0.004f;
		settings.batchSize                                          = 262144;
		settings.trainingBatchGenerationSettings.onlyNonZero        = false;
		settings.trainingBatchGenerationSettings.weightByPdf        = true;
		settings.trainingBatchGenerationSettings.weightByMis        = true;
		settings.trainingBatchGenerationSettings.limitToFirstBounce = false;
		settings.trainingBatchGenerationSettings.useLightSample     = true;
		settings.trainingBatchGenerationSettings.trainOnLightSample = false;
		settings.trainingBatchGenerationSettings.skipSpecular       = true;
		settings.clampBsdfProb                                      = false;
		settings.scaleLossBlendedQ                                  = false;
		settings.blendFactor                                        = 1.0f;
		settings.constantBlendFactor                                = false;
		settings.samplingFractionBlend                              = false;
		settings.constantSamplingFraction                           = false;
		settings.constantSamplingFractionValue                      = 0.5f;
		settings.fractionBlendTrainPercentage                       = 0.1f;
		settings.lossType                                           = L_KL_DIV_MC_ESTIMATION;
		settings.lossReduction                                      = MEAN;

		settings.useEntropyLoss = false;

		settings.adamEps        = 15;
		settings.schedulerGamma = 1.0f;
		settings.emaUpdate      = true;
		settings.emaDecay       = 0.9f;
		settings.l2WeightExp    = 0;

		return settings;
	}


	NetworkSettings getNasgSOTA()
	{
		NetworkSettings settings;
		settings.active                  = true;
		settings.doTraining              = true;
		settings.doInference             = true;
		settings.plotGraphs              = true;
		settings.isUpdated               = true;
		settings.maxTrainingSteps        = 1000;
		settings.inferenceIterationStart = 1;
		settings.clearOnInferenceStart   = false;

		settings.useTriangleId = false;
		settings.useInstanceId = false;
		settings.useMaterialId = false;

		settings.mainNetSettings.hiddenDim       = 128;
		settings.mainNetSettings.numHiddenLayers = 4;

		{
			settings.inputSettings.normalizePosition = true;
			settings.inputSettings.position.otype    = EncodingType::OneBlob;
		}
		{
			settings.inputSettings.normal.otype = EncodingType::Identity;
		}
		{
			settings.inputSettings.wo.otype = EncodingType::Identity;
		}


		settings.distributionType = D_NASG_TRIG;
		settings.mixtureSize      = 4; // 5;// 3;

		//AUXILIARY NETWORK SETTINGS

		settings.useAuxiliaryNetwork = false;

		// TRAINING SETTINGS
		settings.learnInputRadiance                                 = false;
		settings.lossClamp                                          = 0.0f;
		settings.toneMapRadiance                                    = false;
		settings.learningRate                                       = 0.004f;
		settings.batchSize                                          = 262144;
		settings.trainingBatchGenerationSettings.onlyNonZero        = false;
		settings.trainingBatchGenerationSettings.weightByPdf        = true;
		settings.trainingBatchGenerationSettings.weightByMis        = true;
		settings.trainingBatchGenerationSettings.limitToFirstBounce = false;
		settings.trainingBatchGenerationSettings.useLightSample     = true;
		settings.trainingBatchGenerationSettings.trainOnLightSample = false;
		settings.trainingBatchGenerationSettings.skipSpecular       = true;
		settings.clampBsdfProb                                      = false;
		settings.scaleLossBlendedQ                                  = false;
		settings.blendFactor                                        = 0.8f;
		settings.constantBlendFactor                                = true;
		settings.samplingFractionBlend                              = true;
		settings.fractionBlendTrainPercentage                       = 1.0f;
		settings.constantSamplingFraction                           = false;
		settings.lossType                                           = L_KL_DIV_MC_ESTIMATION;
		settings.lossReduction                                      = MEAN;

		settings.useEntropyLoss = false;

		settings.adamEps        = 8;
		settings.schedulerGamma = 1.0f;
		settings.emaUpdate      = false;
		settings.emaDecay       = 0.9f;
		settings.l2WeightExp    = 0;

		return settings;
	}

	std::string toStringHash(const BatchGenerationConfig& config)
	{
		std::stringstream hash;
		hash << stringHash(config.limitToFirstBounce);
		hash << stringHash(config.onlyNonZero);
		hash << stringHash(config.weightByMis);
		hash << stringHash(config.weightByPdf);
		hash << stringHash(config.useLightSample);
		hash << stringHash(config.trainOnLightSample);
		hash << stringHash(config.skipSpecular);
		return hash.str();
	}

	std::string toStringHash(const FrequencyEncoding& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.n_frequencies);
		return hash.str();
	};

	std::string toStringHash(const GridEncoding& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash((int)config.type);
		hash << stringHash(config.n_levels);
		hash << stringHash(config.n_features_per_level);
		hash << stringHash(config.log2_hashmap_size);
		hash << stringHash(config.base_resolution);
		hash << stringHash(config.per_level_scale);
		hash << stringHash((int)config.interpolation);
		return hash.str();
	};

	std::string toStringHash(const IdentityEncoding& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.scale);
		hash << stringHash(config.offset);
		return hash.str();
	};

	std::string toStringHash(const OneBlobEncoding& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.n_bins);
		return hash.str();
	};

	std::string toStringHash(const SphericalHarmonicsEncoding& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.degree);
		return hash.str();
	};

	std::string toStringHash(const TriangleWaveEncoding& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		hash << stringHash(config.n_frequencies);
		return hash.str();
	};

	std::string toStringHash(const EncodingConfig& config)
	{
		std::stringstream hash;
		hash << stringHash((int)config.otype);
		switch (config.otype)
		{
		case EncodingType::Frequency:
			hash << toStringHash(config.frequencyEncoding);
			break;
		case EncodingType::Grid:
			hash << toStringHash(config.gridEncoding);
			break;
		case EncodingType::Identity:
			hash << toStringHash(config.identityEncoding);
			break;
		case EncodingType::OneBlob:
			hash << toStringHash(config.oneBlobEncoding);
			break;
		case EncodingType::SphericalHarmonics:
			hash << toStringHash(config.sphericalHarmonicsEncoding);
			break;
		case EncodingType::TriangleWave:
			hash << toStringHash(config.triangleWaveEncoding);
			break;
		default: ;
		}
		return hash.str();
	};

	std::string toStringHash(const MainNetEncodingConfig& config)
	{
		std::stringstream hash;
		hash << stringHash(config.normalizePosition);
		hash << toStringHash(config.position);
		hash << toStringHash(config.wo);
		hash << toStringHash(config.normal);
		return hash.str();
	};

	std::string toStringHash(const MlpSettings& config)
	{
		std::stringstream hash;
		hash << stringHash(config.inputDim);
		hash << stringHash(config.outputDim);
		hash << stringHash(config.hiddenDim);
		hash << stringHash(config.numHiddenLayers);
		hash << stringHash(config.activationType);
		return hash.str();
	};

	std::string toStringHash(const AuxNetEncodingConfig& config)
	{
		std::stringstream hash;
		hash << toStringHash(config.wi);
		return hash.str();
	};

	std::string getNetworkSettingHash(const NetworkSettings& settings)
	{
		std::stringstream hash;

		// GENERAL SETTINGS
		hash << stringHash(settings.doTraining);
		hash << stringHash(settings.doInference);
		hash << stringHash(settings.maxTrainingSteps);
		hash << stringHash(settings.inferenceIterationStart);

		//MAIN NETWORK SETTINGS
		hash << toStringHash(settings.mainNetSettings);
		hash << toStringHash(settings.inputSettings);

		// DISTRIBUTION SETTINGS
		hash << stringHash(settings.distributionType);
		hash << stringHash(settings.mixtureSize);

		// BATCH GENERATION
		hash << stringHash(settings.batchSize);
		hash << toStringHash(settings.trainingBatchGenerationSettings);

		//LOSS SETTINGS
		hash << stringHash(settings.lossType);
		hash << stringHash(settings.lossReduction);
		hash << stringHash(settings.constantBlendFactor);
		if (settings.constantBlendFactor)
		{
			hash << stringHash(settings.blendFactor);
		}
		hash << stringHash(settings.samplingFractionBlend);
		if (settings.samplingFractionBlend)
		{
			hash << stringHash(settings.fractionBlendTrainPercentage);
		}
		hash << stringHash(settings.clampSamplingFraction);
		if (settings.clampSamplingFraction)
		{
			hash << stringHash(settings.sfClampValue);
		}
		hash << stringHash(settings.targetScale);
		hash << stringHash(settings.scaleBySampleProb);
		hash << stringHash(settings.scaleLossBlendedQ);
		hash << stringHash(settings.clampBsdfProb);
		hash << stringHash(settings.learnInputRadiance);
		hash << stringHash(settings.lossClamp);
		hash << stringHash(settings.toneMapRadiance);

		// ENTROPY LOSS SETTINGS
		hash << stringHash(settings.useEntropyLoss);
		if (settings.useEntropyLoss)
		{
			hash << stringHash(settings.entropyWeight);
			hash << stringHash(settings.targetEntropy);
		}
		// OPTIMIZER SETTINGS
		hash << stringHash(settings.learningRate);
		hash << stringHash(settings.adamEps);
		hash << stringHash(settings.schedulerGamma);
		if (settings.schedulerGamma < 1.0f)
		{
			hash << stringHash(settings.schedulerStep);
		}
		hash << stringHash(settings.emaUpdate);
		if (settings.emaUpdate)
		{
			hash << stringHash(settings.emaDecay);
		}
		hash << stringHash(settings.l2WeightExp);

		// AUXILIARY LOSS SETTINGS
		hash << stringHash(settings.useAuxiliaryNetwork);
		if (settings.useAuxiliaryNetwork)
		{
			hash << toStringHash(settings.auxiliaryNetSettings);
			hash << toStringHash(settings.auxiliaryInputSettings);
			hash << stringHash(settings.totAuxInputSize);
			hash << stringHash(settings.inRadianceLossFactor);
			hash << stringHash(settings.outRadianceLossFactor);
			hash << stringHash(settings.throughputLossFactor);
			hash << stringHash(settings.auxiliaryWeight);
			hash << stringHash(settings.radianceTargetScaleFactor);
			hash << stringHash(settings.throughputTargetScaleFactor);
		}

		// ADDITIONAL INPUTS
		hash << stringHash(settings.useMaterialId);
		if (settings.useMaterialId)
		{
			hash << toStringHash(settings.materialIdEncodingConfig);
		}
		hash << stringHash(settings.useTriangleId);
		if (settings.useTriangleId)
		{
			hash << toStringHash(settings.triangleIdEncodingConfig);
		}
		hash << stringHash(settings.useInstanceId);
		if (settings.useInstanceId)
		{
			hash << toStringHash(settings.instanceIdEncodingConfig);
		}

		return hash.str();
	}
}
