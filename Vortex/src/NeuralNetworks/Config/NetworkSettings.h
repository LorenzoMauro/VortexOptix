#pragma once
#ifndef NETWORK_SETTINGS_H
#define NETWORK_SETTINGS_H
#include <map>
#include <string>

#include "DistributionConfig.h"
#include "EncodingConfig.h"
#include "LossConfig.h"
#include "TrainingBatchConfig.h"

namespace vtx::network::config
{
	enum ActivationType
	{
		AT_RELU,
		AT_TANH,
		AT_SIGMOID,
		AT_SOFTMAX,
		AT_NONE
	};

	struct MlpSettings
	{
		int				inputDim;
		int				outputDim = 64;
		int				hiddenDim = 64;
		int				numHiddenLayers = 3;
		ActivationType	activationType = AT_RELU;
	};


	struct MainNetEncodingConfig
	{
		bool                    normalizePosition = false;
		EncodingConfig position = {};
		EncodingConfig wo = {};
		EncodingConfig normal = {};
	};

	struct AuxNetEncodingConfig
	{
		EncodingConfig wi = {};
	};

	struct NetworkSettings
	{
		// GENERAL SETTINGS
		bool active       = true;
		bool wasActive    = active;
		bool doTraining   = true;
		bool doInference  = true;
		bool plotGraphs   = true;
		bool isUpdated    = true;

		int   maxTrainingSteps        = 1000;
		int   inferenceIterationStart = 1;
		bool  clearOnInferenceStart   = false;


		//MAIN NETWORK SETTINGS
		MlpSettings mainNetSettings;
		MainNetEncodingConfig inputSettings;
		bool            emaUpdate;
		float           emaDecay = 0.9f;

		// DISTRIBUTION SETTINGS
		DistributionType distributionType = D_SPHERICAL_GAUSSIAN;
		int              mixtureSize      = 1;

		// BATCH GENERATION
		int   batchSize = 1;
		BatchGenerationConfig trainingBatchGenerationSettings;

		//LOSS SETTINGS
		LossType      lossType              = L_KL_DIV_MC_ESTIMATION;
		LossReduction lossReduction         = MEAN;

		bool          constantBlendFactor   = false;
		float         blendFactor           = 0.9f;

		bool          samplingFractionBlend = false;
		float		  fractionBlendTrainPercentage = 0.2f;

		bool          clampSamplingFraction = false;
		float         sfClampValue = 1.0f;

		float            constantSamplingFraction = false;
		float            constantSamplingFractionValue = 0.5f;



		float         targetScale			= 1.0f;
		bool		  scaleBySampleProb    = false;

		bool  scaleLossBlendedQ = false;
		bool  clampBsdfProb = false;

		bool  learnInputRadiance = false;
		float lossClamp = 100.0f;
		bool   toneMapRadiance = false;

		// ENTROPY LOSS SETTINGS
		bool          useEntropyLoss = false;
		float         entropyWeight = 1.0f;
		float         targetEntropy = 3.0f;

		// OPTIMIZER SETTINGS
		float learningRate = 0.001f;
		int	  adamEps = 15;
		float schedulerGamma = 0.33f;
		int schedulerStep = 200;
		int l2WeightExp = 0;

		// AUXILIARY LOSS SETTINGS
		bool                 useAuxiliaryNetwork = false;
		MlpSettings          auxiliaryNetSettings;
		AuxNetEncodingConfig auxiliaryInputSettings;
		int                  totAuxInputSize             = 64;
		float                inRadianceLossFactor        = 1.0f;
		float                outRadianceLossFactor       = 1.0f;
		float                throughputLossFactor        = 1.0f;
		float                auxiliaryWeight             = 1.0f;
		float                radianceTargetScaleFactor   = 1.0f;
		float                throughputTargetScaleFactor = 1.0f;

		// ADDITIONAL INPUTS
		bool           useMaterialId            = false;
		EncodingConfig materialIdEncodingConfig = {};
		bool           useTriangleId            = false;
		EncodingConfig triangleIdEncodingConfig = {};
		bool           useInstanceId            = false;
		EncodingConfig instanceIdEncodingConfig = {};


		void resetUpdate()
		{
			isUpdated                                 = false;
			trainingBatchGenerationSettings.isUpdated = false;
		}

		bool isAnyUpdated()
		{
			return isUpdated || trainingBatchGenerationSettings.isUpdated;
		}
	};

}


#endif