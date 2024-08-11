#include "ImageWindowPopUp.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "NeuralNetworks/NeuralNetwork.h"
#include "Gui/GuiProvider.h"
#include "NeuralNetworks/Config/NetworkSettings.h"
#include "NeuralNetworks/Distributions/Mixture.h"
#include "NeuralNetworks/Interface/NetworkInterfaceStructs.h"
#include "NeuralNetworks/Interface/NetworkInterfaceUploader.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"

namespace vtx::gui
{
	using namespace network;
	using namespace config;

	bool GuiProvider::drawEditGui(FrequencyEncoding& config)
	{
		bool isUpdated = vtxImGui::halfSpaceWidget("Frequency Count", ImGui::DragInt, (hiddenLabel + "Frequency Count").c_str(), &config.n_frequencies, 1.0f, 1, 50, "%d", 0);

		return isUpdated;
		
	}

	bool GuiProvider::drawEditGui(GridEncoding& config)
	{
		bool isUpdated = false;
		isUpdated = vtxImGui::halfSpaceCombo("Grid Type", config.type, GridTypeName, (int)GridType::GridTypeCount);
		isUpdated |= vtxImGui::halfSpaceDragInt("Level Count", &config.n_levels, 1.0f, 1, 50);
		isUpdated |= vtxImGui::halfSpaceDragInt("Features Per Level", &config.n_features_per_level, 1.0f, 1, 50, "%d", 0);
		isUpdated |= vtxImGui::halfSpaceDragInt("Log2 Hashmap Size", &config.log2_hashmap_size, 1.0f, 1, 50, "%d", 0);
		isUpdated |= vtxImGui::halfSpaceDragInt("Base resolution",  &config.base_resolution, 1.0f, 1, 50, "%d", 0);
		isUpdated |= vtxImGui::halfSpaceDragFloat("Level Scale",  &config.per_level_scale, 0.1f, 1.0, 10.0, "%.10f", 0);
		isUpdated |= vtxImGui::halfSpaceCombo("Interpolation", config.interpolation, InterpolationTypeName, (int)InterpolationType::InterpolationTypeCount);
		return isUpdated;

	}

	bool GuiProvider::drawEditGui(IdentityEncoding& config)
	{
		bool isUpdated = false;

		isUpdated |= vtxImGui::halfSpaceDragFloat("Scale", &config.scale, 0.1f, 0.0f, 10.0f, "%.10f", 0);
		isUpdated |= vtxImGui::halfSpaceDragFloat("Offset", &config.offset, 0.1f, 0.0f, 10.0f, "%.10f", 0);

		return isUpdated;

	}

	bool GuiProvider::drawEditGui(OneBlobEncoding& config)
	{
		bool isUpdated = false;
		isUpdated |= vtxImGui::halfSpaceDragInt("Bins", &config.n_bins, 1.0f, 1, 50, "%d", 0);
		return isUpdated;

	}

	bool GuiProvider::drawEditGui(SphericalHarmonicsEncoding& config)
	{
		bool isUpdated = false;
		isUpdated |= vtxImGui::halfSpaceDragInt("Degree", &config.degree, 1.0f, 1, 50, "%d", 0);
		return isUpdated;

	}

	bool GuiProvider::drawEditGui(TriangleWaveEncoding& config)
	{
		bool isUpdated = false;
		isUpdated |= vtxImGui::halfSpaceDragInt("Frequencies", &config.n_frequencies, 1.0f, 1, 50, "%d", 0);
		return isUpdated;

	}

	bool GuiProvider::drawEditGui(const std::string& featureName, EncodingConfig& config)
	{
		bool isUpdated = false;
		ImGui::PushID(featureName.c_str());
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::CollapsingHeader((featureName).c_str()))
		{
			ImGui::Indent();
			isUpdated = vtxImGui::halfSpaceCombo("Encoding Type", config.otype, EncodingTypeName, (int)EncodingType::EncodingTypeCount);
			switch(config.otype)
			{
			case EncodingType::Frequency:
				isUpdated |= drawEditGui(config.frequencyEncoding);
					break;
			case EncodingType::Grid:
				isUpdated |= drawEditGui(config.gridEncoding);
					break;
			case EncodingType::Identity:
				isUpdated |= drawEditGui(config.identityEncoding);
					break;
			case EncodingType::OneBlob:
				isUpdated |= drawEditGui(config.oneBlobEncoding);
					break;
			case EncodingType::SphericalHarmonics:
				isUpdated |= drawEditGui(config.sphericalHarmonicsEncoding);
					break;
			case EncodingType::TriangleWave:
				isUpdated |= drawEditGui(config.triangleWaveEncoding);
				break;
			default: ;
			}
			ImGui::Unindent();
		}
		ImGui::PopID();

		return isUpdated;
	}

	bool GuiProvider::drawEditGui(BatchGenerationConfig& settings)
	{
		if (ImGui::CollapsingHeader("Training Batch Generation Settings"))
		{
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Only Non Zero", &settings.onlyNonZero);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Mis Weight", &settings.weightByMis);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Pdf Weight", &settings.weightByPdf);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Light Sample", &settings.useLightSample);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Train on Light Sample", &settings.trainOnLightSample);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Limit To First Bounce", &settings.limitToFirstBounce);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Skip Specular Samples", &settings.skipSpecular);
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Skip Incomplete Batches", &settings.skipIncomplete);
		}
		
		return settings.isUpdated;
	}

	bool GuiProvider::drawEditGui(MlpSettings& config)
	{
		bool isUpdated = false;
		//isUpdated |= vtxImGui::halfSpaceDragInt("Hidden Dimension", &(config.hiddenDim), 1.0f, 1, 500, "%d", 0);
		isUpdated |= vtxImGui::halfSpaceIntCombo("Hidden Dimension", config.hiddenDim, {16, 32, 64, 128});
		isUpdated |= vtxImGui::halfSpaceDragInt("Number of Hidden Layers", &(config.numHiddenLayers), 1.0f, 1, 10, "%d", 0);
		return isUpdated;
	}

	bool GuiProvider::drawEditGui(MainNetEncodingConfig& config)
	{
		bool isUpdated = false;
		isUpdated |= vtxImGui::halfSpaceCheckbox("Normalize Position", &config.normalizePosition);
		isUpdated |= drawEditGui("Position Encoding", config.position);
		isUpdated |= drawEditGui("Normal Encoding", config.normal);
		isUpdated |= drawEditGui("Wo Encoding", config.wo);
		return isUpdated;
	}

	bool GuiProvider::drawEditGui(AuxNetEncodingConfig& config)
	{
		bool isUpdated = false;
		isUpdated |= drawEditGui("Wi Encoding", config.wi);
		return isUpdated;
	}

	bool GuiProvider::drawEditGui(NetworkSettings& settings)
	{
		if (ImGui::CollapsingHeader("Network Settings"))
		{
			ImGui::Indent();
			settings.wasActive = settings.active;

			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Enable", &settings.active);
			//settings.wasActive = (settings.active && !wasActive) ? false : true;
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Reset Network", &settings.isUpdated);

			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("General"))
			{
				ImGui::Indent();
				settings.isUpdated |= vtxImGui::halfSpaceWidget("Do Training", ImGui::Checkbox,(hiddenLabel + "_Do Training").c_str(),&settings.doTraining);
				settings.isUpdated |= vtxImGui::halfSpaceDragInt("Max Training Steps", &settings.maxTrainingSteps, 1, 0, 10000, "%d", 0);
				settings.isUpdated |= vtxImGui::halfSpaceWidget("Do Inference", ImGui::Checkbox,(hiddenLabel + "_Do Inference").c_str(),&settings.doInference);
				settings.isUpdated |= vtxImGui::halfSpaceDragInt("Inference Start", &settings.inferenceIterationStart, 1, 0, 10000, "%d", 0);
				settings.isUpdated |= vtxImGui::halfSpaceWidget("Clear buffer on Inference Start", ImGui::Checkbox,(hiddenLabel + "_Inference Start").c_str(),&settings.clearOnInferenceStart);
			}
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Main Network Settings"))
			{
				ImGui::Indent();
				ImGui::PushID("Main Network Settings");
				settings.isUpdated |= drawEditGui(settings.mainNetSettings);
				settings.isUpdated |= drawEditGui(settings.inputSettings);
				ImGui::PopID();
				ImGui::Unindent();
			}
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Distribution Settings"))
			{
				ImGui::Indent();
				settings.isUpdated |= vtxImGui::halfSpaceCombo("Distribution Type", settings.distributionType, distributionNames, D_COUNT);
				settings.isUpdated |= vtxImGui::halfSpaceDragInt("Mixture Size", &(settings.mixtureSize), 1.0f, 1, 10, "%d", 0);
				ImGui::Unindent();
			}

			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Additional Inputs"))
			{
				ImGui::Indent();
				settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Material Id", &settings.useMaterialId);
				settings.isUpdated |= drawEditGui("Material Id Encoding", settings.materialIdEncodingConfig);
				settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Triangle Id", &settings.useTriangleId);
				settings.isUpdated |= drawEditGui("Triangle Id Encoding", settings.triangleIdEncodingConfig);
				settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Instance Id", &settings.useInstanceId);
				settings.isUpdated |= drawEditGui("Instance Id Encoding", settings.instanceIdEncodingConfig);
				ImGui::Unindent();
			}
			settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Auxiliary Network", &settings.useAuxiliaryNetwork);
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Auxiliary Network Settings"))
			{
				ImGui::Indent();
				ImGui::PushID("Auxiliary Network Settings");
				settings.isUpdated |= vtxImGui::halfSpaceDragInt("Tot Aux Input Size", &(settings.totAuxInputSize), 1.0f, 1, 128, "%d", 0);
				settings.isUpdated |= drawEditGui(settings.auxiliaryNetSettings);
				settings.isUpdated |= drawEditGui(settings.auxiliaryInputSettings);
				settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Input Radiance Loss Factor",   &settings.inRadianceLossFactor, 0.1f, 0, 10);
				settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Output Radiance Loss Factor",   &settings.outRadianceLossFactor, 0.1f, 0, 10);
				settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Throughput Radiance Loss Factor",   &settings.throughputLossFactor, 0.1f, 0, 10);
				settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Auxiliary Weight Loss Factor", &settings.auxiliaryWeight, 0.1f, 0, 10);
				settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Target Radiance Scale Factor", &settings.throughputTargetScaleFactor, 0.1f, 0, 100);
				settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Target Throughput Scale Factor",   &settings.radianceTargetScaleFactor, 0.1f, 0, 100);
				ImGui::PopID();
				ImGui::Unindent();
			}
			ImGui::SetNextItemOpen(true, ImGuiCond_Once);
			if (ImGui::CollapsingHeader("Training"))
			{
				ImGui::Indent();
				vtxImGui::halfSpaceDragInt("BatchSize", &settings.batchSize, 1, 1, 10000, "%d", 0);
				settings.batchSize = (settings.batchSize == 0) ? 1 : settings.batchSize;
				ImGui::SetNextItemOpen(true, ImGuiCond_Once);
				settings.isUpdated |= drawEditGui(settings.trainingBatchGenerationSettings);
				ImGui::SetNextItemOpen(true, ImGuiCond_Once);
				if (ImGui::CollapsingHeader("Optimizer Settings"))
				{
					settings.isUpdated |= vtxImGui::halfSpaceDragInt("Adam Eps Exp Value", &settings.adamEps);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Learning Rate", &settings.learningRate, 0.0000001f, 0, 1, "%.00010f", 0);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Scheduler Gamma", &settings.schedulerGamma, 0.000001f, 0, 1, "%.00010f", 0);
					settings.isUpdated |= vtxImGui::halfSpaceDragInt("Scheduler Step", &settings.schedulerStep);
					settings.isUpdated |= vtxImGui::halfSpaceDragInt("L2 regularization Beta", &settings.l2WeightExp, 1, 0, 15);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Ema Inference Update", &settings.emaUpdate);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Ema Weight Decay", &settings.emaDecay, 0.1f, 0, 1);
				}
				ImGui::SetNextItemOpen(true, ImGuiCond_Once);
				if (ImGui::CollapsingHeader("Loss Settings"))
				{
					ImGui::Indent();
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Tone Map Radiance", &settings.toneMapRadiance);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Learn Input Radiance", &settings.learnInputRadiance);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Loss Smooth Clamp Value", &settings.lossClamp, 1.0f, 0.f, 1000);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Clamp Bsdf Prob", &settings.clampBsdfProb);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Scale Loss Blended", &settings.scaleLossBlendedQ);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Scale By Sample Prob", &settings.scaleBySampleProb);

					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Loss Blending Factor", &settings.blendFactor,	   0.00001, 0, 1, "%.10f", 0);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Constant Loss Blend Factor",	  &settings.constantBlendFactor);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Sampling Fraction Blend",	  &settings.samplingFractionBlend);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Sampling Fraction Blend Train %", &settings.fractionBlendTrainPercentage, 0.01, 0, 1, "%.10f", 0);
					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Clamp Sampling Fraction", &settings.clampSamplingFraction);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Clamp Value", &settings.sfClampValue, 0.01, 0, 1, "%.10f", 0);
					settings.isUpdated |= vtxImGui::halfSpaceCombo("Loss Type", settings.lossType, lossNames, L_COUNT);
					settings.isUpdated |= vtxImGui::halfSpaceCombo("Loss Reduction", settings.lossReduction,   lossReductionNames, COUNT);
					settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Target Scale", &settings.targetScale, 0.1f,	   0.0f, 1.0f);

					settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Use Entropy Loss", &settings.useEntropyLoss);
					ImGui::SetNextItemOpen(true, ImGuiCond_Once);
					if (ImGui::CollapsingHeader("Entropy Loss Settings"))
					{
						ImGui::Indent();
						settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Entropy Loss Weight", &settings.entropyWeight, 0.1f, 0, 10);
						settings.isUpdated |= vtxImGui::halfSpaceDragFloat("Target Entropy", &settings.targetEntropy, 0.1f, 0, 10);
						ImGui::Unindent();
					}
					ImGui::Unindent();
				}
				ImGui::Unindent();
			}

			ImGui::Unindent();
		}

		return settings.isUpdated;
	}

	std::vector<PlotInfo> GuiProvider::getPlots(Network& network)
	{
		auto graphsData = network.getGraphs();
		std::vector<PlotInfo> plots;

		for(auto& [PlotTitle, PlotData] : graphsData.graphs)
		{
			PlotInfo plot;
			plot.title = PlotTitle;
			plot.xLabel = PlotData.xLabel;
			plot.yLabel = PlotData.yLabel;
			for(auto& [PlotName, PlotValues] : PlotData.data)
			{
				plot.addPlot(PlotValues, PlotName);
			}
			plots.push_back(plot);
		}
		/*auto dt = network.settings.distributionType;

		PlotInfo loss;
		loss.title = "Loss";
		loss.xLabel = "Batches";
		loss.yLabel = "Loss";
		loss.addPlot(graphsData.graphs[G_NGP_T_LOSS], "Loss");
		loss.addPlot(graphsData.graphs[G_NGP_T_LOSS_Q], "Loss non Blended");
		loss.addPlot(graphsData.graphs[G_NGP_T_LOSS_BLENDED_Q], "Loss Blended");

		PlotInfo target;
		target.title = "Probabilities";
		target.xLabel = "Batches";
		target.yLabel = "Target";
		target.addPlot(graphsData.graphs[G_NGP_T_TARGET_P], "Target");
		target.addPlot(graphsData.graphs[G_NGP_T_NEURAL_P], "Neural Pdf");

		PlotInfo bsdf;
		bsdf.title = "Bsdf";
		bsdf.xLabel = "Batches";
		bsdf.yLabel = "Bsdf";
		bsdf.addPlot(graphsData.graphs[G_NGP_T_BSDF_P], "Bsdf Pdf");
		bsdf.addPlot(graphsData.graphs[G_NGP_T_BLENDED_P], "Blended Pdf Target");

		PlotInfo samplingFraction;
		samplingFraction.title = "Sampling Fraction";
		samplingFraction.xLabel = "Batches";
		samplingFraction.yLabel = "Sampling Fraction";
		samplingFraction.addPlot(graphsData.graphs[G_NGP_T_SAMPLING_FRACTION], "Sampling Fraction Train");
		samplingFraction.addPlot(graphsData.graphs[G_NGP_I_SAMPLING_FRACTION], "Sampling Fraction Inference");
		samplingFraction.addPlot(graphsData.graphs[G_NGP_TAU], "Tau");

		std::vector<PlotInfo> plots = { loss,target, samplingFraction, bsdf };

		if (dt == D_SPHERICAL_GAUSSIAN)
		{
			PlotInfo concentration;
			concentration.title = "Concentration";
			concentration.xLabel = "Batches";
			concentration.yLabel = "Concentration";
			concentration.addPlot(graphsData.graphs[G_SPHERICAL_GAUSSIAN_T_K], "Concentration Train");
			concentration.addPlot(graphsData.graphs[G_SPHERICAL_GAUSSIAN_I_K], "Concentration Inference");

			plots.push_back(concentration);
		}

		if (dt == D_NASG_ANGLE || dt == D_NASG_TRIG || dt == D_NASG_AXIS_ANGLE)
		{
			PlotInfo lambda;
			lambda.title = "Lambda";
			lambda.xLabel = "Batches";
			lambda.yLabel = "Lambda";
			lambda.addPlot(graphsData.graphs[G_NASG_T_LAMBDA], "Lambda Train");
			lambda.addPlot(graphsData.graphs[G_NASG_I_LAMBDA], "Lambda Inference");

			PlotInfo a;
			a.title = "A";
			a.xLabel = "Batches";
			a.yLabel = "A";
			a.addPlot(graphsData.graphs[G_NASG_T_A], "Anisotropy Train");
			a.addPlot(graphsData.graphs[G_NASG_I_A], "Anisotropy Inference");

			plots.push_back(lambda);
			plots.push_back(a);
		}*/

		return plots;
	}
}

