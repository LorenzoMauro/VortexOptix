#include "Gui/GuiProvider.h"

#include "imgui.h"
#include "Core/Application.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Device/Wrappers/KernelTimings.h"
#include "Scene/Nodes/Renderer.h"

namespace vtx::gui
{
	static const std::string hiddenLabel = "##hidden";

	bool fireflySettingsEditorGui(FireflySettings& settings)
	{
		if (ImGui::CollapsingHeader("Firefly Filtering"))
		{
			if (vtxImGui::halfSpaceWidget("Enable", ImGui::Checkbox, (hiddenLabel + "Enable Fireflies Filtering").c_str(), &settings.active))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Iteration Start", ImGui::DragInt, (hiddenLabel + "Iteration Start").c_str(), &settings.start, 1.0f, -2, 30, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Kernel Size", ImGui::DragInt, (hiddenLabel + "FireFly Kernel Size").c_str(), &settings.kernelSize, 1.0f, 1, 30, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Threshold", ImGui::DragFloat, (hiddenLabel + "FireFly Threshold").c_str(), &settings.threshold, 0.01f, 1.0f, 5.0f, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	bool denoiserSettingsEditorGui(DenoiserSettings& settings)
	{
		if (ImGui::CollapsingHeader("Denoiser Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Enable", ImGui::Checkbox, (hiddenLabel + "Enable Denoiser").c_str(), &settings.active))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Denoiser Iteration Start", ImGui::DragInt, (hiddenLabel + "Denoiser Iteration Start").c_str(), &settings.denoiserStart, 1, 1, 1000, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Denoiser Blend", ImGui::DragFloat, (hiddenLabel + "Denoiser Blend").c_str(), &settings.denoiserBlend, 0.01f, 0.0f, 1.0f, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	bool toneMapperSettingsEditorGui(ToneMapperSettings& settings)
	{
		if (ImGui::CollapsingHeader("Tone Mapper Settings"))
		{
			if (vtxImGui::halfSpaceWidget("White Point", vtxImGui::colorPicker, (hiddenLabel + "_White Point").c_str(), &settings.whitePoint.x))
			{
				settings.invWhitePoint = 1.0f / settings.whitePoint;
				settings.isUpdated     = true;
			}
			if (vtxImGui::halfSpaceWidget("Color balance", vtxImGui::colorPicker, (hiddenLabel + "_Color balance").c_str(), &settings.colorBalance.x))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Burn Highlights", ImGui::DragFloat, (hiddenLabel + "_Burn Highlights").c_str(), &settings.burnHighlights, 0.01f, 0.0f, 1.0f, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Crush Blacks", ImGui::DragFloat, (hiddenLabel + "_Crush Blacks").c_str(), &settings.crushBlacks, 0.01f, 0.0f, 1.0f, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Saturation", ImGui::DragFloat, (hiddenLabel + "_Saturation").c_str(), &settings.saturation, 0.01f, 0.0f, 2.0f, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Gamma", ImGui::DragFloat, (hiddenLabel + "_Gamma").c_str(), &settings.gamma, 0.01f, 0.1f, 5.0f, "%.3f", 0))
			{
				settings.invGamma  = 1.0f / settings.gamma;
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	bool postProcessingSettingsEditorGui(RendererSettings& settings)
	{
		bool isUpdated = false;
		if (ImGui::CollapsingHeader("Post Processing"))
		{
			isUpdated |= fireflySettingsEditorGui(settings.fireflySettings);
			ImGui::Separator();
			isUpdated |= denoiserSettingsEditorGui(settings.denoiserSettings);
			ImGui::Separator();
			isUpdated |= toneMapperSettingsEditorGui(settings.toneMapperSettings);
		}

		return isUpdated;
	}

	bool adaptiveSettingsEditorGui(AdaptiveSamplingSettings& settings)
	{
		if (ImGui::CollapsingHeader("Adaptive Sampling"))
		{
			if (vtxImGui::halfSpaceWidget("Enable", ImGui::Checkbox, (hiddenLabel + "_Adaptive Sampling").c_str(), &settings.active))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Noise Kernel Size", ImGui::DragInt, (hiddenLabel + "_Noise Kernel Size").c_str(), &settings.noiseKernelSize, 1, 0, 40, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Start Adaptive Sample Sample", ImGui::DragInt, (hiddenLabel + "_Start Adaptive Sample Sample").c_str(), &settings.minAdaptiveSamples, 1, 0, 500, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Albedo-Normal Noise Influence", ImGui::DragFloat, (hiddenLabel + "_Albedo-Normal Noise Influence").c_str(), &settings.albedoNormalNoiseInfluence, 0.01f, 0.0f, 1.0f, "%.3f", 0))
			{
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	bool wavefrontSettingsEditorGui(WavefrontSettings& settings)
	{
		if (ImGui::CollapsingHeader("WaveFront Settings"))
		{
			if (vtxImGui::halfSpaceWidget("Use WaveFront", ImGui::Checkbox, (hiddenLabel + "_Use Wavefront").c_str(), &settings.active))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Long Path Mega Kernel", ImGui::Checkbox, (hiddenLabel + "_Long Path Mega Kernel").c_str(), &settings.useLongPathKernel))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Wavefront Long Path Kernel Start", ImGui::DragFloat, (hiddenLabel + "_Wavefront Long Path Kernel Start").c_str(), &settings.longPathPercentage, 0.01, 0, 1, "%.3f", 0))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Fit Kernel Launch", ImGui::Checkbox, (hiddenLabel + "_Fit Kernel Launch").c_str(), &settings.fitWavefront))
			{
				settings.isUpdated = true;
			}

			if (vtxImGui::halfSpaceWidget("Use Optix Shader Kernel", ImGui::Checkbox, (hiddenLabel + "_Use Optix Shader Kernel").c_str(), &settings.optixShade))
			{
				settings.isUpdated = true;
			}
		}

		return settings.isUpdated;
	}

	bool quadrantTechniqueSettingsSplit(QuadrantTechniqueSplit& settings, std::string quadrantName)
	{
		bool isUpdated = false;
		isUpdated |= vtxImGui::halfSpaceCombo((quadrantName + "Sampling").c_str(), settings.st, samplingTechniqueNames, S_COUNT);
		isUpdated |= vtxImGui::halfSpaceCheckbox((quadrantName + "Neural Activated").c_str(), &settings.neuralActivated);
		return isUpdated;
	}

	bool quadrantSettingsEditorGui(QuadrantsSettings& settings)
	{
		settings.isUpdated |= vtxImGui::halfSpaceCheckbox("Activated", &settings.isActivated);
		if (ImGui::CollapsingHeader("Quadrant Settings"))
		{
			settings.isUpdated |= quadrantTechniqueSettingsSplit(settings.topLeft, "Top Left ");
			settings.isUpdated |= quadrantTechniqueSettingsSplit(settings.topRight, "Top Right ");
			settings.isUpdated |= quadrantTechniqueSettingsSplit(settings.bottomLeft, "Bottom Left ");
			settings.isUpdated |= quadrantTechniqueSettingsSplit(settings.bottomRight, "Bottom Right ");
		}
		return settings.isUpdated;
	}

	void GuiProvider::drawDisplayGui(graph::Statistics& statistics)
	{
		if (ImGui::CollapsingHeader("Statistics"))
		{
			vtxImGui::halfSpaceWidget("Total Time:", vtxImGui::booleanText, "%.3f s", statistics.totTimeSeconds);
			vtxImGui::halfSpaceWidget("Samples Per Pixels:", vtxImGui::booleanText, "%d", statistics.samplesPerPixel);
			vtxImGui::halfSpaceWidget("SPP per Seconds:", vtxImGui::booleanText, "%.3f", statistics.sppPerSecond);
			vtxImGui::halfSpaceWidget("Frame Time:", vtxImGui::booleanText, "%.3f ms", statistics.frameTime);
			vtxImGui::halfSpaceWidget("Fps:", vtxImGui::booleanText, "%.3f", statistics.fps);
			ImGui::Separator();
			vtxImGui::halfSpaceWidget("CPU Fps:", vtxImGui::booleanText, "%.3f", statistics.internalFps);
			vtxImGui::halfSpaceWidget("CPU Tot Time:", vtxImGui::booleanText, "%.3f", statistics.totTimeInternal);
			ImGui::Separator();
			vtxImGui::halfSpaceWidget("Renderer Noise", vtxImGui::booleanText, "%.2f ms", statistics.rendererNoise);
			vtxImGui::halfSpaceWidget("Renderer Trace", vtxImGui::booleanText, "%.2f ms", statistics.rendererTrace);
			vtxImGui::halfSpaceWidget("Renderer Post", vtxImGui::booleanText, "%.2f ms", statistics.rendererPost);
			vtxImGui::halfSpaceWidget("Renderer Display", vtxImGui::booleanText, "%.2f ms", statistics.rendererDisplay);
			ImGui::Separator();
			vtxImGui::halfSpaceWidget("WaveFront Generate Ray		", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontGenerateRay);
			vtxImGui::halfSpaceWidget("WaveFront Trace				", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontTrace);
			vtxImGui::halfSpaceWidget("WaveFront Shade				", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontShade);
			vtxImGui::halfSpaceWidget("WaveFront Shadow				", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontShadow);
			vtxImGui::halfSpaceWidget("WaveFront Escaped			", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontEscaped);
			vtxImGui::halfSpaceWidget("WaveFront Accumulate			", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontAccumulate);
			vtxImGui::halfSpaceWidget("WaveFront Reset				", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontReset);
			vtxImGui::halfSpaceWidget("WaveFront Fetch Queue Size	", vtxImGui::booleanText, "%.2f ms", statistics.waveFrontFetchQueueSize);
			ImGui::Separator();																			 
			vtxImGui::halfSpaceWidget("Neural Prepare Dataset		", vtxImGui::booleanText, "%.2f ms", statistics.neuralShuffleDataset);
			vtxImGui::halfSpaceWidget("Neural Network Train			", vtxImGui::booleanText, "%.2f ms", statistics.neuralNetworkTrain);
			vtxImGui::halfSpaceWidget("Neural Network Infer			", vtxImGui::booleanText, "%.2f ms", statistics.neuralNetworkInfer);
		}
	}

	bool GuiProvider::drawEditGui(RendererSettings& settings)
	{
		bool restartRendering = false;
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);
		if (ImGui::CollapsingHeader("General Settings"))
		{
			vtxImGui::halfSpaceWidget("Samples Per Pixels:", vtxImGui::booleanText, "%d", settings.iteration);


			if (vtxImGui::halfSpaceWidget("Max Bounces", ImGui::DragInt, (hiddenLabel + "_Max Bounces").c_str(), &(settings.maxBounces), 1.0f, 1, 1000000, "%d", 0))
			{
				settings.isUpdated = true;
				restartRendering   = true;
			}
			if (vtxImGui::halfSpaceWidget("Max Samples", ImGui::DragInt, (hiddenLabel + "_Max Samples").c_str(), &settings.maxSamples, 1.0f, 1, 1000000, "%d", 0))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Accumulate", ImGui::Checkbox, (hiddenLabel + "_Accumulate").c_str(), &settings.accumulate))
			{
				settings.isUpdated = true;
				restartRendering   = true;
			}
			if (vtxImGui::halfSpaceWidget("Sampling Technique", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Sampling Technique").c_str(), reinterpret_cast<int*>(&settings.samplingTechnique), samplingTechniqueNames, S_COUNT, -1))
			{
				settings.isUpdated = true;
				restartRendering   = true;
			}

			restartRendering |= quadrantSettingsEditorGui(settings.quadrantsSettings);
			settings.isUpdated |= restartRendering;

			if (vtxImGui::halfSpaceWidget("Display Buffer Type", (ComboFuncType)ImGui::Combo, (hiddenLabel + "_Display Buffer Type").c_str(), reinterpret_cast<int*>(&settings.displayBuffer), displayBufferNames, FB_COUNT, -1))
			{
				settings.isUpdated = true;
			}
			if (vtxImGui::halfSpaceWidget("Min Clip", ImGui::DragFloat, (hiddenLabel + "_Min Clip").c_str(), &settings.minClamp, 0.01, 0, 1000, "%.3f", 0))
			{
				settings.isUpdated = true;
				restartRendering   = true;
			}
			if (vtxImGui::halfSpaceWidget("Max Clip", ImGui::DragFloat, (hiddenLabel + "_Max Clip").c_str(), &settings.maxClamp, 1, 0, 10000, "%.3f", 0))
			{
				settings.isUpdated = true;
				restartRendering   = true;
			}
			if (vtxImGui::halfSpaceWidget("Russian Roulette", ImGui::Checkbox, (hiddenLabel + "_Russian Roulette").c_str(), &settings.useRussianRoulette))
			{
				settings.isUpdated = true;
				restartRendering   = true;
			}
			if (vtxImGui::halfSpaceWidget("Separate Thread", ImGui::Checkbox, (hiddenLabel + "_Separate Thread").c_str(), &settings.runOnSeparateThread))
			{
				settings.isUpdated = true;
				restartRendering = true;
			}
		}
		return restartRendering;
	}

	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::Renderer>& renderNode)
    {
		const float availableWidth = ImGui::GetContentRegionAvail().x;
		if (ImGui::CollapsingHeader("Renderer"))
		{
			ImGui::Indent();
			ImGui::PushItemWidth(availableWidth);
			vtxImGui::halfSpaceWidget("Node Id", ImGui::Text, std::to_string(renderNode->getUID()).c_str());
			ImGui::Separator();

			bool restartRendering = false;

			const float availableWidth = ImGui::GetContentRegionAvail().x;

			ImGui::PushItemWidth(availableWidth); // Set the width of the next widget to 200

			const bool runOnSeparateThread = renderNode->settings.runOnSeparateThread;
			restartRendering |= drawEditGui(renderNode->settings);
			if (runOnSeparateThread != renderNode->settings.runOnSeparateThread)
			{
				renderNode->resizeGlBuffer = true;
			}

			ImGui::Separator();

			restartRendering |= wavefrontSettingsEditorGui(renderNode->waveFrontIntegrator.settings);

			ImGui::Separator();

			restartRendering |= adaptiveSettingsEditorGui(renderNode->settings.adaptiveSamplingSettings);

			if (renderNode->waveFrontIntegrator.network.settings.active && renderNode->settings.adaptiveSamplingSettings.active)
			{
				//	CURRENTLY NOT SUPPORTED
				renderNode->settings.adaptiveSamplingSettings.active = false;
			}


			ImGui::Separator();

			drawEditGui(renderNode->waveFrontIntegrator.network.settings);

			ImGui::Separator();

			postProcessingSettingsEditorGui(renderNode->settings);

			ImGui::Separator();

			renderNode->statistics.update(renderNode);
			drawDisplayGui(renderNode->statistics);

			ImGui::PopItemWidth();


			if (renderNode->waveFrontIntegrator.settings.active)
			{
				if (renderNode->waveFrontIntegrator.network.settings.active)
				{
					if (renderNode->waveFrontIntegrator.network.settings.isUpdated)
					{
						renderNode->waveFrontIntegrator.network.reset();
						restartRendering = true;
					}
				}
			}
			else
			{
				renderNode->waveFrontIntegrator.network.settings.active = false;
			}


			if (renderNode->waveFrontIntegrator.network.settings.isAnyUpdated())
			{
				restartRendering = true;
			}

			if (restartRendering)
			{
				renderNode->restart();
			}
		}

		return false;
    }
}
