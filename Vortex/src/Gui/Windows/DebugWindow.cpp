#include "DebugWindow.h"

#include "Core/CustomImGui/CustomImGui.h"
#include "Device/DevicePrograms/DebugDownloadData.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Renderer.h"
#include "Device/CudaFunctions/cudaFunctions.h"
#include "Gui/GuiElements/ImageWindowPopUp.h"

#define green math::vec3f{0.0f, 1.0f, 0.0f}
#define red math::vec3f{1.0f, 0.0f, 0.0f}
#define blue math::vec3f{0.0f, 0.0f, 1.0f}
#define orange math::vec3f{1.0f, 0.5f, 0.0f}
#define purple math::vec3f{0.5f, 0.0f, 1.0f}
#define yellow math::vec3f{1.0f, 1.0f, 0.0f}
static float vecScale = 0.2f;
static const float inf = std::numeric_limits<float>::infinity();
namespace vtx
{
	DebugWindow::DebugWindow()
	{
		isBorderLess = true;
		name = "Debug";
	}

	void DebugWindow::OnUpdate(float ts)
	{
		renderer = graph::Scene::get()->renderer;
		if(download || continuousDownload)
		{
			if (renderer->settings.debugPixel >= 0)
			{
				debugBounceData = DebugData::getFromDevice();
			}
			else {
				debugBounceData.clear();
			}
			download = false;
		}
	}

	void DebugWindow::mainContent()
	{
		const std::shared_ptr<graph::Renderer> renderer = graph::Scene::get()->renderer;

		const ImVec2 imageStartPosition = ImGui::GetCursorScreenPos();
		const GlFrameBuffer& bf = renderer->getFrame();
		ImGui::Image((ImTextureID)bf.colorAttachment, ImVec2{ static_cast<float>(bf.width), static_cast<float>(bf.height) }, ImVec2{ 0, 1 }, ImVec2{ 1, 0 });

		if (ImGui::IsItemHovered())
		{
			renderer->camera->navigationActive = true;
			if (ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift && ImGui::GetIO().KeyAlt)
			{
				const ImVec2 mousePos = ImGui::GetMousePos();

				const int mouseXRelativeToWindow = mousePos.x - imageStartPosition.x;
				int mouseYRelativeToWindow = mousePos.y - imageStartPosition.y;
				// flip y
				mouseYRelativeToWindow = bf.height - mouseYRelativeToWindow;

				const int pixelIndex = mouseYRelativeToWindow * bf.width + mouseXRelativeToWindow;
				renderer->settings.debugPixel= pixelIndex;
			}
		}
		
		math::vec3f p0;
		for (int i = 0; i < debugBounceData.size(); i++)
		{
			

			const math::vec3f& p1 = debugBounceData[i].position;

			// Surface reference Frame
			//if(i==0)
			{
				vtxImGui::drawVector(renderer->camera, p1, debugBounceData[i].shadingNormal * vecScale, green);
				vtxImGui::drawVector(renderer->camera, p1, debugBounceData[i].trueNormal * vecScale, (green+blue)/2.0f);
				vtxImGui::drawVector(renderer->camera, p1, debugBounceData[i].tangent * vecScale, yellow);
				vtxImGui::drawVector(renderer->camera, p1, debugBounceData[i].bitangent * vecScale, orange);
			}

			if (i == (debugBounceData.size() - 1))
			{
				vtxImGui::drawVector(renderer->camera, p1, debugBounceData[i].wi * vecScale, red);
			}
			if (i!= 0)
			{
				vtxImGui::connectScenePoints(renderer->camera, p0, p1, blue);
			}
			p0 = p1;
		}
	}

	void displayBounce(DebugBounceData& data)
	{
		vtxImGui::halfSpaceWidget("throughput", vtxImGui::vectorGui, (float*)&data.throughput, true);
		vtxImGui::halfSpaceWidget("Continuation Probability:", vtxImGui::booleanText, "%f", data.continuationProbability);

		if (ImGui::CollapsingHeader("Geometric Data"))
		{
			vtxImGui::halfSpaceWidget("Depth:", vtxImGui::booleanText, "%d", data.depth);
			vtxImGui::halfSpaceWidget("Position", vtxImGui::vectorGui, (float*)&data.position, true);
			vtxImGui::halfSpaceWidget("True Normal", vtxImGui::vectorGui, (float*)&data.trueNormal, true);
			vtxImGui::halfSpaceWidget("ShadingNormal", vtxImGui::vectorGui, (float*)&data.shadingNormal, true);
			vtxImGui::halfSpaceWidget("Tangent", vtxImGui::vectorGui, (float*)&data.tangent, true);
			vtxImGui::halfSpaceWidget("Bitangent", vtxImGui::vectorGui, (float*)&data.bitangent, true);
			vtxImGui::halfSpaceWidget("Uv", vtxImGui::vectorGui, (float*)&data.uv, true);
		}
		
		if (ImGui::CollapsingHeader("Bsdf Sample Data"))
		{
			//Bsdf Sample Data
			vtxImGui::halfSpaceWidget("wi", vtxImGui::vectorGui, (float*)&data.wi, true);
			vtxImGui::halfSpaceWidget("wiPdf:", vtxImGui::booleanText, "%f", data.wiPdf);
			vtxImGui::halfSpaceWidget("bsdfOverPdf", vtxImGui::vectorGui, (float*)&data.bsdfOverPdf, true);
			vtxImGui::halfSpaceWidget("Event Type:", vtxImGui::booleanText, "%d", data.eventType);
			vtxImGui::halfSpaceWidget("bsdf", vtxImGui::vectorGui, (float*)&data.bsdf, true);
			vtxImGui::halfSpaceWidget("bsdfPdf:", vtxImGui::booleanText, "%f", data.bsdfPdf);
			vtxImGui::halfSpaceWidget("bsdfSample", vtxImGui::vectorGui, (float*)&data.bsdfSample, true);

			if (data.neuralActive)
			{
				vtxImGui::halfSpaceWidget("samplingFraction:", vtxImGui::booleanText, "%f", data.samplingFraction);
				vtxImGui::halfSpaceWidget("neuralSamplePdf:", vtxImGui::booleanText, "%f", data.neuralSamplePdf);
				if (data.isNeuralSample) {
					vtxImGui::halfSpaceWidget("neuralSample", vtxImGui::vectorGui, (float*)&data.neuralSample, true);
				}
			}
		}

		if(data.isLsSample)
		{
			if (ImGui::CollapsingHeader("Light Sample Data"))
			{
				if (data.lsDoNeural)
				{
					vtxImGui::halfSpaceWidget("Neural Prob:", vtxImGui::booleanText, "%f", data.lsNeuralPdf);
					vtxImGui::halfSpaceWidget("Bsdf Prob:", vtxImGui::booleanText, "%f", data.lsBsdfPdf);
					vtxImGui::halfSpaceWidget("Bsdf Mis Prob:", vtxImGui::booleanText, "%f", data.lsBsdfMisPdf);
				}
				else
				{
					vtxImGui::halfSpaceWidget("Bsdf Prob:", vtxImGui::booleanText, "%f", data.lsBsdfMisPdf);
				}
				vtxImGui::halfSpaceWidget("lsPdf:", vtxImGui::booleanText, "%f", data.lsPdf);
				vtxImGui::halfSpaceWidget("lsWeightMis:", vtxImGui::booleanText, "%f", data.lsWeightMis);
				vtxImGui::halfSpaceWidget("lsBsdf:", vtxImGui::vectorGui, (float*)&data.lsBsdf, true);
				vtxImGui::halfSpaceWidget("lsWi:", vtxImGui::vectorGui, (float*)&data.lsWi, true);
				vtxImGui::halfSpaceWidget("lsLiOverPdf:", vtxImGui::vectorGui, (float*)&data.lsLiOverPdf, true);
			}
		}

		if(data.isSeSample)
		{
			if (ImGui::CollapsingHeader("Surface Emitter Sample Data"))
			{
				vtxImGui::halfSpaceWidget("sePdf:", vtxImGui::booleanText, "%f", data.sePdf);
				vtxImGui::halfSpaceWidget("rayPdf:", vtxImGui::booleanText, "%f", data.rayPdf);
				vtxImGui::halfSpaceWidget("seWeightMis:", vtxImGui::booleanText, "%f", data.seWeightMis);
				vtxImGui::halfSpaceWidget("seEdf:", vtxImGui::vectorGui, (float*)&data.seEdf, true);
				vtxImGui::halfSpaceWidget("seIntensity:", vtxImGui::vectorGui, (float*)&data.seIntensity, true);
				vtxImGui::halfSpaceWidget("isMisComputed:", vtxImGui::booleanText, "%d", data.isMisComputed);
				vtxImGui::halfSpaceWidget("Event Type:", vtxImGui::booleanText, "%d", data.seEventType);
				vtxImGui::halfSpaceWidget("cond 1:", vtxImGui::booleanText, "%d", data.cond1);
				vtxImGui::halfSpaceWidget("cond 2:", vtxImGui::booleanText, "%d", data.cond2);
				vtxImGui::halfSpaceWidget("cond 3:", vtxImGui::booleanText, "%d", data.cond3);
			}
		}
	}

	void DebugWindow::toolBarContent()
	{
		vtxImGui::halfSpaceDragInt("Debug Pixel", &renderer->settings.debugPixel);
		vtxImGui::halfSpaceDragInt("Debug Path", &renderer->settings.debugDepth);
		vtxImGui::halfSpaceCheckbox("Continuous Download", &continuousDownload);
		if(!continuousDownload)
		{
			vtxImGui::halfSpaceCheckbox("Download Bounce", &download);
		}
		if(debugBounceData.empty())
		{
			return;
		}

		if(const int depthToDebug = renderer->settings.debugDepth;
			depthToDebug < debugBounceData.size() 
			&& renderer->waveFrontIntegrator.network.settings.doInference 
			&& renderer->waveFrontIntegrator.network.settings.active
			)
		{
			const DebugBounceData& dI = debugBounceData[depthToDebug];
			auto& printBuffer = onDeviceData->debugData.resourceBuffers.distributionPrintBuffer;
			const int width = ImGui::GetContentRegionAvail().x;
			const int height = width / 2;
			float pdfIntegral = cuda::printDistribution(printBuffer, width, height, dI.shadingNormal, dI.neuralSample);
			vtx::gui::popUpImageWindow("Distribution", printBuffer, width, height, 3, true);
			vtxImGui::halfSpaceWidget("Pdf Integral:", vtxImGui::booleanText, "%f", pdfIntegral);

		}
		vtxImGui::halfSpaceWidget("Radiance", vtxImGui::vectorGui, (float*)&debugBounceData[0].accumulatedRadiance, true);
		if(length(debugBounceData[0].accumulatedRadiance) >  1000)
		{
			continuousDownload = false;
		}
		for (int i = 0; i < debugBounceData.size(); i++)
		{
			std::string name = "Bounce " + std::to_string(i);
			ImGui::PushID(name.c_str());
			if (ImGui::CollapsingHeader(name.c_str()))
			{
				ImGui::Indent();
				displayBounce(debugBounceData[i]);
				ImGui::Unindent();
			}
			ImGui::PopID();
			
		}
	}

}
