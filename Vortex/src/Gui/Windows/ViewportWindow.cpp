#include "ViewportWindow.h"
#include "imgui.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "NeuralNetworks/Interface/NetworkInterface.h"
#include "NeuralNetworks/Interface/NetworkInterfaceStructs.h"
#include "NeuralNetworks/Interface/NetworkInterfaceUploader.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Instance.h"
#include "Scene/Nodes/Material.h"

namespace vtx {

#define green {0.0f, 1.0f, 0.0f}
#define red {1.0f, 0.0f, 0.0f}
#define blue {0.0f, 0.0f, 1.0f}
#define orange {1.0f, 0.5f, 0.0f}
#define purple {0.5f, 0.0f, 1.0f}
#define yellow {1.0f, 1.0f, 0.0f}

    ViewportWindow::ViewportWindow() : renderer(graph::Scene::get()->renderer)
    {
		name         = "Viewport";
		useToolbar   = false;
        isBorderLess = true;
    }

    void ViewportWindow::OnUpdate(const float ts)
    {
    }

    void ViewportWindow::mainContent()
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		const int width  = (int)ImGui::GetContentRegionAvail().x;
        const int height = (int)ImGui::GetContentRegionAvail().y;
		if (!renderer->isSizeLocked)
        {
            renderer->resize(width, height);
        }
        const GlFrameBuffer& bf = renderer->getFrame();
		ImGui::Image((ImTextureID)bf.colorAttachment, ImVec2{static_cast<float>(bf.width), static_cast<float>(bf.height)}, ImVec2{0, 1}, ImVec2{1, 0});

        bool isTransforming = false;
        if (ImGui::IsItemHovered())
        {
            renderer->camera->navigationActive = true;

            if (renderer->camera->navigationMode == graph::NAV_NONE)
            {
                transformUI.monitorTransformUI(renderer->camera);
                isTransforming = transformUI.isTransforming();

                // activate selection interaction only if not transforming
				if (!isTransforming && ImGui::IsItemClicked())
				{
                    if (ImGui::GetIO().KeyCtrl)
                    {
                        //We remove all selected instances
                        graph::Scene::get()->removeInstancesFromSelection();
                    }
                    else
                    {
						const ImVec2 mousePos  = ImGui::GetMousePos();
                        const ImVec2 windowPos = ImGui::GetWindowPos();

                        const int mouseXRelativeToWindow = static_cast<int>(mousePos.x - windowPos.x);
                        const int mouseYRelativeToWindow = static_cast<int>(bf.height) - static_cast<int>(mousePos.y - windowPos.y);
						const int pixelIndex             = mouseYRelativeToWindow * width + mouseXRelativeToWindow;

                        vtxID selected = renderer->getInstanceIdOnClick(pixelIndex);

                        if (ImGui::GetIO().KeyShift)
                        {
							graph::Scene::get()->addNodesToSelection({selected});
                        }
                        else
                        {
							graph::Scene::get()->setSelected({selected});
						}
                    }
                }


                if (ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift && ImGui::GetIO().KeyAlt)
                {
                    const ImVec2 mousePos = ImGui::GetMousePos();
                    const ImVec2 windowPos = ImGui::GetWindowPos();

                    const int mouseXRelativeToWindow = static_cast<int>(mousePos.x - windowPos.x);
                    const int mouseYRelativeToWindow = static_cast<int>(bf.height) - static_cast<int>(mousePos.y - windowPos.y);
                    const int pixelIndex = mouseYRelativeToWindow * width + mouseXRelativeToWindow;

                    renderer->waveFrontIntegrator.network.settings.debugPixelId = pixelIndex;
                    renderer->waveFrontIntegrator.network.settings.isDebuggingUpdated = true;

				}
            }
        }

        // Draw pivots and transform line
        const std::set<std::shared_ptr<graph::Instance>>& instances = graph::Scene::get()->getSelectedInstances();
        if (!instances.empty())
        {
            math::vec3f meanPivot = 0.0f;
            math::vec2f projectedMeanPivot;
			for (const auto& instance : instances)
            {
				const math::vec3f pivot         = instance->transform->globalTransform.p;
				math::vec2f       onScreenPivot = renderer->camera->project(pivot, true);
                // add circle
                vtxImGui::drawOrigin(onScreenPivot);
                meanPivot += pivot;
            }
			if (instances.size() > 1)
            {
	            meanPivot /= instances.size();
                projectedMeanPivot = renderer->camera->project(meanPivot, true);
				vtxImGui::drawOrigin(projectedMeanPivot);
            }
            else
            {
                projectedMeanPivot = renderer->camera->project(meanPivot, true);
            }

			if (isTransforming)
            {
				const ImVec2 mousePos  = ImGui::GetMousePos();
                const ImVec2 windowPos = ImGui::GetWindowPos();
				vtxImGui::drawDashedLine(projectedMeanPivot, {mousePos.x - windowPos.x, mousePos.y - windowPos.y});
            }
        }

        // Draw Network Debug Vectors
        if (const auto& netSettings =renderer->waveFrontIntegrator.network.settings; netSettings.active && netSettings.doInference && netSettings.debugPixelId >=0)
        {
	        NetworkDebugInfo debugInfo = getNetworkDebugInfoFromDevice();
            const float vectorScale = 0.2f;
            vtxImGui::drawVector(renderer->camera, debugInfo.position, debugInfo.sample* vectorScale, purple);
            vtxImGui::drawVector(renderer->camera, debugInfo.position, debugInfo.bsdfSample* vectorScale, orange);

            std::vector<BounceData> debugBounceData = getPixelBounceData(netSettings.debugPixelId, renderer->settings.maxBounces);
            NetworkInterface::finalizePathStatic(netSettings, debugBounceData.size(), debugBounceData.data());

            float inf = std::numeric_limits<float>::infinity();
            math::vec3f p0 = { inf, inf, inf };

            bool bsdfFoundEmission = false;

            for (int i = 0; i < debugBounceData.size(); i++)
            {
                if(!math::isZero(debugBounceData[i].surfaceEmission.Le))
                {
                    bsdfFoundEmission = true;
					break;
				}
            }

            math::vec3f pathColor = (bsdfFoundEmission) ? math::vec3f(yellow) : math::vec3f(blue);

            for (int i = 0; i < debugBounceData.size(); i++)
            {
                if(i==(debugBounceData.size()-1))
                {
                    if(i>0){
                        vtxImGui::drawVector(renderer->camera, debugBounceData[i-1].hit.position, debugBounceData[i-1].bsdfSample.wi*vectorScale, red);
                    }
                    break;
                }

                const math::vec3f& p1 = debugBounceData[i].hit.position;
                vtxImGui::drawVector(renderer->camera, p1, debugBounceData[i].hit.normal*vectorScale, green);

                if (!math::isInf(p0))
                {
                    vtxImGui::connectScenePoints(renderer->camera, p0, p1, pathColor);
                }
                p0 = p1;
			}
        }
        ImGui::PopStyleVar();
    }

    void ViewportWindow::preRender()
    {
		return;
        if (renderer->isSizeLocked)
        {
            const float titleBarHeight = ImGui::GetFontSize() + ImGui::GetStyle().FramePadding.y * 2;
			const int   height         = renderer->height + 10 + titleBarHeight;
			const int   width          = renderer->width + 10;

            ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);
			//windowFlags |= ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize;
        }
    }
}
