#include "AppWindow.h"
#include "Core/Options.h"
#define IMGUI_ENABLE_DOCKING
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "EnvironmentWindow.h"
#include "ExperimentsWindow.h"
#include "GraphWindow.h"
#include "PropertiesWindow.h"
#include "SceneHierarchyWindow.h"
#include "ShaderGraphWindow.h"
#include "ViewportWindow.h"
#include "Core/Application.h"
#include "Core/FileDialog.h"
#include "Core/LoadingSaving.h"
#include "GLFW/glfw3.h"

namespace vtx {

    AppWindow::AppWindow() {
        name = "AppWindow";
        useToolbar = false;
        createWindow = false;
    }


    void AppWindow::preRender()
    {
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

        // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
        // because it would be confusing to have two docking targets within each others.
        ImGuiWindowFlags window_flags = 
            ImGuiWindowFlags_NoDocking |
            //ImGuiWindowFlags_MenuBar |
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoNavFocus;

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
        // and handle the pass-thru hole, so we ask Begin() to not render a background.
        if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
            window_flags |= ImGuiWindowFlags_NoBackground;

        // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
        // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
        // all active windows docked into it will lose their parent and become undocked.
        // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
        // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace Vortex", nullptr, window_flags);
        ImGui::PopStyleVar();

        ImGui::PopStyleVar(2);

        // Submit the DockSpace
		if (const ImGuiIO& io = ImGui::GetIO(); io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
        {
            const ImGuiID dockSpaceId = ImGui::GetID("VortexDockSpace");
            ImGui::DockSpace(dockSpaceId, ImVec2(0.0f, 0.0f), dockspace_flags);
        }

        mainMenuBar();

        ImGui::End();
        ImGui::ShowDemoWindow();
    }

    void AppWindow::mainMenuBar()
    {
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {2.0f, 6.0f});
        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("File"))
            {
                if (ImGui::MenuItem("Open Demo"))
                {
                    LoadingSaving::get().manuallySetOpenFile("./assets/model/DemoSpaceship.vtx");
                }
                if (ImGui::MenuItem("Open"))
                {
                    LoadingSaving::get().loadFileDialog();
                }
                if (ImGui::MenuItem("Save"))
                {
					LoadingSaving::get().saveFileDialog();
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Window"))
            {
                if (ImGui::MenuItem("Graph View"))
                {
                    windowManager->createWindow<GraphWindow>();
                }
                if (ImGui::MenuItem("Experiments"))
                {
                    windowManager->createWindow<ExperimentsWindow>();
                }
                if (ImGui::MenuItem("Material Editor"))
                {
                    windowManager->createWindow<ShaderGraphWindow>();
                }
                if (ImGui::MenuItem("Viewport"))
                {
                    windowManager->createWindow<ViewportWindow>();
                }
                if (ImGui::MenuItem("Environment Light"))
                {
                    windowManager->createWindow<EnvironmentWindow>();
                }
                if (ImGui::MenuItem("Properties"))
                {
                    windowManager->createWindow<PropertiesWindow>();
                }
                if (ImGui::MenuItem("Scene Hierarchy"))
                {
                    windowManager->createWindow<SceneHierarchyWindow>();
                }
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleVar();
    }
}
