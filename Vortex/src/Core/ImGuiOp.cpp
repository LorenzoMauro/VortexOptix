#include "ImGuiOp.h"
#include "Core/Utils.h"
#include "Core/Log.h"
#include "imgui.h"
#include "imnodes.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "Core/Options.h"
#include "glad/glad.h"
#include "font.h"

namespace vtx
{
	// Callback function for GLFW window UI
	void SetWindowTitleBarColor(GLFWwindow* window, const float r, const float g, const float b)
	{
		// Retrieve the GLFW window handle
		GLFWwindow* glfwWindow = static_cast<GLFWwindow*>(ImGui::GetIO().UserData);

		// Set the window UI callback functions
		glfwSetWindowAttrib(glfwWindow, GLFW_DECORATED, GLFW_FALSE);
		glfwSetWindowAttrib(glfwWindow, GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
		glfwSetWindowAttrib(glfwWindow, GLFW_FOCUS_ON_SHOW, GLFW_FALSE);
		glfwSetWindowAttrib(glfwWindow, GLFW_FLOATING, GLFW_TRUE);

		// Set the window frame color
		//glfwSetWindowMonitor(glfwWindow, nullptr, 0, 0, 0, 0, GLFW_DONT_CARE);
		//glfwSetWindowPos(glfwWindow, 0, 0);
		//glfwSetWindowSizeLimits(glfwWindow, 0, 0, GLFW_DONT_CARE, GLFW_DONT_CARE);
		//glfwSetWindowPos(glfwWindow, 0, 0);
		//glfwSetWindowSize(glfwWindow, 0, 0);
		//glfwSetWindowOpacity(glfwWindow, 1.0f);
		//glfwSetWindowAttrib(glfwWindow, GLFW_RESIZABLE, GLFW_FALSE);
		//glfwSetWindowAttrib(glfwWindow, GLFW_MAXIMIZABLE, GLFW_FALSE);
		//glfwSetWindowAttrib(glfwWindow, GLFW_DECORATED, GLFW_FALSE);
		//glfwSetWindowAttrib(glfwWindow, GLFW_FLOATING, GLFW_TRUE);

		// Set the window background color
		//glfwSetWindowSizeLimits(glfwWindow, 0, 0, GLFW_DONT_CARE, GLFW_DONT_CARE);
		//glfwSetWindowPos(glfwWindow, 0, 0);
		//glfwSetWindowSize(glfwWindow, 1, 1);
		//glfwSetWindowOpacity(glfwWindow, 1.0f);
		//glfwSetWindowAttrib(glfwWindow, GLFW_DECORATED, GLFW_FALSE);
		//glfwSetWindowAttrib(glfwWindow, GLFW_FLOATING, GLFW_TRUE);

		// Set the color of the window title bar
		GLFWimage image;
		image.width     = 1;
		image.height    = 1;
		image.pixels    = new unsigned char[4];
		image.pixels[0] = static_cast<unsigned char>(r * 255);
		image.pixels[1] = static_cast<unsigned char>(g * 255);
		image.pixels[2] = static_cast<unsigned char>(b * 255);
		image.pixels[3] = 255;
		glfwSetWindowIcon(glfwWindow, 1, &image);
		delete[] image.pixels;
	}

	void loadFonts()
	{
		ImGuiIO& io = ImGui::GetIO();
		(void)io;

		// Load Font
		// Default Font
		std::string  FontPath = utl::absolutePath(getOptions()->fontPath);
		ImFontConfig fontConfig01;
		fontConfig01.FontDataOwnedByAtlas = false;
		//ImFont* robotoFont = io.Fonts->AddFontFromFileTTF(FontPath.data(), 12.0f, &fontConfig01);
		ImFont* robotoFont = io.Fonts->AddFontFromMemoryTTF(LucidaGrande_ttf, LucidaGrande_ttf_len, 12.0f, &fontConfig01);
		io.FontDefault     = robotoFont;

		//Icond Font
		//std::string iconFontPath = Utils::absolutePath(ICON_FONT);
		//ImFontConfig fontConfig02;
		//fontConfig02.MergeMode = true;
		//fontConfig02.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
		//ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
		//io.Fonts->AddFontFromFileTTF(iconFontPath.data(), 18.0f, &fontConfig02, icon_ranges);
	}

	void Init_ImGui(GLFWwindow* window)
	{
		VTX_INFO("Starting ImGui");
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImNodes::CreateContext();
		ImPlot::CreateContext();

		ImGuiIO& io  = ImGui::GetIO();
		io.DeltaTime = 1.0f / 1000000.0f;
		(void)io;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		//io.ConfigFlags |= ImGuiConfigFlags_NavEnableSetMousePos;
		io.IniFilename = nullptr;

		ImGui::StyleColorsDark();
		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.


		// Send Vulkan Info to ImGui
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		const char* glsl_version = "#version 130";
		ImGui_ImplOpenGL3_Init(glsl_version);
		//SetWindowTitleBarColor(window, 0.2f, 0.2f, 0.2f);
		// Upload Fonts
		loadFonts();

		ImGui::LoadIniSettingsFromDisk(utl::absolutePath(getOptions()->imGuiIniFile).data());

		SetAppStyle();
		ImGuiStyle& style = ImGui::GetStyle();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			style.WindowRounding              = 0.0f;
			style.Colors[ImGuiCol_WindowBg].w = 1.0f;
		}
	}

	uint32_t vec4toImCol32(const ImVec4 color)
	{
		return IM_COL32((uint8_t)(color.x * 255.0f), (uint8_t)(color.y * 255.0f), (uint8_t)(color.z * 255.0f), (uint8_t)(color.w * 255.0f));
	}

	void SetAppStyle()
	{
		const ImVec4 darkGrey    = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
		const ImVec4 lightGrey   = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
		const ImVec4 transparent = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
		//const ImVec4 tabColor = darkGrey;
		const ImVec4 transparentGrey = ImVec4(0.26f, 0.26f, 0.26f, 0.50f);
		const ImVec4 unifiedVeryDark = ImVec4(0.09f, 0.09f, 0.09f, 1.00f);


		const ImVec4 orange         = ImVec4(0.88f, 0.50f, 0.24f, 1.00f);
		const ImVec4 orangeTownDown = ImVec4(0.98f, 0.55f, 0.26f, 1.00f);
		const ImVec4 orangeActive   = ImVec4(0.98f, 0.43f, 0.06f, 1.00f);

		const ImVec4 coolColor          = ImVec4(0.5f, 0.5f, 0.5f, 1.00f);
		const ImVec4 coolColorTonedDown = ImVec4(0.4f, 0.4f, 0.4f, 1.00f);
		const ImVec4 coolColorActive    = ImVec4(0.55f, 0.55f, 0.55f, 1.00f);


		const ImVec4 accent         = coolColor;
		const ImVec4 accentTownDown = coolColorTonedDown;
		const ImVec4 accentStrong   = coolColorActive;

		ImVec4* colors                         = ImGui::GetStyle().Colors;
		colors[ImGuiCol_Text]                  = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
		colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
		colors[ImGuiCol_ChildBg]               = darkGrey;
		colors[ImGuiCol_PopupBg]               = darkGrey;
		colors[ImGuiCol_Border]                = coolColorTonedDown;
		colors[ImGuiCol_BorderShadow]          = transparent;
		colors[ImGuiCol_FrameBg]               = lightGrey;
		colors[ImGuiCol_FrameBgHovered]        = lightGrey;
		colors[ImGuiCol_FrameBgActive]         = lightGrey;
		colors[ImGuiCol_TitleBg]               = darkGrey;
		colors[ImGuiCol_TitleBgActive]         = darkGrey;
		colors[ImGuiCol_TitleBgCollapsed]      = darkGrey;
		colors[ImGuiCol_MenuBarBg]             = darkGrey;
		colors[ImGuiCol_ScrollbarBg]           = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
		colors[ImGuiCol_ScrollbarGrab]         = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
		colors[ImGuiCol_ScrollbarGrabHovered]  = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
		colors[ImGuiCol_ScrollbarGrabActive]   = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
		colors[ImGuiCol_CheckMark]             = ImVec4(0.79f, 0.79f, 0.79f, 1.00f);
		colors[ImGuiCol_SliderGrab]            = accent;
		colors[ImGuiCol_SliderGrabActive]      = accentStrong;
		colors[ImGuiCol_Button]                = lightGrey;
		colors[ImGuiCol_ButtonHovered]         = accentTownDown;
		colors[ImGuiCol_ButtonActive]          = accentStrong;
		colors[ImGuiCol_Header]                = lightGrey;
		colors[ImGuiCol_HeaderHovered]         = ImVec4(0.36f, 0.36f, 0.36f, 0.80f);
		colors[ImGuiCol_HeaderActive]          = ImVec4(0.53f, 0.53f, 0.53f, 1.00f);
		colors[ImGuiCol_Separator]             = transparentGrey;
		colors[ImGuiCol_SeparatorHovered]      = transparentGrey;
		colors[ImGuiCol_SeparatorActive]       = transparentGrey;
		colors[ImGuiCol_ResizeGrip]            = ImVec4(0.88f, 0.00f, 1.00f, 0.20f);
		colors[ImGuiCol_ResizeGripHovered]     = ImVec4(0.97f, 0.00f, 1.00f, 0.67f);
		colors[ImGuiCol_ResizeGripActive]      = ImVec4(0.97f, 0.00f, 1.00f, 0.95f);
		colors[ImGuiCol_Tab]                   = darkGrey;
		colors[ImGuiCol_TabHovered]            = lightGrey;
		colors[ImGuiCol_TabActive]             = lightGrey;
		colors[ImGuiCol_TabUnfocusedActive]    = darkGrey;
		colors[ImGuiCol_DockingPreview]        = ImVec4(0.31f, 0.31f, 0.31f, 0.70f);
		colors[ImGuiCol_DockingEmptyBg]        = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
		colors[ImGuiCol_PlotLines]             = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
		colors[ImGuiCol_PlotLinesHovered]      = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
		colors[ImGuiCol_PlotHistogram]         = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
		colors[ImGuiCol_PlotHistogramHovered]  = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
		colors[ImGuiCol_TableHeaderBg]         = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
		colors[ImGuiCol_TableBorderStrong]     = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
		colors[ImGuiCol_TableBorderLight]      = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
		colors[ImGuiCol_TableRowBg]            = transparent;
		colors[ImGuiCol_TableRowBgAlt]         = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
		colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.27f, 0.59f, 1.00f, 0.35f);
		colors[ImGuiCol_DragDropTarget]        = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
		colors[ImGuiCol_NavHighlight]          = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
		colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
		colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
		colors[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

		colors[ImGuiCol_WindowBg]         = unifiedVeryDark;
		colors[ImGuiCol_TitleBg]          = unifiedVeryDark;
		colors[ImGuiCol_TitleBgActive]    = unifiedVeryDark;
		colors[ImGuiCol_TitleBgCollapsed] = unifiedVeryDark;
		colors[ImGuiCol_TabUnfocused]     = unifiedVeryDark;

		auto& imguiStyle = ImGui::GetStyle();
		imguiStyle.Alpha = 1.0f;

		imguiStyle.WindowPadding    = ImVec2(2.0f, 2.0f);
		imguiStyle.FramePadding     = ImVec2(2.0f, 2.0f);
		imguiStyle.CellPadding      = ImVec2(2.0f, 2.0f);
		imguiStyle.ItemSpacing      = ImVec2(2.0f, 2.0f);
		imguiStyle.ItemInnerSpacing = ImVec2(2.0f, 2.0f);
		imguiStyle.IndentSpacing    = 11.0f;
		imguiStyle.ScrollbarSize    = 16.0f;
		imguiStyle.GrabMinSize      = 4.0f;

		imguiStyle.DockingSeparatorSize = 0.0f;
		imguiStyle.WindowBorderSize     = 0;
		imguiStyle.ChildBorderSize      = 0;
		imguiStyle.PopupBorderSize      = 1;
		imguiStyle.FrameBorderSize      = 0;
		imguiStyle.TabBorderSize        = 0;

		imguiStyle.WindowRounding    = 5;
		imguiStyle.ChildRounding     = 5;
		imguiStyle.FrameRounding     = 5;
		imguiStyle.PopupRounding     = 5;
		imguiStyle.ScrollbarRounding = 5;
		imguiStyle.GrabRounding      = 5;
		imguiStyle.TabRounding       = 5;

		//imguiStyle.WindowTitleAlign = ;
		/*imguiStyle.WindowMenuButtonPosition = 1;
		imguiStyle.ColorButtonPosition = 0;*/
		//imguiStyle.ButtonTextAlign = ;

		ImNodesStyle& ImNodesStyle = ImNodes::GetStyle();

		// Set colors
		ImNodesStyle.Colors[ImNodesCol_GridBackground]         = vec4toImCol32(unifiedVeryDark);
		ImNodesStyle.Colors[ImNodesCol_NodeBackground]         = vec4toImCol32(darkGrey);
		ImNodesStyle.Colors[ImNodesCol_NodeBackgroundHovered]  = vec4toImCol32(lightGrey);
		ImNodesStyle.Colors[ImNodesCol_NodeBackgroundSelected] = vec4toImCol32(lightGrey);
		ImNodesStyle.Colors[ImNodesCol_NodeOutline]            = vec4toImCol32(lightGrey);
		ImNodesStyle.Colors[ImNodesCol_TitleBar]               = vec4toImCol32(lightGrey);
		ImNodesStyle.Colors[ImNodesCol_TitleBarHovered]        = vec4toImCol32(lightGrey);
		ImNodesStyle.Colors[ImNodesCol_TitleBarSelected]       = vec4toImCol32(lightGrey);
		ImNodesStyle.Colors[ImNodesCol_Link]                   = vec4toImCol32(accentTownDown);
		ImNodesStyle.Colors[ImNodesCol_LinkHovered]            = vec4toImCol32(accentTownDown);
		ImNodesStyle.Colors[ImNodesCol_LinkSelected]           = vec4toImCol32(accentTownDown);
		ImNodesStyle.Colors[ImNodesCol_Pin]                    = vec4toImCol32(accentTownDown);
		// ... set other colors

		// Set ImNodesStyle variables
		ImNodesStyle.NodeCornerRounding        = 5.0f;
		ImNodesStyle.NodePadding               = ImVec2(10.0f, 4.0f);
		ImNodesStyle.NodeBorderThickness       = 1.5f;
		ImNodesStyle.LinkThickness             = 1.5f;
		ImNodesStyle.LinkLineSegmentsPerLength = 0.1f;
		ImNodesStyle.BezierDeviationFactor     = 0.025;
		ImNodesStyle.LinkHoverDistance         = 10.0f;
		ImNodesStyle.PinCircleRadius           = 2.0f;
		// ... set other ImNodesStyle variables

		// Set ImNodesStyle flags
		ImNodesStyle.Flags = ImNodesStyleFlags_NodeOutline;
		ImNodesStyle.Flags |= ImNodesStyleFlags_GridLines;
		ImNodesStyle.Flags |= ImNodesStyleFlags_DrawCirclesGrid;

		ImNodesStyle.GridDotSize = 1.0f;
		ImNodesStyle.GridSpacing = 10.0;
		// ... set other style flags


		// Edit style flags
		//ImGui::Text("Style Flags");
		//ImGui::CheckboxFlags("NodeOutline", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_NodeOutline);
		//ImGui::CheckboxFlags("GridLines", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridLines);
		//ImGui::CheckboxFlags("GridLinesPrimary", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridLinesPrimary);
		//ImGui::CheckboxFlags("GridSnapping", reinterpret_cast<unsigned int*>(&style.Flags), ImNodesStyleFlags_GridSnapping);
	}

	void ImGuiRenderStart()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	void ImGuiDraw(GLFWwindow* window)
	{
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);

		ImVec4 clear_color(vtx::getOptions()->clearColor[1], vtx::getOptions()->clearColor[2], vtx::getOptions()->clearColor[2], vtx::getOptions()->clearColor[4]);
		glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		ImDrawData* main_draw_data    = ImGui::GetDrawData();
		const bool  main_is_minimized = (main_draw_data->DisplaySize.x <= 0.0f || main_draw_data->DisplaySize.y <= 0.0f);


		ImGuiIO& io = ImGui::GetIO();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}
	}

	std::string labelPrefix(const char* const label)
	{
		const float width = ImGui::CalcItemWidth();

		const float x = ImGui::GetCursorPosX();
		ImGui::Text(label);
		ImGui::SameLine();
		ImGui::SetCursorPosX(x + width * 0.5f + ImGui::GetStyle().ItemInnerSpacing.x);
		ImGui::SetNextItemWidth(-1);

		std::string labelID = "##";
		labelID += label;

		return labelID;
	}

	void shutDownImGui()
	{
		VTX_INFO("ShutDown: ImGui");
		ImGui::SaveIniSettingsToDisk((utl::absolutePath(getOptions()->imGuiIniFile)).data());
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImNodes::DestroyContext();
		ImGui::DestroyContext();
		ImPlot::DestroyContext();

		IMGUI_API
	}
}
