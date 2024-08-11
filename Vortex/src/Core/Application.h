#pragma once
#include <GLFW/glfw3.h>
#include "Options.h"
#include "Log.h"
#include "Gui/GuiWindow.h"
#include "Scene/Scene.h"

namespace vtx
{
	static void glfwErrorCallback(const int error, const char* description)
	{
		VTX_ERROR("Glfw Error {}: {}", error, description);
	}

	class Application : public std::enable_shared_from_this<Application> {
	public:
		void init();
		void reset();

		void shutDown();

		void initWindow();
		void dataLoop();

		void run();
		void standardRun();
		void batchExperimentAppLoopBody(int i, const std::shared_ptr<graph::Renderer>& renderer);
		void BatchExperimentRun();

		static void setStartUpFile(const std::string& filePath);

		static Application* get();
		
	public:
		GLFWwindow*                    glfwWindow;
		std::shared_ptr<WindowManager> windowManager;
		float                          timeStep      = 0.0f;
		float                          frameTime     = 0.0f;
		float                          lastFrameTime = 0.0f;
		unsigned                       iteration     = 0;
		bool                           loadEM        = false;
		std::string                            EmFile;
	};
}
