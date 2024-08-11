#include "Application.h"

#include "ShutDownOperations.h"
#include "glad/glad.h"
#include "ImGuiOp.h"
#include "LoadingSaving.h"
#include "Device/OptixWrapper.h"
#include "Device/PipelineConfiguration.h"
#include "Device/UploadCode/DeviceDataCoordinator.h"
#include "Gui/Windows/AppWindow.h"
#include "Gui/Windows/EnvironmentWindow.h"
#include "Gui/Windows/ShaderGraphWindow.h"
#include "Gui/Windows/ViewportWindow.h"
#include "Gui/Windows/ExperimentsWindow.h"
#include "Gui/Windows/SceneHierarchyWindow.h"
#include "Gui/Windows/GraphWindow.h"
#include "Gui/Windows/PropertiesWindow.h"
#include "MDL/MdlWrapper.h"
#include "Scene/Nodes/Material.h"
#include "Serialization/Serializer.h"

namespace vtx
{
	static Application* appInstance = nullptr;

	bool isWindowMinimized(GLFWwindow* window)
	{
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		return (width == 0 || height == 0);
	}

	void Application::init()
	{
		appInstance = this;
		initWindow();
		Input::SetWindowHandle(glfwWindow);
		Init_ImGui(glfwWindow);

		optix::init();
		pipelineConfiguration();
		mdl::init();

		graph::Scene::get()->init();

		windowManager = std::make_shared<WindowManager>();
		windowManager->createWindow<AppWindow>();
		windowManager->createWindow<ExperimentsWindow>();
		windowManager->createWindow<SceneHierarchyWindow>();
		windowManager->createWindow<ViewportWindow>();
		windowManager->createWindow<PropertiesWindow>();
		windowManager->createWindow<EnvironmentWindow>();
		windowManager->createWindow<ShaderGraphWindow>();
		windowManager->createWindow<GraphWindow>();

	}

	void Application::reset()
	{
		onDeviceData->reset();
		optix::getRenderingPipeline()->invalidate();
		cudaDeviceReset();
		optix::init();
	}

	void Application::shutDown()
	{
		//Last run!
		shutDownOperations();
		glfwDestroyWindow(glfwWindow);
		glfwTerminate();
		VTX_INFO("GLFW DESTROYED");
	}

	void framebuffer_size_callback(GLFWwindow* window, const int width, const int height)
	{
		// Here you can set your new window size. You could, for example, update
		// your options object. Without knowing the structure of your options
		// object or your application, here's a hypothetical example:

		getOptions()->width  = width;
		getOptions()->height = height;

		// You may also need to update your OpenGL viewport size here:
		glViewport(0, 0, width, height);
	}

	void Application::initWindow()
	{
		glfwSetErrorCallback(glfwErrorCallback);
		if (!glfwInit())
		{
			VTX_ERROR("GLFW: Could not initalize GLFW!");
			return;
		}
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE); // Set the window to start maximized

		// Get the primary monitor
		GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
		const GLFWvidmode* mode    = glfwGetVideoMode(monitor);

		getOptions()->width  = mode->width;
		getOptions()->height = mode->height;

		glfwWindow = glfwCreateWindow(getOptions()->width, getOptions()->height, getOptions()->windowName.c_str(), nullptr, nullptr);
		glfwMakeContextCurrent(glfwWindow);
		if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
		{
			VTX_ERROR("Failed to create GLFW window");
		}
		glfwSwapInterval(0); // Enable vsync
		// Initialize the window
		glfwSetWindowUserPointer(glfwWindow, this);
		glfwSetScrollCallback(glfwWindow, 
			[](GLFWwindow* window, const double xOffset, const double yOffset)
			{
				ImGuiIO& io = ImGui::GetIO();
				io.MouseWheelH += (float)xOffset;
				io.MouseWheel += (float)yOffset;
			}
		);
		//glfwSetFramebufferSizeCallback(m_Window, FramebufferResizeCallback);
	}

	static HostVisitor hostVisitor;

	void Application::dataLoop()
	{
		graph::Scene*                           scene    = graph::Scene::get();
		const std::shared_ptr<graph::Renderer>& renderer = scene->renderer;
		renderer->camera->onUpdate(timeStep);
		//This step speed up the material computation, but is not really coherent with the rest of the code
		graph::computeMaterialsMultiThreadCode();
		renderer->traverse(hostVisitor);
		if (renderer->settings.runOnSeparateThread)
		{
			if (renderer->isReady())
			{
				renderer->prepare();
				renderer->threadedRender();
			}
		}
		else
		{
			renderer->prepare();
			renderer->render();
		}
	}

	void Application::run()
	{
		if (const std::shared_ptr<ExperimentsWindow> ew = windowManager->getWindow<ExperimentsWindow>();
			ew != nullptr && ew->performBatchExperiment)
		{
			BatchExperimentRun();
			ew->performBatchExperiment = false;
		}
		else
		{
			standardRun();
		}
	}

	void Application::standardRun()
	{
		glfwPollEvents();
		constexpr int maxTries = 5;
		int           tryCount = 0;
		while (tryCount < maxTries)
		{
			try
			{
				dataLoop();
				break;
			}
			catch (std::exception& e)
			{
				VTX_ERROR("Standard exception in experiment {}", e.what());
			}
			catch (...)
			{
				VTX_ERROR("Unknown exception in experiment");
			}
			if (const auto& renderer = graph::Scene::get()->renderer; renderer->waveFrontIntegrator.network.settings.active == true)
			{
				renderer->waveFrontIntegrator.network.reset();
			}
			else
			{
				tryCount++;
			}
		}
		windowManager->updateWindows(timeStep);

		if (!isWindowMinimized(glfwWindow))
		{
			ImGuiRenderStart();
			if (iteration > 3) // it seems the minimum frame to render the initial gui
			{
				LoadingSaving::get().performLoadSave();
				if (loadEM && LoadingSaving::get().getCurrentState() == LoadingSaving::LoadSaveState::Idle)
				{
					ExperimentsManager* em;
					if (utl::fileExists(EmFile))
					{
						vtx::serializer::deserializeExperimentManager(EmFile);
						loadEM = false;
					}
					else
					{
						graph::Scene::get()->renderer->waveFrontIntegrator.network.experimentManager.saveFilePath = EmFile;
					}
					if (const std::shared_ptr<ExperimentsWindow> ew = windowManager->getWindow<ExperimentsWindow>(); ew != nullptr)
					{
						ew->performBatchExperiment = true;
					}

				}
			}
			windowManager->renderWindows();
			ImGuiDraw(glfwWindow);
			glfwSwapBuffers(glfwWindow);
		}
		windowManager->removeClosedWindows();
		const auto time = static_cast<float>(glfwGetTime());
		frameTime       = time - lastFrameTime;
		timeStep        = std::min(frameTime, 0.0333f);
		lastFrameTime   = time;
		iteration += 1;
	}

	void Application::batchExperimentAppLoopBody(int i, const std::shared_ptr<graph::Renderer>& renderer)
	{
		glfwPollEvents();
		if (i == 0)
		{
			renderer->camera->onUpdate(timeStep);
			graph::computeMaterialsMultiThreadCode();
			renderer->traverse(hostVisitor);
		}
		renderer->render();

		{
			glfwPollEvents();
			windowManager->updateWindows(timeStep);

			if (!isWindowMinimized(glfwWindow))
			{
				ImGuiRenderStart();
				if (iteration > 3) // it seems the minimum frame to render the initial gui
				{
					LoadingSaving::get().performLoadSave();
				}
				windowManager->renderWindows();
				ImGuiDraw(glfwWindow);
				glfwSwapBuffers(glfwWindow);
			}
			windowManager->removeClosedWindows();
			const auto time = static_cast<float>(glfwGetTime());
			frameTime       = time - lastFrameTime;
			timeStep        = std::min(frameTime, 0.0333f);
			lastFrameTime   = time;
			iteration += 1;
		}
	}

	void Application::BatchExperimentRun()
	{
		auto& em = graph::Scene::get()->renderer->waveFrontIntegrator.network.experimentManager;
		em.BatchExperimentRun();
	}

	void Application::setStartUpFile(const std::string& filePath)
	{
		LoadingSaving::get().setManualLoad(filePath);
	}

	Application* Application::get()
	{
		return appInstance;
	}
}
