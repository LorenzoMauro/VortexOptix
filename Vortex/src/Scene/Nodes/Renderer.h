#pragma once
#include <cstdint>
#include "Device/glFrameBuffer.h"
#include "Scene/Node.h"
#include "Camera.h"
#include "Group.h"
#include "Core/Timer.h"
#include <mutex>
#include <condition_variable>

#include "RendererSettings.h"
#include "Statistics.h"
#include "Device/InteropWrapper.h"
#include "Device/DevicePrograms/WavefrontIntegrator.h"

namespace vtx::graph
{
	class Renderer : public Node
	{
	public:
		Renderer();

		void resize(int width, int height);

		void render();

		GlFrameBuffer getFrame();

		std::vector<std::shared_ptr<Node>> getChildren() const override;

		bool isReady(bool setBusy = false);

		void threadedRender();
		void restart();
		void prepare();

		void copyToGl();
		void setWindow(GLFWwindow* window);

		void accept(NodeVisitor& visitor) override;

		vtxID getInstanceIdOnClick(int pixelID);

		void saveFrame(std::string filePath);

	public:
		//GL Interop
		WaveFrontIntegrator waveFrontIntegrator;

		InteropWrapper interopDraw;
		InteropWrapper interopDisplay;

		std::shared_ptr<Camera>           camera;
		std::shared_ptr<Group>            sceneRoot;
		std::shared_ptr<EnvironmentLight> environmentLight;

		uint32_t         width;
		uint32_t         height;
		int              lockWidth    = 0;
		int              lockHeight   = 0;
		bool             isSizeLocked = false;
		RendererSettings settings;
		bool             resized = true;
		vtx::Timer       timer;

		Statistics statistics;

		struct ThreadData
		{
			template <typename Fn>
			ThreadData(Fn renderFunction, Renderer* instance):
				//renderThreadBusy(false),
				//bufferUpdateReady(false),
				rendererBusy(false),
				exitRenderThread(false)
			{
				renderThread = std::thread(
										   [this, renderFunction, instance]
										   {
											   while (true)
											   {
												   // start timer
												   std::unique_lock<std::mutex> lock(renderMutex);
												   renderCondition.wait(lock, [this] { return exitRenderThread || rendererBusy; });
												   if (exitRenderThread)
												   {
													   break;
												   }
												   glfwMakeContextCurrent(instance->sharedContext); // Make the shared context current in this thread
												   std::invoke(renderFunction, instance);           // Use std::invoke to call the member function pointer
												   rendererBusy = false;
												   //bufferUpdateReady = true;
											   }
										   }
										  );
			}

			~ThreadData()
			{
				{
					std::lock_guard<std::mutex> lock(renderMutex);
					exitRenderThread = true;
				}
				renderCondition.notify_one();
				renderThread.join();
			}

			//bool renderThreadBusy;
			bool                    rendererBusy;
			bool                    exitRenderThread;
			std::mutex              renderMutex;
			std::condition_variable renderCondition;
			std::thread             renderThread;
			bool                    outputBufferBusy;
			int                     bufferCount;
			//std::atomic<bool> bufferUpdateReady;
		} threadData;

		bool        resizeGlBuffer;
		GLFWwindow* sharedContext;
	};
}
