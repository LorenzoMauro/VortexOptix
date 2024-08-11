#include "Application.h"
#include "NeuralNetworks/Networks/tcnn/test.h"

int main(const int argc, char** argv) {
	testTccnTorch();
	vtx::Log::Init();
	vtx::Application app = vtx::Application();
	app.init();

	if (argc > 1) {
		std::string arg = argv[1];
		app.setStartUpFile(arg);
	}
	if (argc > 2)
	{
		std::string arg = argv[2];
		if (arg == "-E")
		{
			app.loadEM = true;
			app.EmFile = argv[3];
		}
	}

	try {
		while (!glfwWindowShouldClose(app.glfwWindow)) {
			app.run();
		}
	}
	catch (const std::exception& e)
	{
		VTX_ERROR("Error: {}", e.what());
	}
	app.shutDown();

	return 0;
}
