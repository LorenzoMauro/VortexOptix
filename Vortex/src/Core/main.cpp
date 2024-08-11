#include "Application.h"
#include "NeuralNetworks/Networks/tcnn/test.h"

int main(const int argc, char** argv)
{
	testTccnTorch();
	vtx::Log::Init();
	vtx::Application app = vtx::Application();
	app.init();

	if (argc > 1)
	{
		std::string arg = argv[1];
		app.setStartUpFile(arg);
	}
	if (argc > 2)
	{
		int i = 2;
		while (i < argc)
		{
			std::string tag = argv[i];
			int         iStep;
			if (tag == "-E")
			{
				std::string value           = argv[i + 1];
				app.experimentArgs.filePath = value;
				iStep                       = 2;
			}
			if (tag == "-D")
			{
				app.experimentArgs.doExperiments = true;
				iStep                            = 1;
			}
			else if (tag == "-GTS")
			{
				std::string value            = argv[i + 1];
				app.experimentArgs.gtSamples = std::stoi(value);
				iStep                        = 2;
			}
			else if (tag == "-ES")
			{
				std::string value             = argv[i + 1];
				app.experimentArgs.expSamples = std::stoi(value);
				iStep                         = 2;
			}
			else if (tag == "-W")
			{
				std::string value        = argv[i + 1];
				app.experimentArgs.width = std::stoi(value);
				iStep                    = 2;
			}
			else if (tag == "-H")
			{
				std::string value         = argv[i + 1];
				app.experimentArgs.height = std::stoi(value);
				iStep                     = 2;
			}
			else if (tag == "-O")
			{
				app.experimentArgs.overwriteExperiment = true;
				iStep                                  = 1;
			}
			else if (tag == "-R")
			{
				app.experimentArgs.recomputeGt = true;
				iStep                          = 1;
			}
			else if (tag == "-OP")
			{
				app.experimentArgs.stopAfterPlanned = true;
				iStep                               = 1;
			}
			else if (tag == "-A")
			{
				app.experimentArgs.doAblationStudy = true;
				iStep                              = 1;
			}
			else
			{
				VTX_ERROR("Unknown argument: {}", tag);
				iStep = 1;
			}
			i += iStep;
		}
	}

	try
	{
		while (!glfwWindowShouldClose(app.glfwWindow))
		{
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
