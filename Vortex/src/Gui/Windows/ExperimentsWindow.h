#pragma once
#include <memory>
#include "Gui/GuiWindow.h"
#include "NeuralNetworks/Experiment.h"

namespace vtx {
	namespace graph
	{
		class Renderer;
	}

    enum PopUpImage
	{
		POPUP_IMAGE_NONE,
		POPUP_IMAGE_GROUND_TRUTH,
		POPUP_IMAGE_MAPE,
		POPUP_IMAGE_MSE,
        POPUP_IMAGE_COUNT
	};

    const static inline char* popUpImageNames[] =
	{
			"None",
			"Ground Truth",
			"MAPE",
			"MSE"
	};

	class ExperimentsWindow : public Window {
    public:

        ExperimentsWindow();

        void OnUpdate(float ts) override;

        void mainContent() override;
        void toolBarContent() override;

        void startNewRender(SamplingTechnique technique);

        void mapeComputation();

        void generateGroundTruth();

        void runCurrentSettingsExperiment();

        void stopExperiment();

        void storeGroundTruth();
		void test();

	public:
        CUDABuffer                              mapeBuffer;
        const std::shared_ptr<graph::Renderer>& renderer;
        bool                                    toggleMisExperiment     = true;
        bool                                    toggleBsdfExperiment    = true;
        float                                   lossesContentPercentage = 0.75f;
        bool                                    performBatchExperiment  = false;
        bool                                    displayAll              = false;
        bool                                    displayGtImage          = false;
		bool                                    displayMSE = false;
		bool                                    displayMAPE = false;
		bool                                    displayMSEPlot = true;
		bool                                    displayMAPEPlot = true;
	};
};
