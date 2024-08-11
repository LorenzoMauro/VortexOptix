#pragma once
#include "Gui/GuiWindow.h"

namespace vtx
{
	namespace graph
	{
		class Renderer;
	}

	struct DebugBounceData;

	class DebugWindow : public Window
	{
	public:
		DebugWindow();
		void OnUpdate(float ts) override;
		void mainContent() override;
		void toolBarContent() override;

		std::vector<DebugBounceData> debugBounceData;
		std::shared_ptr<graph::Renderer> renderer;

		bool continuousDownload = false;
		bool download = false;
	};
}