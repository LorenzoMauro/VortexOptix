#include "NodeEditor.h"

#include "imnodes_internal.h"
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::gui
{
	static std::map<graph::NodeType, std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> nodeColors;

	ImVec4 HSLtoRGB(const float h, const float s, const float l)
	{
		float r, g, b;

		if (s == 0)
		{
			r = g = b = l; // achromatic
		}
		else
		{
			auto hue2rgb = [](const float p, const float q, float t)
			{
				if (t < 0.f) t += 1.f;
				if (t > 1.f) t -= 1.f;
				if (t < 1.f / 6.f) return p + (q - p) * 6.f * t;
				if (t < 1.f / 2.f) return q;
				if (t < 2.f / 3.f) return p + (q - p) * (2.f / 3.f - t) * 6.f;
				return p;
			};

			float q = l < 0.5f ? l * (1.f + s) : l + s - l * s;
			float p = 2.f * l - q;

			r = hue2rgb(p, q, h + 1.f / 3.f);
			g = hue2rgb(p, q, h);
			b = hue2rgb(p, q, h - 1.f / 3.f);
		}

		return {r, g, b, 1.0f};
	}

	uint32_t vec4toImCol32(const ImVec4 color)
	{
		return IM_COL32((uint8_t)(color.x * 255.0f), (uint8_t)(color.y * 255.0f), (uint8_t)(color.z * 255.0f), (uint8_t)(color.w * 255.0f));
	}

	const std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>& colorByNodeType(const graph::NodeType type)
	{
		// Check if color is already computed
		if (nodeColors.find(type) != nodeColors.end())
		{
			return nodeColors[type];
		}


		constexpr auto  darkGrey        = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
		constexpr auto  lightGrey       = ImVec4(0.2f, 0.2f, 0.2f, 1.0f);
		constexpr auto  accent          = ImVec4(0.3f, 0.3f, 0.3f, 1.0f);
		constexpr auto  super           = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
		constexpr float fixedSaturation = 0.4f; // 60%

		// Define hue ranges
		constexpr std::pair<float, float> hueRangeViolets = {0.75f, 1.0f}; // Including the wrap-around nature of hue.
		constexpr std::pair<float, float> hueRangeOrange  = {0.04f, 0.16f};
		constexpr std::pair<float, float> hueRangeGreen   = {0.16f, 0.43f};
		constexpr std::pair<float, float> hueRangeBlue    = {0.43f, 0.75f};

		// Assigning hue ranges to groups
		constexpr std::pair<float, float> hueRendering = hueRangeViolets;
		constexpr std::pair<float, float> hueMeshes    = hueRangeOrange;
		constexpr std::pair<float, float> hueLights    = hueRangeBlue;
		constexpr std::pair<float, float> hueMaterials = hueRangeGreen;

		constexpr int numRenderingNodes = 2;  // Adjust this to the correct number of nodes for the rendering group
		constexpr int numMeshNodes      = 4;  // Similarly, adjust for the mesh group
		constexpr int numLightNodes     = 3;  // Adjust for the light group
		constexpr int numMaterialNodes  = 15; // Adjust for the material group

		// Calculate hue increments for each group
		constexpr float hueIncrementRendering = (hueRendering.second - hueRendering.first) / numRenderingNodes;
		constexpr float hueIncrementMeshes    = (hueMeshes.second - hueMeshes.first) / numMeshNodes;
		constexpr float hueIncrementLights    = (hueLights.second - hueLights.first) / numLightNodes;
		constexpr float hueIncrementMaterials = (hueMaterials.second - hueMaterials.first) / numMaterialNodes;

		float hue = 0.0f; // Default hue
		switch (type)
		{
		//Geometry
		case graph::NT_GROUP: hue = hueMeshes.first + 0 * hueIncrementMeshes;
			break;
		case graph::NT_INSTANCE: hue = hueMeshes.first + 1 * hueIncrementMeshes;
			break;
		case graph::NT_MESH: hue = hueMeshes.first + 2 * hueIncrementMeshes;
			break;
		case graph::NT_TRANSFORM: hue = hueMeshes.first + 3 * hueIncrementMeshes;
			break;
		//Lights
		case graph::NT_LIGHT: hue = hueLights.first + 0 * hueIncrementLights;
			break;
		case graph::NT_MESH_LIGHT: hue = hueLights.first + 1 * hueIncrementLights;
			break;
		case graph::NT_ENV_LIGHT: hue = hueLights.first + 2 * hueIncrementLights;
			break;
		//Rendering
		case graph::NT_CAMERA: hue = hueRendering.first + 0 * hueIncrementRendering;
			break;
		case graph::NT_RENDERER: hue = hueRendering.first + 1 * hueIncrementRendering;
			break;
		//Shaders
		case graph::NT_MATERIAL: hue = hueMaterials.first + 0 * hueIncrementMaterials;
			break;
		case graph::NT_MDL_TEXTURE: hue = hueMaterials.first + 1 * hueIncrementMaterials;
			break;
		case graph::NT_MDL_BSDF: hue = hueMaterials.first + 2 * hueIncrementMaterials;
			break;
		case graph::NT_MDL_LIGHTPROFILE: hue = hueMaterials.first + 3 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_DF: hue = hueMaterials.first + 4 * hueIncrementMaterials;
			break;
		case graph::NT_PRINCIPLED_MATERIAL:
		case graph::NT_SHADER_MATERIAL: hue = hueMaterials.first + 5 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_SURFACE: hue = hueMaterials.first + 6 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_IMPORTED: hue = hueMaterials.first + 7 * hueIncrementMaterials;
			break;
		case graph::NT_GET_CHANNEL: hue = hueMaterials.first + 8 * hueIncrementMaterials;
			break;
		case graph::NT_NORMAL_MIX: hue = hueMaterials.first + 9 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_COORDINATE: hue = hueMaterials.first + 10 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_NORMAL_TEXTURE: hue = hueMaterials.first + 11 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_MONO_TEXTURE: hue = hueMaterials.first + 12 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_COLOR_TEXTURE: hue = hueMaterials.first + 13 * hueIncrementMaterials;
			break;
		case graph::NT_SHADER_BUMP_TEXTURE: hue = hueMaterials.first + 14 * hueIncrementMaterials;
			break;

		case graph::NT_NUM_NODE_TYPES: hue = 0.69f;
			break; // This might be an enum limit, so be cautious if it's used in logic.
		default:
			{
				nodeColors[type] = {vec4toImCol32(lightGrey), vec4toImCol32(darkGrey), vec4toImCol32(accent), vec4toImCol32(super)};
				return nodeColors[type];
			}
		}


		const ImVec4 lightColor  = HSLtoRGB(hue, fixedSaturation, lightGrey.x); // lightGrey Value
		const ImVec4 darkColor   = HSLtoRGB(hue, fixedSaturation, darkGrey.x);  // darkGrey Value
		const ImVec4 accentColor = HSLtoRGB(hue, fixedSaturation, accent.x);    // accent Value
		const ImVec4 superColor  = HSLtoRGB(hue, fixedSaturation, super.x);     // accent Value

		nodeColors[type] = {vec4toImCol32(lightColor), vec4toImCol32(darkColor), vec4toImCol32(accentColor), vec4toImCol32(superColor)};

		return nodeColors[type];
	}


	NodeEditorVisitor::NodeEditorVisitor()
	{
		collectWidthsAndDepths = true;
	}

	void NodeEditorVisitor::visit(const std::shared_ptr<graph::Node>& node)
	{
		nodeEditor->submitNode(node);
	}

	NodeEditor::NodeEditor()
	{
		visitor.nodeEditor      = this;
		guiVisitor.isNodeEditor = true;
	}

	void NodeEditor::updateNodeLinks(const std::shared_ptr<graph::Node>& node)
	{
		// TODO Here we should check if the node has changed.
		NodeInfo&                                       nodeInfo         = nodes[node->getUID()];
		const std::vector<std::shared_ptr<graph::Node>> children         = node->getChildren();
		const int                                       numberOfChildren = children.size();

		nodeInfo.links.clear();


		if (const std::shared_ptr<graph::shader::ShaderNode>& shaderNode = node->as<graph::shader::ShaderNode>(); shaderNode)
		{
			for (const auto& [socketName, socket] : shaderNode->sockets)
			{
				if (socket.node)
				{
					LinkInfo linkInfo;
					linkInfo.linkId              = socket.linkId;
					linkInfo.inputSocketId       = socket.Id;
					linkInfo.childOutputSocketId = socket.node->getUID();
					linkInfo.childNodeType       = socket.node->getType();
					nodeInfo.links.push_back(linkInfo);
				}
			}
		}
		else
		{
			for (int i = 0; i < numberOfChildren; i++)
			{
				LinkInfo linkInfo;
				linkInfo.inputSocketId       = node->getUID() * 1000 + i;
				linkInfo.linkId              = node->getUID() * 1000 + children[i]->getUID();
				linkInfo.childOutputSocketId = children[i]->getUID();
				linkInfo.childNodeType       = children[i]->getType();
				nodeInfo.links.push_back(linkInfo);
			}
		}
	}

	void NodeEditor::submitNode(const std::shared_ptr<graph::Node>& node)
	{
		runVisitedNodes.insert(node->getUID());
		if (nodes.find(node->getUID()) == nodes.end() || nodes[node->getUID()].node.lock() == nullptr)
		{
			NodeInfo nodeInfo;
			nodeInfo.node = node;
			//nodeInfo.nodeType = node->getType();
			nodeInfo.title = node->name;
			//nodeInfo.id = node->getUID();
			//nodeInfo.depth = node->treePosition.depth;
			//nodeInfo.width = node->treePosition.width;
			//nodeInfo.overallWidth = node->treePosition.overallWidth;
			nodes[node->getUID()] = nodeInfo;
		}


		updateNodeLinks(node);
	}

	void NodeEditor::traverseAndGetNodeInfos()
	{
		if (rootNode == nullptr)
		{
			if (flags & ImNodesStyleFlags_VerticalLayout)
			{
				const std::vector<std::shared_ptr<graph::Node>> allNodes = graph::Scene::getSim()->getAllNodes();
				for (const auto& node : allNodes)
				{
					submitNode(node);
				}
			}
			return;
		}
		rootNode->traverse(visitor);
	}

	void NodeEditor::pushNodeColorByNodeType(const graph::NodeType& nodeType)
	{
		const auto [light, dark, accent, super] = colorByNodeType(nodeType);
		ImNodes::PushColorStyle(ImNodesCol_NodeBackground, dark);
		ImNodes::PushColorStyle(ImNodesCol_NodeBackgroundHovered, light);
		ImNodes::PushColorStyle(ImNodesCol_NodeBackgroundSelected, light);
		ImNodes::PushColorStyle(ImNodesCol_NodeOutline, accent);
		ImNodes::PushColorStyle(ImNodesCol_TitleBar, light);
		ImNodes::PushColorStyle(ImNodesCol_TitleBarHovered, accent);
		ImNodes::PushColorStyle(ImNodesCol_TitleBarSelected, accent);
	}

	void NodeEditor::popNodeColorByNodeType()
	{
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
	}

	void NodeEditor::pushPinColorByChildrenNodeType(const graph::NodeType& nodeType)
	{
		const auto& [light, dark, accent, super] = colorByNodeType(nodeType);
		ImNodes::PushColorStyle(ImNodesCol_Pin, accent);
		ImNodes::PushColorStyle(ImNodesCol_PinHovered, super);
	}

	void NodeEditor::popPinColorByChildrenNodeType()
	{
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
	}

	void NodeEditor::pushLinkColorByChildNodeType(const graph::NodeType& nodeType)
	{
		const auto& [light, dark, accent, super] = colorByNodeType(nodeType);
		ImNodes::PushColorStyle(ImNodesCol_Link, accent);
		ImNodes::PushColorStyle(ImNodesCol_LinkHovered, super);
		ImGui::PushStyleColor(ImNodesCol_LinkSelected, super);
	}

	void NodeEditor::popLinkColorByChildNodeType()
	{
		ImNodes::PopColorStyle();
		ImNodes::PopColorStyle();
		ImGui::PopStyleColor();
	}

	bool NodeEditor::verticalArrangementDrawNode(NodeInfo& nodeInfo) const
	{
		const int   numberOfInputSockets = nodeInfo.links.size();
		float       nodeWidth            = spacing * std::max(numberOfInputSockets, 1);
		const float titleWidth           = ImGui::CalcTextSize(nodeInfo.title.c_str()).x;
		nodeWidth                        = std::max(nodeWidth, titleWidth);
		const float localSpacing         = nodeWidth / std::max(numberOfInputSockets, 1);
		const float leftTitlePad         = (nodeWidth - titleWidth) / 2;

		const std::shared_ptr<graph::Node>& node     = nodeInfo.node.lock();
		const vtxID                         id       = node->getUID();
		const graph::NodeType               nodeType = node->getType();

		nodesWithColor ? pushNodeColorByNodeType(nodeType) : 0;

		ImNodes::BeginNode(id);

		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
		nodeInfo.size.x = std::max(nodeInfo.size.x, getOptions()->nodeWidth); // Enforce a minimum size
		ImGui::PushItemWidth(nodeInfo.size.x);                                // Set the width of the next widget to 200

		ImNodes::BeginNodeTitleBar();
		ImGui::Dummy(ImVec2(leftTitlePad, 0));
		ImGui::SameLine();
		ImGui::TextUnformatted((nodeInfo.title).c_str());
		ImNodes::EndNodeTitleBar();


		//{
		//const int                          depth = node->treePosition.depth;
		//const int                          width = node->treePosition.width;
		//const int                          overallWidth = node->treePosition.overallWidth;
		//ImGui::Text("UID: %d", (int)node->getUID());
		//ImGui::Text("TID: %d", (int)node->getTypeID());
		//ImGui::Text("Use Count %d", (int)node.use_count());
		//ImGui::Text("Depth: %d", depth);
		//ImGui::Text("Width: %d", width);
		//ImGui::Text("Overall Width %d", overallWidth);
		//	ImGui::Text("Position x: %d, y: %d", (int)ImNodes::GetNodeEditorSpacePos(nodeInfo.id).x, (int)ImNodes::GetNodeEditorSpacePos(nodeInfo.id).y);
		//
		//}

		nodesWithColor ? pushPinColorByChildrenNodeType(nodeType) : 0;
		ImNodes::BeginOutputAttribute(id);
		ImGui::Dummy(ImVec2(nodeWidth, 0)); // Placeholder for input
		ImNodes::EndOutputAttribute();
		nodesWithColor ? popPinColorByChildrenNodeType() : 0;

		for (int idx = 0; idx < numberOfInputSockets; idx++)
		{
			nodesWithColor ? pushPinColorByChildrenNodeType(nodeInfo.links[idx].childNodeType) : 0;
			ImNodes::BeginInputAttribute(nodeInfo.links[idx].inputSocketId);
			ImGui::Dummy(ImVec2(localSpacing, 0)); // Placeholder for input
			ImNodes::EndInputAttribute();
			nodesWithColor ? popPinColorByChildrenNodeType() : 0;

			if (idx < numberOfInputSockets - 1)
			{
				ImGui::SameLine();
			}
		}

		ImGui::PopStyleVar();
		ImNodes::EndNode();

		nodesWithColor ? popNodeColorByNodeType() : 0;

		ImGui::PopItemWidth();

		return false;
	}

	void NodeEditor::nodeResizeButton(math::vec2f& size, const int id) const
	{
		// Invisible button for corner dragging
		ImVec2 corner     = ImGui::GetItemRectMax(); // Get the maximum point of the last drawn item (should be the last input/output attribute)
		int    cornerSize = 20;                      // Size of the draggable corner
		corner.x -= cornerSize;                      // Make sure the corner does not overlap the node content
		corner.y -= cornerSize;

		ImGui::SetCursorScreenPos(corner);
		ImGui::InvisibleButton(("cornerbutton" + std::to_string(id)).c_str(), ImVec2((float)cornerSize, (float)cornerSize));

		if (ImGui::IsMouseHoveringRect(corner, ImVec2(corner.x + cornerSize, corner.y + cornerSize)))
		{
			ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
		}

		if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
		{
			// Update node size based on mouse drag delta
			size.x += (int)ImGui::GetIO().MouseDelta.x;
			size.y += (int)ImGui::GetIO().MouseDelta.y;
		}
		//ImGui::GetWindowDrawList()->AddRectFilled(corner, ImVec2(corner.x + cornerSize, corner.y + cornerSize), IM_COL32(255, 0, 0, 255));

		// Dummy item with the size of the node
		ImGui::SetCursorScreenPos(ImGui::GetItemRectMin()); // Set cursor to the minimum point of the last drawn item (should be the last input/output attribute)
		ImGui::Dummy(ImVec2((float)size.x, (float)size.y));
	}

	bool NodeEditor::horizontalArrangementDrawNode(NodeInfo& nodeInfo)
	{
		bool                                changed = false;
		const std::shared_ptr<graph::Node>& node    = nodeInfo.node.lock();
		const int                           id      = node->getUID();

		ImNodes::BeginNode(id);

		nodeInfo.size.x = std::max(nodeInfo.size.x, getOptions()->nodeWidth); // Enforce a minimum size
		nodeInfo.size.y = std::max(nodeInfo.size.y, getOptions()->nodeWidth);
		ImGui::PushItemWidth(nodeInfo.size.x);
		guiVisitor.changed = false;
		node->accept(guiVisitor);
		changed = guiVisitor.changed;
		//{
		//	ImGui::Text("Size x: %d, y: %d", (int)nodeInfo.size.x, (int)nodeInfo.size.y);
		//	ImGui::Text("Position x: %d, y: %d", (int)ImNodes::GetNodeEditorSpacePos(id).x, (int)ImNodes::GetNodeEditorSpacePos(id).y);
		//	const float totalItemWidth = ImGui::GetContentRegionAvail().x;
		//	ImGui::Text("Total item width: %f", totalItemWidth);
		//}
		ImGui::PopItemWidth();

		ImNodes::EndNode();
		nodeInfo.pos = {ImNodes::GetNodeEditorSpacePos(id).x, ImNodes::GetNodeEditorSpacePos(id).y};

		nodeResizeButton(nodeInfo.size, id);

		return changed;
	}

	bool NodeEditor::drawNode(NodeInfo& node)
	{
		if (flags & ImNodesStyleFlags_VerticalLayout)
		{
			return verticalArrangementDrawNode(node);
		}
		else
		{
			return horizontalArrangementDrawNode(node);
		}
	}

	void NodeEditor::drawLinks(const NodeInfo& nodeInfo)
	{
		for (const auto& linkInfo : nodeInfo.links)
		{
			nodesWithColor ? pushLinkColorByChildNodeType(linkInfo.childNodeType) : 0;
			ImNodes::Link(linkInfo.linkId, linkInfo.inputSocketId, linkInfo.childOutputSocketId);
			nodesWithColor ? popLinkColorByChildNodeType() : 0;
		}
	}

	void NodeEditor::arrangeNodes(const LayoutDirection direction)
	{
		std::map<int, std::vector<vtxID>> depthToNodes;
		std::map<int, ImVec2>             depthToBoundingBox; // Bounding box for each depth

		// Collect nodes per depth and calculate bounding box
		for (auto& [nodeId, nodeInfo] : nodes)
		{
			const std::shared_ptr<graph::Node>& node         = nodeInfo.node.lock();
			int                                 depth        = node->treePosition.depth;
			const int                           overallWidth = node->treePosition.overallWidth;
			if (depthToNodes.find(depth) == depthToNodes.end())
			{
				depthToNodes[depth] = std::vector<vtxID>();
			}
			if (depthToNodes[depth].size() <= overallWidth)
			{
				depthToNodes[depth].resize(overallWidth + 1, 0);
			}
			depthToNodes[depth][overallWidth] = nodeId;
			ImVec2 nodeDim                    = ImNodes::GetNodeDimensions(nodeId);

			if (direction == LayoutDirection::Vertical)
			{
				depthToBoundingBox[depth].x += nodeDim.x + treeWidthPadding;                    // Width accumulates
				depthToBoundingBox[depth].y = std::max(depthToBoundingBox[depth].y, nodeDim.y); // Max height
			}
			else
			{
				depthToBoundingBox[depth].y += nodeDim.y + treeWidthPadding;                    // Height accumulates
				depthToBoundingBox[depth].x = std::max(depthToBoundingBox[depth].x, nodeDim.x); // Max width
			}
		}

		const ImNodesContext* nodesContext           = ImNodes::GetCurrentContext();
		const ImRect          availableSpace         = nodesContext->CanvasRectScreenSpace;
		const ImVec2          contentRegionAvailable = availableSpace.GetSize();
		//const ImVec2 contentRegionAvailable = ImGui::GetWindowSize();
		// Vertical Layout
		if (direction == LayoutDirection::Vertical)
		{
			const float windowWidth = contentRegionAvailable.x;
			float       currentY    = treeDepthPadding;

			for (int depth = 0; depth < depthToNodes.size(); ++depth)
			{
				const ImVec2& boundingBox = depthToBoundingBox[depth];
				// Centering nodes
				float currentX = (windowWidth - boundingBox.x) / 2.0f;

				for (const unsigned int nodeId : depthToNodes[depth])
				{
					if (nodeId == 0) continue; // Skip dummy node //TODO this shouldn't happen
					const ImVec2 nodeDim    = ImNodes::GetNodeDimensions(nodeId);
					ImVec2       currentPos = {currentX, currentY};
					ImNodes::SetNodeEditorSpacePos(nodeId, currentPos);

					currentX += nodeDim.x + treeWidthPadding;
				}
				currentY += depthToBoundingBox[depth].y + treeDepthPadding;
			}
		}
		// Horizontal Layout
		else
		{
			//const float windowHeight = contentRegionAvailable.y;
			float currentX = contentRegionAvailable.x - depthToBoundingBox[0].x - treeDepthPadding;

			for (int depth = 0; depth < depthToNodes.size(); ++depth)
			{
				// Centering nodes
				//float currentY = (windowHeight - depthToBoundingBox[depth].y * depthToNodes[depth].size() - padding * (depthToNodes[depth].size() - 1)) / 2.0f;
				float currentY = treeWidthPadding;
				for (const unsigned int nodeId : depthToNodes[depth])
				{
					const ImVec2 nodeDim    = ImNodes::GetNodeDimensions(nodeId);
					ImVec2       currentPos = {currentX, currentY};
					ImNodes::SetNodeEditorSpacePos(nodeId, currentPos);

					currentY += nodeDim.y + treeWidthPadding;
				}
				currentX -= depthToBoundingBox[depth].x + treeDepthPadding;
			}
		}
	}

	void NodeEditor::removeUnvisitedNodes()
	{
		std::vector<vtxID> nodesToRemove; // Collecting IDs to remove

		// Identify nodes to remove
		for (const auto& [id, _] : nodes)
		{
			if (runVisitedNodes.find(id) == runVisitedNodes.end())
			{
				nodesToRemove.push_back(id);
			}
		}

		// Remove the nodes
		for (const auto& id : nodesToRemove)
		{
			nodes.erase(id);
		}
	}

	void NodeEditor::updateNodeSelection()
	{
		std::set<vtxID> selectedNodes = graph::Scene::get()->getSelected();
		ImNodes::ClearNodeSelection();
		for (const vtxID id : selectedNodes)
		{
			const bool doesExists = (ImNodes::ObjectPoolFind(ImNodes::EditorContextGet().Nodes, (int)id) >= 0);
			if (doesExists && !ImNodes::IsNodeSelected((int)id))
			{
				ImNodes::SelectNode((int)id);
			}
		}
	}

	bool NodeEditor::draw()
	{
		runVisitedNodes.clear();
		traverseAndGetNodeInfos();
		removeUnvisitedNodes();

		ImNodes::EditorContextSet(sceneGraphContext);
		ImNodes::BeginNodeEditor();
		ImNodes::PushAttributeFlag(flags);

		bool updated = false;
		for (auto& [nodeId, node] : nodes)
		{
			updated |= drawNode(node);
		}
		for (auto& [nodeId, node] : nodes)
		{
			drawLinks(node);
		}

		if (isFirstTime)
		{
			if (flags & ImNodesStyleFlags_VerticalLayout)
			{
				arrangeNodes(LayoutDirection::Vertical);
			}
			else
			{
				//arrangeNodes(LayoutDirection::Horizontal);
			}
			isFirstTime = false;
		}

		updateNodeSelection();
		ImNodes::EndNodeEditor();
		ImNodes::PopAttributeFlag();


		return updated;
	}

	std::set<vtxID> NodeEditor::getSelected()
	{
		if (const int numberOfSelectedNodes = ImNodes::NumSelectedNodes(); numberOfSelectedNodes != 0)
		{
			std::vector<int> selectedImNodesId;
			std::set<vtxID>  selectedVtxID;
			selectedImNodesId.resize(numberOfSelectedNodes);
			ImNodes::GetSelectedNodes(selectedImNodesId.data());
			for (int i = 0; i < numberOfSelectedNodes; ++i)
			{
				selectedVtxID.insert((vtxID)selectedImNodesId[i]);
			}
			return selectedVtxID;
		}

		return {};
	}

	bool NodeEditor::styleEditor()
	{
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
		static ImNodesStyle& style = ImNodes::GetStyle();

		ImGui::Text("Node Editor Style");

		bool rearrangeNodes = false;
		// Edit style variables
		std::string hiddenLabel = "##hidden";
		vtxImGui::halfSpaceWidget("Grid Spacing", ImGui::DragFloat, (hiddenLabel + "GridSpacing").c_str(), &style.GridSpacing, 0.01f, 10.0f, 50.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Node Corner Rounding", ImGui::DragFloat, (hiddenLabel + "NodeCornerRounding").c_str(), &style.NodeCornerRounding, 0.01f, 0.0f, 10.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Node Padding", ImGui::DragFloat2, (hiddenLabel + "NodePadding").c_str(), reinterpret_cast<float*>(&style.NodePadding), 0.01f, 0.0f, 20.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Node Border Thickness", ImGui::DragFloat, (hiddenLabel + "NodeBorderThickness").c_str(), &style.NodeBorderThickness, 0.01f, 0.0f, 5.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Link Thickness", ImGui::DragFloat, (hiddenLabel + "LinkThickness").c_str(), &style.LinkThickness, 0.01f, 0.0f, 5.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Link LineSegmentsPerLength", ImGui::DragFloat, (hiddenLabel + "LinkLineSegmentsPerLength").c_str(), &style.LinkLineSegmentsPerLength, 0.01f, 0.1f, 1.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Link HoverDistance", ImGui::DragFloat, (hiddenLabel + "LinkHoverDistance").c_str(), &style.LinkHoverDistance, 0.01f, 0.0f, 20.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Bezier Divergence Factor", ImGui::DragFloat, (hiddenLabel + "BezierDivergenceFactor").c_str(), &style.BezierDeviationFactor, 0.01f, 0.0f, 2.0f, "%.3f", 0);
		vtxImGui::halfSpaceWidget("Pin CircleRadius", ImGui::DragFloat, (hiddenLabel + "PinCircleRadius").c_str(), &style.PinCircleRadius, 0.01f, 1.0f, 10.0f, "%.3f", 0);
		//vtxImGui::halfSpaceWidget("Pin QuadSideLength", ImGui::DragFloat, (hiddenLabel + "PinQuadSideLength").c_str(), &style.PinQuadSideLength, 0.01f, 1.0f, 10.0f,"%.3f", 0);
		//vtxImGui::halfSpaceWidget("Pin TriangleSideLength", ImGui::DragFloat, (hiddenLabel + "PinTriangleSideLength").c_str(), &style.PinTriangleSideLength, 0.01f, 1.0f, 10.0f,"%.3f", 0);
		//vtxImGui::halfSpaceWidget("Pin LineThickness", ImGui::DragFloat, (hiddenLabel + "PinLineThickness").c_str(), &style.PinLineThickness, 0.01f, 0.1f, 5.0f,"%.3f", 0);
		//vtxImGui::halfSpaceWidget("Pin HoverRadius", ImGui::DragFloat, (hiddenLabel + "PinHoverRadius").c_str(), &style.PinHoverRadius, 0.01f, 0.0f, 20.0f,"%.3f", 0);
		//vtxImGui::halfSpaceWidget("Pin Offset", ImGui::DragFloat, (hiddenLabel + "PinOffset").c_str(), &style.PinOffset, 0.01f, 0.0f, 20.0f,"%.3f", 0);
		//vtxImGui::halfSpaceWidget("MiniMap Padding", ImGui::DragFloat2, (hiddenLabel + "MiniMapPadding").c_str(), reinterpret_cast<float*>(&style.MiniMapPadding), 0.01f, 0.0f, 20.0f,"%.3f", 0);
		//vtxImGui::halfSpaceWidget("MiniMap Offset", ImGui::DragFloat2,(hiddenLabel + "MiniMapOffset").c_str(),				reinterpret_cast<float*>(&style.MiniMapOffset), 0.01f, 0.0f, 100.0f,"%.3f", 0);

		rearrangeNodes |= ImGui::SliderFloat("Tree Width Padding", &treeWidthPadding, 0.0f, 100.0f);
		rearrangeNodes |= ImGui::SliderFloat("Tree Depth Padding", &treeDepthPadding, 0.0f, 100.0f);

		// Edit style flags
		ImGui::Text("Style Flags");
		auto chosenCheckBoxOverload = static_cast<bool(*)(const char*, int*, int)>(ImGui::CheckboxFlags);

		vtxImGui::halfSpaceWidget("NodeOutline", chosenCheckBoxOverload, (hiddenLabel + "NodeOutline").c_str(), (&style.Flags), ImNodesStyleFlags_NodeOutline);
		vtxImGui::halfSpaceWidget("GridLines", chosenCheckBoxOverload, (hiddenLabel + "GridLines").c_str(), (&style.Flags), ImNodesStyleFlags_GridLines);
		vtxImGui::halfSpaceWidget("GridLinesPrimary", chosenCheckBoxOverload, (hiddenLabel + "GridLinesPrimary").c_str(), (&style.Flags), ImNodesStyleFlags_GridLinesPrimary);
		vtxImGui::halfSpaceWidget("GridSnapping", chosenCheckBoxOverload, (hiddenLabel + "GridSnapping").c_str(), (&style.Flags), ImNodesStyleFlags_GridSnapping);
		vtxImGui::halfSpaceWidget("GridSnapping", chosenCheckBoxOverload, (hiddenLabel + "GridSnapping").c_str(), (&style.Flags), ImNodesStyleFlags_DrawCirclesGrid);
		ImGui::PopItemWidth();

		return rearrangeNodes;
	}

	void NodeEditor::toolBar()
	{
		ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
		bool rearrange = ImGui::Button("Arrange Nodes");
		ImGui::PopItemWidth();

		ImGui::Separator();
		if (ImGui::CollapsingHeader("Node Editor Style"))
		{
			rearrange |= styleEditor();
		}

		if (rearrange)
		{
			if (flags & ImNodesStyleFlags_VerticalLayout)
			{
				arrangeNodes(LayoutDirection::Vertical);
			}
			else
			{
				arrangeNodes(LayoutDirection::Horizontal);
			}
		}
	}
}
