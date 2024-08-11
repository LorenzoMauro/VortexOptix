#include <imgui.h>
#include <string>
#include "Gui/GuiProvider.h"
#include "Core/Utils.h"
#include <imnodes.h>
#include "Core/CustomImGui/CustomImGui.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::gui
{
	bool GuiProvider::drawEditGui(const graph::shader::ShaderNodeSocket& socket)
	{
		graph::shader::ParameterInfo param = socket.parameterInfo;

		std::string HiddenIdentifier = "##hidden" + param.annotation.displayName;

		bool changed = false;
		// Choose proper edit control depending on the parameter kind
		switch (param.kind)
		{
		case graph::shader::PK_FLOAT:

			changed |= vtxImGui::halfSpaceDragFloat(param.annotation.displayName.c_str(), &param.data<float>(), 0.1f, param.annotation.range[0], param.annotation.range[1]);
			break;
		case graph::shader::PK_FLOAT2:
			changed |= ImGui::SliderFloat2(param.annotation.displayName.c_str(), &param.data<float>(), param.annotation.range[0], param.annotation.range[1]);
			break;
		case graph::shader::PK_FLOAT3:
			changed |= ImGui::SliderFloat3(param.annotation.displayName.c_str(), &param.data<float>(), param.annotation.range[0], param.annotation.range[1]);
			break;
		case graph::shader::PK_COLOR:
			changed |= vtx::vtxImGui::halfSpaceWidget(param.annotation.displayName.c_str(), vtxImGui::colorPicker, HiddenIdentifier.c_str(), &param.data<float>());
			break;
		case graph::shader::PK_BOOL:
			changed |= vtxImGui::halfSpaceWidget(param.annotation.displayName.c_str(), ImGui::Checkbox, HiddenIdentifier.c_str(), &param.data<bool>());
			break;
		case graph::shader::PK_INT:
			changed |= vtxImGui::halfSpaceWidget(param.annotation.displayName.c_str(), ImGui::SliderInt, HiddenIdentifier.c_str(), &param.data<int>(), int(param.annotation.range[0]), int(param.annotation.range[1]), "%d", 0);
			break;
		case graph::shader::PK_ARRAY:
			{
				ImGui::Text("%s", param.annotation.displayName.c_str());
				ImGui::Indent(16.0f);

				char* ptr = &param.data<char>();

				for (mi::Size i = 0, n = param.arraySize; i < n; ++i)
				{
					std::string idxStr = std::to_string(i);

					switch (param.arrayElemKind)
					{
					case graph::shader::PK_FLOAT:
						changed |= ImGui::SliderFloat(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.annotation.range[0], param.annotation.range[1]);
						break;
					case graph::shader::PK_FLOAT2:
						changed |= ImGui::SliderFloat2(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.annotation.range[0], param.annotation.range[1]);
						break;
					case graph::shader::PK_FLOAT3:
						changed |= ImGui::SliderFloat3(idxStr.c_str(), reinterpret_cast<float*>(ptr), param.annotation.range[0], param.annotation.range[1]);
						break;
					case graph::shader::PK_COLOR:
						changed |= ImGui::ColorEdit3(idxStr.c_str(), reinterpret_cast<float*>(ptr));
						break;
					case graph::shader::PK_BOOL:
						changed |= vtxImGui::halfSpaceWidget(param.annotation.displayName.c_str(), ImGui::Checkbox, HiddenIdentifier.c_str(), reinterpret_cast<bool*>(ptr));
						break;
					case graph::shader::PK_INT:
						changed |= ImGui::SliderInt(param.annotation.displayName.c_str(), reinterpret_cast<int*>(ptr), int(param.annotation.range[0]), int(param.annotation.range[1]));
						break;
					default:
						std::cerr << "ERROR: guiWindow() Material parameter " << param.annotation.displayName.c_str() << " array element " << idxStr << " type invalid or unhandled.\n";
					}
					ptr += param.arrayPitch;
				}
				ImGui::Unindent(16.0f);
			}
			break;
		case graph::shader::PK_ENUM:
			{
				const int   value = param.data<int>();
				std::string currentValue;

				const graph::shader::EnumTypeInfo* info = param.enumInfo;
				for (size_t i = 0, n = info->values.size(); i < n; ++i)
				{
					if (info->values[i].value == value)
					{
						currentValue = info->values[i].name;
						break;
					}
				}

				if (ImGui::BeginCombo(param.annotation.displayName.c_str(), currentValue.c_str()))
				{
					for (size_t i = 0, n = info->values.size(); i < n; ++i)
					{
						const std::string& name = info->values[i].name;

						const bool isSelected = (currentValue == name);

						if (ImGui::Selectable(info->values[i].name.c_str(), isSelected))
						{
							param.data<int>() = info->values[i].value;
							changed           = true;
						}
						if (isSelected)
						{
							ImGui::SetItemDefaultFocus();
						}
					}
					ImGui::EndCombo();
				}
			}
			break;
		case graph::shader::PK_STRING:
		case graph::shader::PK_TEXTURE:
			{
				std::string inputText = mdl::getTexturePathFromExpr(socket.directExpression);

				//int textureId = param.data<int>();
				//std::string inputText = material->textures[textureId - 1]->filePath;
				//
				//if(inputText.empty())
				//{
				//    inputText = material->textures[textureId - 1]->databaseName;
				//}
				if (!inputText.empty())
				{
					vtxImGui::halfSpaceWidget("Texture:", vtxImGui::clippedText, inputText.c_str());
				}
			}
		case graph::shader::PK_LIGHT_PROFILE:
		case graph::shader::PK_BSDF_MEASUREMENT:
			// Currently not supported by this example
			break;
		default:
			break;
		}
		return changed;
	}

	bool GuiProvider::drawEditGui(const std::shared_ptr<graph::shader::ShaderNode>& shaderNode, const bool isNodeEditor)
	{
		bool              changed           = false;
		const std::string functionSignature = shaderNode->functionInfo.name;
		const std::string title             = (utl::splitString(functionSignature, "::")).back();
		bool              isTitleHeaderOpen = false;

		ImGui::PushID(shaderNode->getUID());
		if (isNodeEditor)
		{
			ImNodes::BeginNodeTitleBar();
			ImGui::TextUnformatted(title.c_str());
			ImNodes::EndNodeTitleBar();
		}
		else
		{
			isTitleHeaderOpen = ImGui::CollapsingHeader(title.c_str());
		}

		if (isNodeEditor || isTitleHeaderOpen)
		{
			vtxImGui::halfSpaceWidget("Function Signature", vtxImGui::clippedText, functionSignature.c_str());
			vtxImGui::halfSpaceWidget("Node Id:", vtxImGui::clippedText, std::to_string(shaderNode->getUID()).c_str());

			isNodeEditor ? ImNodes::BeginOutputAttribute(shaderNode->getUID()) : 0;
			vtxImGui::halfSpaceWidget("Output Type", vtxImGui::clippedText, shaderNode->outputSocket.parameterInfo.annotation.displayName.c_str());
			isNodeEditor ? ImNodes::EndOutputAttribute() : 0;

			//TODO: Some socket appear with no description
			for (auto& [SocketGroup, socketGroupName] : shaderNode->socketsGroupedByGroup)
			{
				vtxImGui::halfSpaceWidget(" ", vtxImGui::booleanText, " ");
				vtxImGui::halfSpaceWidget("Group:", vtxImGui::clippedText, SocketGroup.c_str());
				for (const auto& socketName : socketGroupName)
				{
					std::string name   = socketName;
					auto&       socket = shaderNode->sockets[name];

					if (socket.node)
					{
						if (isNodeEditor)
						{
							ImNodes::BeginInputAttribute(socket.Id);
							ImGui::Text(name.c_str());
							ImNodes::EndInputAttribute();
						}
						else
						{
							ImGui::Text((name + " :").c_str());
							changed |= drawEditGui(socket.node);
						}
					}
					else
					{
						if (!socket.parameterInfo.annotation.displayName.empty())
						{
							isNodeEditor ? ImNodes::BeginInputAttribute(socket.Id) : 0;
							changed |= drawEditGui(socket);
							isNodeEditor ? ImNodes::EndInputAttribute() : 0;
						}
					}
				}
			}
		}
		ImGui::PopID();

		if (changed)
		{
			shaderNode->state.isShaderArgBlockUpdated = true;
		}

		return changed;
	}
}
