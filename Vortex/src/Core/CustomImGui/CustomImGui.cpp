#include "CustomImGui.h"

#include <stack>
#include <vector>
#include "Core/Log.h"
#include "Scene/Nodes/Camera.h"

namespace vtx::vtxImGui
{
	const char* emptyFormatCallback()
	{
		return "";
	}

	bool sliderFloat(const char* label, float* value, const float min, const float max, const char* format)
	{
		// Get the current style
		const ImGuiStyle& style = ImGui::GetStyle();
		ImGui::PushStyleColor(ImGuiCol_SliderGrab, IM_COL32(255, 255, 255, 0)); // Hide the slider grab

		// Set the format callback to our custom callback
		const bool valueChanged = ImGui::SliderFloat(("##" + std::string(label)).c_str(), value, min, max, emptyFormatCallback());

		ImGui::PopStyleColor(); // Restore the slider grab color

		// Get the slider's rectangle
		ImVec2       pos  = ImGui::GetItemRectMin();
		const ImVec2 size = ImGui::GetItemRectSize();

		// Draw a rectangle on the left part of the slider, filled with the desired color
		ImDrawList* draw_list = ImGui::GetWindowDrawList();
		const float x         = pos.x + size.x * (*value - min) / (max - min);
		const ImU32 color     = ImGui::GetColorU32(ImGuiCol_SliderGrabActive);                // Change to the ImGui color variable you want
		draw_list->AddRectFilled(pos, ImVec2(x, pos.y + size.y), color, style.FrameRounding); // Add rounding to match the ImGui style

		// Display the text and the value over the slider
		ImGui::SetCursorScreenPos({pos.x + style.ItemInnerSpacing.x, pos.y}); // Add ImGui style padding
		ImGui::TextUnformatted(label);

		ImGui::SetCursorScreenPos({pos.x + size.x - ImGui::CalcTextSize(std::to_string(*value).c_str()).x - 5, pos.y});
		ImGui::Text(format, *value);

		return valueChanged;
	}

	void fakeButton(const char* label, const ImVec2& size_arg = ImVec2(0, 0))
	{
		const ImVec4 regularColor = ImGui::GetStyleColorVec4(ImGuiCol_Button);

		// Set the hovered and active colors to the regular color
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, regularColor);
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, regularColor);

		ImGui::Button(label, size_arg);

		ImGui::PopStyleColor(2);
	}

	bool clippedText(const char* label)
	{
		const float totalItemWidth = ImGui::CalcItemWidth();

		std::string labelStr = label;
		// Adjust the label to fit the halfItemWidth
		float       labelWidth    = ImGui::CalcTextSize(labelStr.c_str()).x;
		float       spaceWidth    = ImGui::CalcTextSize(" ").x;
		const float ellipsisWidth = ImGui::CalcTextSize("...").x;

		std::vector<float> positionalWidth = {};
		float              width           = 0;
		for (const auto character : labelStr)
		{
			width += ImGui::CalcTextSize(std::string(1, character).c_str()).x;
			positionalWidth.push_back(width);
		}

		std::string newLabel;
		if (!positionalWidth.empty() && positionalWidth.back() > totalItemWidth)
		{
			size_t currentPosition = positionalWidth.size() - 1;
			while (currentPosition > 0 && positionalWidth[currentPosition] + ellipsisWidth > totalItemWidth)
			{
				currentPosition--;
			}

			newLabel = labelStr.substr(0, currentPosition) + "...";
		}

		if (!newLabel.empty())
		{
			labelStr   = newLabel;
			labelWidth = ImGui::CalcTextSize(labelStr.c_str()).x;
		}

		while (labelWidth < totalItemWidth && !labelStr.empty())
		{
			// If the text is too short, append spaces to it
			labelStr += ' ';
			labelWidth = ImGui::CalcTextSize(labelStr.c_str()).x;
		}

		// Draw the label
		ImGui::TextUnformatted(labelStr.c_str());

		return false;
	}

	bool colorPicker(const char* label, float* col)
	{
		const std::string labelStr       = label;
		bool              valueChanged   = false;
		const float       totalItemWidth = ImGui::CalcItemWidth();

		const ImVec4 color(col[0], col[1], col[2], 1.0f);
		ImGui::PushStyleColor(ImGuiCol_Button, color);                                                                            // Set the color for the button
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color);                                                                     // Set the hover color for the button
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, color);                                                                      // Set the active color for the button
		const bool clicked = ImGui::Button((labelStr + "_ColorButton").c_str(), ImVec2(totalItemWidth, ImGui::GetFrameHeight())); // Draw the button in the second column
		ImGui::PopStyleColor(3);                                                                                                  // Restore the color

		// Create a color picker when the button is clicked
		if (clicked)
		{
			ImGui::OpenPopup((labelStr + "_ColorPicker").c_str());
		}

		if (ImGui::BeginPopup((labelStr + "_ColorPicker").c_str()))
		{
			valueChanged = ImGui::ColorPicker3((labelStr + "_ColorPicker").c_str(), col);
			ImGui::EndPopup();
		}

		return valueChanged;
	}

	bool colorEdit3NoInputs(const char* label, float* col)
	{
		bool              valueChanged = false;
		const ImGuiStyle& style        = ImGui::GetStyle();

		// Start a new ImGui ID Scope
		ImGui::PushID(label);

		// Calculate the size of the text and button
		const float totalItemWidth = ImGui::CalcItemWidth();
		const float halfItemWidth  = totalItemWidth * 0.5f; // Subtract the ItemInnerSpacing

		ImGui::PushItemWidth(halfItemWidth - style.ItemSpacing.x); // Set the width of the next widget to 200

		clippedText(label); // Draw the label in the first column

		ImGui::PopItemWidth(); // Restore the width

		// After rendering the clipped text
		const ImVec2 textSize       = ImGui::GetItemRectSize();
		const float  remainingWidth = totalItemWidth - textSize.x - style.ItemSpacing.x;

		ImGui::SameLine();                    // Draw the button on the same line as the label
		ImGui::PushItemWidth(remainingWidth); // Set the width of the next widget to 200
		valueChanged = colorPicker(("##" + std::string(label)).c_str(), col);
		ImGui::PopItemWidth(); // Restore the width

		// End ImGui ID Scope
		ImGui::PopID();


		return valueChanged;
	}

	bool booleanText(const char* fmt, ...)
	{
		va_list args;
		va_start(args, fmt);
		ImGui::TextV(fmt, args);
		va_end(args);
		return false;
	}


	void affineGui(math::affine3f& affine)
	{
		const float availableWidth = ImGui::CalcItemWidth() - ImGui::GetStyle().ItemSpacing.x * 3;
		float       elementWidth   = availableWidth / 4.0f; //
		ImGui::BeginGroup();
		ImGui::PushItemWidth(elementWidth);
		for (int row = 0; row < 3; ++row)
		{
			for (int col = 0; col < 4; ++col)
			{
				fakeButton(std::to_string(affine[row][col]).c_str(), {elementWidth, 0});
				if (col < 3)
				{
					ImGui::SameLine();
				}
			}
		}
		ImGui::PopItemWidth();
		ImGui::EndGroup();
	}

	bool customVectorInput(const char* label, float& value, const ImU32 labelColor, const bool disableEdit)
	{
		const float availableWidth = ImGui::CalcItemWidth();

		// Start grouping to treat both the label and the input as a single item
		ImGui::BeginGroup();

		ImVec2 p        = ImGui::GetCursorScreenPos();
		float  rounding = ImGui::GetStyle().FrameRounding;

		// Rounded rectangle for the label
		ImGui::PushStyleColor(ImGuiCol_Button, labelColor);
		ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
		fakeButton(label);
		const float buttonSize = ImGui::GetItemRectSize().x;
		ImGui::PopStyleColor();
		ImGui::PopStyleColor();

		ImGui::SameLine();

		float remaininWidth = availableWidth - buttonSize - ImGui::GetStyle().ItemSpacing.x;
		ImGui::PushItemWidth(remaininWidth);
		// Float input for the value
		bool updated = false;
		if (!disableEdit)
		{
			updated = ImGui::InputFloat(("##" + std::string(label)).c_str(), &value, 0.0f, 0.0f, "%.3f");
		}
		else
		{
			fakeButton((std::to_string(value)).c_str(), {remaininWidth, 0});
		}
		ImGui::PopItemWidth();

		// End the grouping
		ImGui::EndGroup();

		return updated;
	}

	bool vectorGui(float* data, const bool disableEdit)
	{
		bool updated = false;

		const float availableWidth = ImGui::CalcItemWidth() - ImGui::GetStyle().ItemSpacing.x * 2;

		const float perComponentWidth = availableWidth / 3.0f;

		ImGui::PushItemWidth(perComponentWidth);

		// X Component in red
		updated |= customVectorInput("X :", data[0], IM_COL32(200, 0, 0, 255), disableEdit);
		ImGui::SameLine();

		// Y Component in green
		updated |= customVectorInput("Y :", data[1], IM_COL32(0, 200, 20, 255), disableEdit);
		ImGui::SameLine();

		// Z Component in blue
		updated |= customVectorInput("Z :", data[2], IM_COL32(0, 0, 200, 255), disableEdit);

		ImGui::PopItemWidth();
		return updated;
	}

	static float             halfWidgetFraction      = 0.3f;
	static std::stack<float> halfWidgetFractionStack = std::stack<float>();

	void pushHalfSpaceWidgetFraction(const float fraction)
	{
		halfWidgetFractionStack.push(halfWidgetFraction);
	}

	void popHalfSpaceWidgetFraction()
	{
		if (!halfWidgetFractionStack.empty())
		{
			halfWidgetFractionStack.pop();
		}
	}

	float getHalfWidgetFraction()
	{
		if (halfWidgetFractionStack.empty())
		{
			halfWidgetFractionStack.push(halfWidgetFraction);
		}
		return halfWidgetFractionStack.top();
	}

	bool halfSpaceDragInt(const char* label, int* v, float v_speed, int v_min, int v_max, const char* format, ImGuiSliderFlags flags)
	{
		return vtxImGui::halfSpaceWidget(label, ImGui::DragInt, (hiddenLabel + label).c_str(), v, v_speed, v_min, v_max, format, flags);
	}

	bool halfSpaceDragFloat(const char* label, float* v, float v_speed, float v_min, float v_max, const char* format, ImGuiSliderFlags flags)
	{
		return vtxImGui::halfSpaceWidget(label, ImGui::DragFloat, (hiddenLabel + label).c_str(), v, v_speed, v_min, v_max, format, flags);
	}

	bool halfSpaceIntCombo(const char* label, int& intVariable, const std::vector<int>& options)
	{
		const int                nOptions = options.size();
		std::vector<std::string> stringOptions(nOptions);
		std::vector<const char*> charOptions(nOptions);
		int                      optionIndex = 0;
		for (int i = 0; i < nOptions; ++i)
		{
			const int opt    = options[i];
			stringOptions[i] = std::to_string(opt);
			charOptions[i]   = stringOptions[i].c_str();
			if (intVariable == opt)
			{
				optionIndex = i;
			}
		}

		if (vtxImGui::halfSpaceWidget(label, (ComboFuncType)ImGui::Combo, (hiddenLabel + label).c_str(), &optionIndex, charOptions.data(), nOptions, -1))
		{
			intVariable = std::stoi(stringOptions[optionIndex]);
			return true;
		}
		return false;
	}

	bool halfSpaceStringCombo(const char* label, std::string& stringVariable, const std::vector<std::string>& options)
	{
		const int                nOptions = options.size();
		std::vector<const char*> charOptions(nOptions);
		int                      optionIndex = 0;
		for (int i = 0; i < nOptions; ++i)
		{
			charOptions[i] = options[i].c_str();
			if (stringVariable == options[i])
			{
				optionIndex = i;
			}
		}
		if (vtxImGui::halfSpaceWidget(label, (ComboFuncType)ImGui::Combo, (hiddenLabel + label).c_str(), &optionIndex, charOptions.data(), nOptions, -1))
		{
			stringVariable = options[optionIndex];
			return true;
		}
		return false;
	}

	bool halfSpaceCheckbox(const char* label, bool* v)
	{
		return vtxImGui::halfSpaceWidget(label, ImGui::Checkbox, (hiddenLabel + label).c_str(), v);
	}

	void DrawRowsBackground(const int row_count, const float line_height, const float x1, const float x2, const float y_offset, const ImU32 col_even, const ImU32 col_odd)
	{
		ImDrawList* draw_list = ImGui::GetWindowDrawList();
		const float y0        = ImGui::GetCursorScreenPos().y + (float)(int)y_offset;

		int              row_display_start;
		int              row_display_end;
		ImGuiListClipper clipper{};
		clipper.Begin(row_count);
		while (clipper.Step())
		{
			for (int row_n = clipper.DisplayStart; row_n < clipper.DisplayEnd; row_n++)
			{
				const ImU32 col = (row_n & 1) ? col_odd : col_even;
				if ((col & IM_COL32_A_MASK) == 0)
					continue;
				const float y1 = y0 + (line_height * row_n);
				const float y2 = y1 + line_height;
				draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), col);
			}
		}
		/*ImGui::CalcListClipping(row_count, line_height, &row_display_start, &row_display_end);
		for (int row_n = row_display_start; row_n < row_display_end; row_n++)
		{
			const ImU32 col = (row_n & 1) ? col_odd : col_even;
			if ((col & IM_COL32_A_MASK) == 0)
				continue;
			const float y1 = y0 + (line_height * row_n);
			const float y2 = y1 + line_height;
			draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), col);
		}*/
	}

	void childWindowResizerButton(float& percentage, const float& resizerSize, const bool isHorizontalSplit)
	{
		const ImVec2 buttonSize = isHorizontalSplit ? ImVec2(resizerSize, ImGui::GetContentRegionAvail().y) : ImVec2(ImGui::GetContentRegionAvail().x, resizerSize);
		if (ImGui::Button("##resizer", buttonSize))
		{
			isHorizontalSplit ? ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW) : ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
		}
		if (ImGui::IsItemHovered())
		{
			isHorizontalSplit ? ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW) : ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
		}
		if (ImGui::IsItemActive() && ImGui::IsMouseDragging(0))
		{
			const float delta           = isHorizontalSplit ? ImGui::GetMouseDragDelta(0).x : ImGui::GetMouseDragDelta(0).y;
			const float deltaPercentage = isHorizontalSplit ? delta / ImGui::GetWindowWidth() : delta / ImGui::GetWindowHeight();
			percentage -= deltaPercentage;
			percentage = std::clamp(percentage, 0.0f, 1.0f);
			ImGui::ResetMouseDragDelta(0);
		}
	}

	void drawOrigin(const math::vec2f& screenPosition)
	{
		ImDrawList*  drawList = ImGui::GetWindowDrawList();
		const ImVec2 winPos   = ImGui::GetWindowPos(); // Top-left corner
		// Draw the border first
		drawList->AddCircle(ImVec2(winPos.x + screenPosition.x, winPos.y + screenPosition.y), 3, ImColor(0, 0, 0, 255), 12, 1.0f);
		// Draw the filled circle with opacity
		drawList->AddCircleFilled(ImVec2(winPos.x + screenPosition.x, winPos.y + screenPosition.y), 3, ImColor(255, 255 / 2, 0, 200));
	}

	void drawCornersAndCenter()
	{
		ImDrawList*  drawList = ImGui::GetWindowDrawList();
		const ImVec2 winPos   = ImGui::GetWindowPos();  // Top-left corner
		const ImVec2 winSize  = ImGui::GetWindowSize(); // Window dimensions

		// Define the positions for the 5 circles
		const ImVec2 topLeft     = winPos;
		const ImVec2 topRight    = ImVec2(winPos.x + winSize.x, winPos.y);
		const ImVec2 bottomLeft  = ImVec2(winPos.x, winPos.y + winSize.y);
		const ImVec2 bottomRight = ImVec2(winPos.x + winSize.x, winPos.y + winSize.y);
		const ImVec2 center      = ImVec2(winPos.x + winSize.x * 0.5f, winPos.y + winSize.y * 0.5f);

		// Draw circles at these positions
		drawList->AddCircle(topRight, 5, ImColor(255, 0, 0, 255), 12, 1.0f);
		drawList->AddCircle(bottomLeft, 5, ImColor(0, 255, 0, 255), 12, 1.0f);
		drawList->AddCircle(bottomRight, 5, ImColor(0, 0, 255, 255), 12, 1.0f);
		drawList->AddCircle(topLeft, 5, ImColor(255, 255, 0, 255), 12, 1.0f);
		drawList->AddCircle(center, 5, ImColor(255, 255, 255, 255), 12, 1.0f);
	}

	void drawDashedLine(const math::vec2f& point1, const math::vec2f& point2, const float dashLength, const float gapLength)
	{
		ImDrawList*  drawList = ImGui::GetWindowDrawList();
		const ImVec2 winPos   = ImGui::GetWindowPos(); // Top-left corner of the window

		// Convert local window-based coordinates to global screen-based coordinates
		const ImVec2 globalPoint1(point1.x + winPos.x, point1.y + winPos.y);
		const ImVec2 globalPoint2(point2.x + winPos.x, point2.y + winPos.y);

		const ImVec2 delta       = ImVec2(globalPoint2.x - globalPoint1.x, globalPoint2.y - globalPoint1.y);
		const float  fullLength  = sqrtf(delta.x * delta.x + delta.y * delta.y);
		const int    numSegments = static_cast<int>(fullLength / (dashLength + gapLength));

		for (int i = 0; i < numSegments; ++i)
		{
			const float t0 = i * (dashLength + gapLength) / fullLength;
			const float t1 = (i * (dashLength + gapLength) + dashLength) / fullLength;

			ImVec2 p0 = ImVec2(globalPoint1.x + t0 * delta.x, globalPoint1.y + t0 * delta.y);
			ImVec2 p1 = ImVec2(globalPoint1.x + t1 * delta.x, globalPoint1.y + t1 * delta.y);

			drawList->AddLine(p0, p1, ImColor(0, 0, 0, 200), 1.0f);
		}
	}

	void drawVector(const std::shared_ptr<graph::Camera>& camera, const math::vec3f& origin, const math::vec3f& direction, const math::vec3f& color)
	{
		// Project vector endpoints to screen space
		const math::vec2f screenOrigin   = camera->project(origin, true);
		const math::vec2f screenPosition = camera->project(origin + direction, true);

		//drawOrigin(screenOrigin);
		//drawOrigin(screenPosition);

		// Draw the vector in ImGui
		ImDrawList*  drawList = ImGui::GetWindowDrawList();
		const ImVec2 winPos   = ImGui::GetWindowPos(); // Top-left corner
		const ImVec2 start    = ImVec2(winPos.x + screenOrigin.x, winPos.y + screenOrigin.y);
		const ImVec2 end      = ImVec2(winPos.x + screenPosition.x, winPos.y + screenPosition.y);
		drawList->AddLine(
						  start,
						  end,
						  ImColor((int)(255 * color.x), (int)(255 * color.y), (int)(255 * color.z), 255),
						  1.0f
						 );
	}

	void connectScenePoints(const std::shared_ptr<graph::Camera>& camera, const math::vec3f& point1, const math::vec3f& point2, const math::vec3f& color)
	{
		// Project vector endpoints to screen space
		const math::vec2f screenOrigin   = camera->project(point1, true);
		const math::vec2f screenPosition = camera->project(point2, true);

		//drawOrigin(screenOrigin);
		//drawOrigin(screenPosition);

		// Draw the vector in ImGui
		ImDrawList*  drawList = ImGui::GetWindowDrawList();
		const ImVec2 winPos   = ImGui::GetWindowPos(); // Top-left corner
		const ImVec2 start    = ImVec2(winPos.x + screenOrigin.x, winPos.y + screenOrigin.y);
		const ImVec2 end      = ImVec2(winPos.x + screenPosition.x, winPos.y + screenPosition.y);
		drawList->AddLine(
						  start,
						  end,
						  ImColor((int)(255 * color.x), (int)(255 * color.y), (int)(255 * color.z), 255),
						  2.0f
						 );
	}
}
