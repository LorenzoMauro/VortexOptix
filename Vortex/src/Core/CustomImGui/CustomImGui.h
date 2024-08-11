#pragma once
#include <string>
#include "imgui.h"
#include <type_traits>
#include <vector>

#include "Core/Math.h"

namespace vtx
{
	namespace graph
	{
		class Camera;
	}
}

static const std::string hiddenLabel = "##hidden";

typedef bool (*ComboFuncType)(const char*, int*, const char* const[], int, int);

namespace vtx::vtxImGui
{
    const char* emptyFormatCallback();

	bool sliderFloat(const char* label, float* value, const float min, const float max, const char* format = "%.3f");

    bool clippedText(const char* label);

    bool colorPicker(const char* label, float* col);

    bool colorEdit3NoInputs(const char* label, float* col);

    bool booleanText(const char* fmt, ...);

    void affineGui(math::affine3f& affine);

    bool vectorGui(float* data, bool disableEdit = false);

	void  pushHalfSpaceWidgetFraction(const float fraction);
	void  popHalfSpaceWidgetFraction();
    float getHalfWidgetFraction();

	template <typename Func, typename... Args>
	bool halfSpaceWidget(const char* label, Func&& widgetFunc, Args&&... args)
	{
        bool              valueChanged = false;
		const ImGuiStyle& style        = ImGui::GetStyle();

        // Start a new ImGui ID Scope
        ImGui::PushID(label);

        const float labelFraction = getHalfWidgetFraction();

        // Calculate the size of the text and button
        //const float totalItemWidth = ImGui::CalcItemWidth();
        const float totalItemWidth = ImGui::GetContentRegionAvail().x;

		const float labelWidth  = totalItemWidth * labelFraction;
        const float widgetWidth = totalItemWidth * (1.0f - labelFraction);

        ImGui::PushItemWidth(labelWidth - style.ItemSpacing.x); // Set the width of the next widget to 200

		const float cursorPosX = ImGui::GetCursorPosX();

        clippedText(label); // Draw the label in the first column

        ImGui::PopItemWidth(); // Restore the width

		ImGui::SameLine();                                       // Draw the button on the same line as the label
		ImGui::SetCursorPosX(cursorPosX + labelWidth);           // Position in window coordinates
        ImGui::PushItemWidth(widgetWidth - style.ItemSpacing.x); // Set the width of the next widget to 200
		if constexpr (std::is_same_v<bool, std::invoke_result_t<Func, Args...>>)
		{
            valueChanged = widgetFunc(std::forward<Args>(args)...);
        }
		else
		{
            widgetFunc(std::forward<Args>(args)...);
        }
        ImGui::PopItemWidth(); // Restore the width

        // End ImGui ID Scope
        ImGui::PopID();

        return valueChanged;
    }

    bool halfSpaceDragInt(const char* label, int* v, float v_speed = 1.0f, int v_min = 0, int v_max = 0, const char* format = "%d", ImGuiSliderFlags flags = 0);

    bool halfSpaceDragFloat(const char* label, float* v, float v_speed = 1.0f, float v_min = 0.0f, float v_max = 0.0f, const char* format = "%.3f", ImGuiSliderFlags flags = 0);

	template <typename Enum>
    bool halfSpaceCombo(const char* label, Enum& enumVariable, const char* const enumNames[], int enumCount)
    {
		int  intEnum   = (int)enumVariable;
        bool isUpdated = false;
        if (vtxImGui::halfSpaceWidget(label, (ComboFuncType)ImGui::Combo, (hiddenLabel + label).c_str(), &intEnum, enumNames, enumCount, -1))
        {
			isUpdated    = true;
            enumVariable = static_cast<Enum>(intEnum);
        }
        return isUpdated;
    }

    bool halfSpaceIntCombo(const char* label, int& intVariable, const std::vector<int>& options);

    bool halfSpaceStringCombo(const char* label, std::string& stringVariable, const std::vector<std::string>& options);

    bool halfSpaceCheckbox(const char* label, bool* v);


    void DrawRowsBackground(int row_count, float line_height, float x1, float x2, float y_offset, ImU32 col_even, ImU32 col_odd);

    void childWindowResizerButton(float& percentage, const float& resizerSize, bool isHorizontalSplit);

    void drawOrigin(const math::vec2f& screenPosition);
    void drawCornersAndCenter();

    void drawDashedLine(const math::vec2f& point1, const math::vec2f& point2, float dashLength = 5.0f, float gapLength = 4.0f);

    void drawVector(const std::shared_ptr<graph::Camera>& camera, const math::vec3f& origin, const math::vec3f& direction, const math::vec3f& color = 1.0f);

    void connectScenePoints(const std::shared_ptr<graph::Camera>& camera, const math::vec3f& point1, const math::vec3f& point2, const math::vec3f& color);
}
