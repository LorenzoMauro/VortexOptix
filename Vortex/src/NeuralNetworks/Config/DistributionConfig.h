#pragma once
#include <map>
#include <string>

namespace vtx::network::config
{
	enum DistributionType
	{
		D_SPHERICAL_GAUSSIAN,
		D_NASG_TRIG,
		D_NASG_ANGLE,
		D_NASG_AXIS_ANGLE,
		D_NASG_TRIG_NORMALIZED,

		D_COUNT
	};

	inline static const char* distributionNames[] = {
		"Spherical Gaussian",
		"NASG Trigonometric",
		"NASG Angle",
		"NASG Axis Angle",
		"NASG Trigonometric Normalized"
	};

	inline static std::map<std::string, DistributionType> distributionNameToEnum =
	{
		{"Spherical Gaussian", D_SPHERICAL_GAUSSIAN},
		{"NASG Trigonometric", D_NASG_TRIG},
		{"NASG Angle", D_NASG_ANGLE},
		{"NASG Axis Angle", D_NASG_AXIS_ANGLE},
		{"NASG Trigonometric Normalized", D_NASG_TRIG_NORMALIZED}
	};
}
