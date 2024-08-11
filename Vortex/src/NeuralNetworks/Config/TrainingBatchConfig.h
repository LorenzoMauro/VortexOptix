#pragma once
#include <map>
#include <string>

namespace vtx::network::config
{
	enum SamplingStrategy
	{
		SS_ALL,
		SS_PATHS_WITH_CONTRIBUTION,
		SS_LIGHT_SAMPLES,

		SS_COUNT
	};

	inline static const char* samplingStrategyNames[] = {
		"All",
		"Paths with Contribution",
		"Light Samples"
	};

	inline static std::map<std::string, SamplingStrategy> samplingStrategyNameToEnum =
	{
		{"All", SS_ALL},
		{"Paths with Contribution", SS_PATHS_WITH_CONTRIBUTION},
		{"Light Samples", SS_LIGHT_SAMPLES}
	};

	struct BatchGenerationConfig
	{
		bool             limitToFirstBounce = true;
		bool			 onlyNonZero		= true;
		bool             weightByMis		= true;
		bool             weightByPdf		= true;
		bool             useLightSample 	= true;
		bool             trainOnLightSample = false;
		bool             skipSpecular 		= true;
		bool		     skipIncomplete		= true;
		bool             isUpdated			= true;
	};

}
