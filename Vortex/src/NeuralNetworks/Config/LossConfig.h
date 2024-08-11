#pragma once
#include <map>
#include <string>

namespace vtx::network::config
{
	enum LossType
	{
		L_KL_DIV,
		L_KL_DIV_MC_ESTIMATION,
		L_KL_DIV_MC_ESTIMATION_NORMALIZED,
		L_MSE,
		L_PEARSON_DIV_MC_ESTIMATION,
		L_COUNT
	};

	inline static const char* lossNames[] = {
		"KL Divergence",
		"KL Divergence MC Estimation",
		"KL Divercence MC Estimation Normalized",
		"Mean Square Error",
		"Pearson Divergence MC Estimation"
	};

	inline static std::map<std::string, LossType> lossNameToEnum =
	{
			{"KL Divergence", L_KL_DIV},
			{"KL Divergence MC Estimation", L_KL_DIV_MC_ESTIMATION},
			{"KL Divergence MC Estimation Normalized", L_KL_DIV_MC_ESTIMATION_NORMALIZED},
			{"Pearson Divergence", L_MSE},
			{"Pearson Divergence MC Estimation", L_PEARSON_DIV_MC_ESTIMATION}
	};

	enum LossReduction
	{
		SUM,
		MEAN,
		ABS,
		COUNT
	};

	inline static const char* lossReductionNames[] = {
		"Sum",
		"Mean",
		"Abs"
	};

	inline static std::map<std::string, LossReduction> lossReductionNameToEnum =
	{
			{"Sum", SUM},
			{"Mean", MEAN},
			{"Abs", ABS}
	};
}
