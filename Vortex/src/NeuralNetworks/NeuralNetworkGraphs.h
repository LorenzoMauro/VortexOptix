#ifndef NETWORK_GRAPHS_H
#define NETWORK_GRAPHS_H
#pragma once
#include <map>
#include <stdexcept>
#include <string>
#include <vector>


namespace vtx::network
{
#define LOSS_PLOT "Loss","Iteration","Loss"
#define PROB_PLOT "Target Prob","Iteration","Prob"
#define SAMPLING_PROB_PLOT "Sampling Prob","Iteration","Prob"
#define SAMPLING_FRACTION_PLOT "Sampling Fraction","Iteration","Sampling Fraction"



	struct PlotData
	{
		std::string name;
		std::string xLabel;
		std::string yLabel;
		std::map<std::string, std::vector<float>> data;
	};
	struct GraphsData
	{
		std::map<std::string, PlotData> graphs;

		void addData(const std::string& plotName, const std::string& xLabel, const std::string& yLabel, const std::string& curveName, const float& data, const int& depth=0)
		{
			if(isnan(data) || isinf(data))
			{
				std::string type = isnan(data) ? "NaN" : "Infinity";
				throw std::runtime_error("Detected " + type + "  while adding data to graph " + curveName + ", Network has probably crashed!");
			}
			if(graphs.count(plotName) == 0)
			{
				graphs[plotName] = {plotName, xLabel, yLabel, {}};
			}
			if(depth == 0)
			{
				graphs[plotName].data[curveName].push_back(data);
			}
			else
			{
				const float prevValue = graphs[plotName].data[curveName].back();
				const float newMeanValue = (prevValue * ((float)depth - 1.0f) + (float)data) / (float)depth;
				// update value
				graphs[plotName].data[curveName].back() = newMeanValue;
			}
		}
		void reset()
		{
			graphs.clear();
		}
	};
}

#endif