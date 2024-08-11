#pragma once
#include <string>
#include <vector>

namespace vtx::network::config
{
	struct NetworkSettings;

	std::vector<NetworkSettings> generateNetworkSettingNeighbors(NetworkSettings& original);

	NetworkSettings getBestGuess();

	NetworkSettings getSOTA();

	NetworkSettings getNasgSOTA();

	NetworkSettings getWhishfull();

	std::vector<NetworkSettings> ablationVariations(NetworkSettings& original);
	#define stringHash(x) #x << "|" << std::to_string(x) << "\n"

	std::string getNetworkSettingHash(const NetworkSettings& netSettings);
}
