#include "TcnnTorchModule.h"
#include "TcnnAutoGrad.h"
#include "tcnnUtils.h"
#include "NeuralNetworks/Config/NetworkSettings.h"

namespace torchTcnn
{
    TcnnModuleImpl::TcnnModuleImpl(int _inputDim, int _outputDim, const nlohmann::json& encodingConfig, const nlohmann::json& networkConfig, int _seed) :
        inputDims(_inputDim),
        outputDims(_outputDim),
        seed(_seed),
        nativeTcnnModule(createNetworkWithInputEncoding(inputDims, outputDims, encodingConfig, networkConfig))
    {
        dType = torchPrecision(nativeTcnnModule.paramPrecision());
        seed = _seed;
        params = register_parameter("params", nativeTcnnModule.initialParams(seed), true);
        loss_scale = default_loss_scale(nativeTcnnModule.paramPrecision());
		//loss_scale = 65536.0f;
		//loss_scale = 1.0f;
    }

    torch::Tensor TcnnModuleImpl::forward(torch::Tensor x) {
        if (!x.is_cuda()) {
            std::cerr << "Warning: input must be a CUDA tensor, but isn't. This indicates suboptimal performance." << std::endl;
            x = x.to(torch::kCUDA);
        }

        int64_t         batchSize = x.size(0);
        const int       batchSizeGranularity = int(tcnn::cpp::batch_size_granularity());
        const long long paddedBatchSize = (batchSize + batchSizeGranularity - 1) / batchSizeGranularity *
            batchSizeGranularity;

        const torch::Tensor xPadded = (batchSize == paddedBatchSize) ? x : torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({ 0, 0, 0, paddedBatchSize - batchSize }));
        torch::Tensor output = ModuleFunction::apply(
            nativeTcnnModule,
            xPadded.to(torch::kFloat).contiguous(),
            params.to(torchPrecision(nativeTcnnModule.paramPrecision())).contiguous(),
            loss_scale
        );
        output = output.slice(0, 0, batchSize).slice(1, 0, outputDims);
		//output = overflowLayer->forward(output);
        return output;
    }


	nlohmann::json getEncodingSettings(const vtx::network::config::EncodingConfig& inputSettings) {
		using namespace vtx::network::config;
		nlohmann::json config;
		config["otype"] = EncodingTypeName[static_cast<int>(inputSettings.otype)];

		switch (inputSettings.otype) {
		case EncodingType::Composite:
			// Assuming you have logic to handle Composite type
			break;
		case EncodingType::Frequency:
			config["n_frequencies"] = inputSettings.frequencyEncoding.n_frequencies;
			break;
		case EncodingType::Grid:
			config["type"] = GridTypeName[static_cast<int>(inputSettings.gridEncoding.type)]; // Assuming GridTypeName is similar to config::EncodingTypeName
			config["n_levels"]             = inputSettings.gridEncoding.n_levels;
			config["n_features_per_level"] = inputSettings.gridEncoding.n_features_per_level;
			config["log2_hashmap_size"]    = inputSettings.gridEncoding.log2_hashmap_size;
			config["base_resolution"]      = inputSettings.gridEncoding.base_resolution;
			config["per_level_scale"]      = inputSettings.gridEncoding.per_level_scale;
			config["interpolation"]        = InterpolationTypeName[static_cast<int>(inputSettings.gridEncoding.interpolation)]; // Assuming a similar mapping for InterpolationType
			break;
		case EncodingType::Identity:
			config["scale"] = inputSettings.identityEncoding.scale;
			config["offset"] = inputSettings.identityEncoding.offset;
			break;
		case EncodingType::OneBlob:
			config["n_bins"] = inputSettings.oneBlobEncoding.n_bins;
			break;
		case EncodingType::SphericalHarmonics:
			config["degree"] = inputSettings.sphericalHarmonicsEncoding.degree;
			break;
		case EncodingType::TriangleWave:
			config["n_frequencies"] = inputSettings.triangleWaveEncoding.n_frequencies;
			break;
		default:
			throw std::runtime_error("Unsupported encoding type");
		}

		return config;
	}

	nlohmann::json getEncodingSettings(const vtx::network::config::MainNetEncodingConfig& inputSettings)
	{
		nlohmann::json config;
		config["otype"] = "Composite";
		config["nested"][0] = getEncodingSettings(inputSettings.position);
		config["nested"][0]["n_dims_to_encode"] = 3;
		config["nested"][1] = getEncodingSettings(inputSettings.normal);
		config["nested"][1]["n_dims_to_encode"] = 3;
		config["nested"][2] = getEncodingSettings(inputSettings.wo);
		config["nested"][2]["n_dims_to_encode"] = 3;
		return config;
	}


	nlohmann::json getEncodingSettings(int rawFeatureSize, const vtx::network::config::AuxNetEncodingConfig& inputSettings)
	{
		nlohmann::json config;
		config["otype"] = "Composite";
		config["nested"][0] = {};
		config["nested"][0]["otype"] = "Identity";
		config["nested"][0]["n_dims_to_encode"] = rawFeatureSize;
		config["nested"][1] = getEncodingSettings(inputSettings.wi);
		config["nested"][1]["n_dims_to_encode"] = 3;
		return config;
	}


	nlohmann::json getNetworkSettings(const vtx::network::config::MlpSettings& settings)
	{
		nlohmann::json config;
		config["otype"] = "FullyFusedMLP";
		config["activation"] = "ReLU";
		config["output_activation"] = "None";
		config["n_neurons"] = settings.hiddenDim;
		config["n_hidden_layers"] = settings.numHiddenLayers;
		return config;
	}
}
