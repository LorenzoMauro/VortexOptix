#include "Mixture.h"
#include "SphericalGaussian.h"
#include "Core/Log.h"
#include "NeuralNetworks/tools.h"

namespace vtx::distribution
{
    torch::Tensor Mixture::finalizeParams(torch::Tensor& mixtureParameters, const network::config::DistributionType& type)
    {
        if (type == network::config::DistributionType::D_SPHERICAL_GAUSSIAN)
        {
            return SphericalGaussian::finalizeRawParams(mixtureParameters);
        }
        else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE)
        {
            return Nasg::finalizeRawParams(mixtureParameters, type);
        }
        else
        {
            VTX_ERROR("Mixture::finalizeParams not implemented for this distribution type");
        }

        return mixtureParameters;
    }

    torch::Tensor Mixture::prob(const torch::Tensor& x, const torch::Tensor& mixtureParams, const torch::Tensor& mixtureWeights, const network::config::DistributionType type)
    {
        // expand x if not 3D to match the Batch Size x Mixture Size x x last dim

		const torch::Tensor expandedX = x.unsqueeze(1).expand({-1, mixtureParams.size(1), -1});
        TRACE_TENSOR(expandedX);
        torch::Tensor prob;
        if (type == network::config::DistributionType::D_SPHERICAL_GAUSSIAN)
        {
            prob = SphericalGaussian::prob(expandedX, mixtureParams);
        }
        else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE)
        {
	        prob = Nasg::prob(expandedX, mixtureParams);
		}
        else
        {
            VTX_ERROR("Mixture::prob not implemented for this distribution type");
        }
        TRACE_TENSOR(prob);
        // Weighing the probability of each mixture
        // mixtureWeights is in the shape Batch Size x Mixture Size
        torch::Tensor weightedP = prob * mixtureWeights.unsqueeze(-1);
		weightedP               = weightedP.sum(1);
        TRACE_TENSOR(weightedP);
        return weightedP;
    }

    std::tuple<torch::Tensor, torch::Tensor> Mixture::sample(const torch::Tensor& mixtureParams, const torch::Tensor& mixtureWeights, const network::config::DistributionType type)
    {
        // Select mixture from mixtureWeights tensor
        const torch::Tensor selectedMixture = multinomial(mixtureWeights, 1);

        // Expand the selectedMixture for compatibility with reshapedMixtureParameters
		int                 distributionParamsCount = mixtureParams.size(mixtureParams.dim() - 1);
		const torch::Tensor expandedSelectedMixture = selectedMixture.unsqueeze(-1).expand({-1, -1, distributionParamsCount});

        // Gather the selected parameters
        const torch::Tensor selectedParameters = mixtureParams.gather(1, expandedSelectedMixture).squeeze(1);


        torch::Tensor sampleTensor;
        if (type == network::config::D_SPHERICAL_GAUSSIAN)
        {
            sampleTensor = SphericalGaussian::sample(selectedParameters);
        }
        else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE)
        {
	        sampleTensor = Nasg::sample(selectedParameters);
        }
        else
        {
            VTX_ERROR("Unknown distribution type");
            return {};
        }


        torch::Tensor prob = Mixture::prob(sampleTensor, mixtureParams, mixtureWeights, type);
        TRACE_TENSOR(sampleTensor);
        TRACE_TENSOR(prob);
		return {sampleTensor, prob};
    }

    void Mixture::setGraphData(const network::config::DistributionType type, const torch::Tensor& params, const torch::Tensor& mixtureWeights, network::GraphsData& graphData, const bool isTraining, const int depth)
    {
        if (type == network::config::DistributionType::D_SPHERICAL_GAUSSIAN)
        {
	        SphericalGaussian::setGraphData(params, mixtureWeights, graphData, isTraining, depth);
		}
		else if (type == network::config::D_NASG_TRIG || type == network::config::D_NASG_ANGLE || type == network::config::D_NASG_AXIS_ANGLE)
		{
			Nasg::setGraphData(params, mixtureWeights, graphData, isTraining, depth);
		}
		else
		{
			VTX_ERROR("Unknown distribution type");
		}
    }
}
