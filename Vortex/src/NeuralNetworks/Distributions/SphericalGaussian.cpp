#include "SphericalGaussian.h"

#include "Core/Constants.h"
#include "NeuralNetworks/NeuralNetworkGraphs.h"
#include "NeuralNetworks/tools.h"

namespace vtx::distribution
{
    torch::Tensor SphericalGaussian::sample(const torch::Tensor& loc, const torch::Tensor& scale) {

        int64_t batchSize = loc.size(0);

        torch::Device       device = loc.device();
        const torch::Tensor uniform = torch::rand({ batchSize, 1 }, torch::TensorOptions().device(device));
        torch::Tensor       w = 1.0f + torch::log(uniform + (1.0f - uniform) * torch::exp(-2.0f * scale) + sEps) / (scale + sEps);

        torch::Tensor angleUniform = torch::rand({ batchSize, 1 }, torch::TensorOptions().device(device));
        angleUniform = angleUniform * 2.0f * M_PI_F;
        const torch::Tensor v = torch::cat({ torch::cos(angleUniform), torch::sin(angleUniform) }, -1);

        const torch::Tensor w_ = torch::sqrt(clamp(1 - w.pow(2), sEps, 1));

        const torch::Tensor x = torch::cat({ w, w_ * v }, -1);

        // Householder rotation
        const torch::Tensor e1 = torch::tensor({ 1.0, 0.0, 0.0 }, loc.options()).expand_as(loc);
        auto       u = e1 - loc;
        u = u / (u.norm(2, 1, true) + sEps);
        auto sample = x - 2 * (x * u).sum(-1, true) * u;

        TRACE_TENSOR(loc);
        TRACE_TENSOR(scale);
        TRACE_TENSOR(uniform);
        TRACE_TENSOR(w);
        TRACE_TENSOR(w_);
        TRACE_TENSOR(v);
        TRACE_TENSOR(x);
        TRACE_TENSOR(e1);
        TRACE_TENSOR(u);
        TRACE_TENSOR(sample);;

        return sample;
    }

    torch::Tensor SphericalGaussian::prob(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale)
    {
        const torch::Tensor a = (2.0f * M_PI_F * (1.0f - torch::exp(-2.0f * scale)));
        const torch::Tensor b = torch::exp(scale * ((loc * x).sum(-1).unsqueeze(-1) - 1.0f))toPrecision;
        torch::Tensor p = scale / a * b;
        TRACE_TENSOR(a);
        TRACE_TENSOR(b);
        TRACE_TENSOR(p);
        return p;
    }

    torch::Tensor SphericalGaussian::logLikelihood(const torch::Tensor& x, const torch::Tensor& loc, const torch::Tensor& scale) {

        torch::Tensor logP = torch::log(prob(x, loc, scale) + sEps);
        TRACE_TENSOR(x);
        TRACE_TENSOR(loc);
        TRACE_TENSOR(scale);
        TRACE_TENSOR(logP);
        return logP;
    }

    std::tuple<torch::Tensor, torch::Tensor> SphericalGaussian::splitParams(const torch::Tensor& params)
    {
        torch::Tensor loc = params.narrow(params.dim() - 1, 0, 3);
        torch::Tensor scale = params.narrow(params.dim() - 1, 3, 1);
        return { loc, scale };
    }

    torch::Tensor SphericalGaussian::prob(const torch::Tensor& x, const torch::Tensor& params)
    {
        auto [loc, scale] = splitParams(params);
        TRACE_TENSOR(loc);
        TRACE_TENSOR(scale);
        return prob(x, loc, scale);
    }

    torch::Tensor SphericalGaussian::logLikelihood(const torch::Tensor& x, const torch::Tensor& params)
    {
        auto [loc, scale] = splitParams(params);
        return logLikelihood(x, loc, scale);
    }

    torch::Tensor SphericalGaussian::sample(const torch::Tensor& params)
    {
        auto [loc, scale] = splitParams(params);
        return sample(loc, scale);
    }

    torch::Tensor SphericalGaussian::finalizeRawParams(const torch::Tensor& params)
    {
        // The params comes from the network output but are not yet "activated" properly
        // The dimension is Batch Size x Distribution Parameters Count
        // Extracting means and k for all mixtures at once
        torch::Tensor theta = params.narrow(params.dim() - 1, 0, 1);
        torch::Tensor phi = params.narrow(params.dim() - 1, 1, 1);
        torch::Tensor scale = params.narrow(params.dim() - 1, 2, 1);

        phi = sigmoid(phi)* M_PI;
        theta = sigmoid(theta)* 2*M_PI;
        torch::Tensor loc = torch::cat({ torch::sin(phi) * torch::cos(theta), torch::sin(phi) * torch::sin(theta), torch::cos(phi) }, -1);
        TRACE_TENSOR(loc);
        TRACE_TENSOR(scale);
        loc = loc + sEps;
        torch::Tensor locNorm = linalg_vector_norm(loc, 2, -1, true);
        loc                 = loc / (locNorm);
        scale               = softplus(scale) + sEps;
        TRACE_TENSOR(loc);
        TRACE_TENSOR(scale);
        // Stacking them along the last dimension
        torch::Tensor elaboratedParams = torch::cat({ loc, scale }, params.dim() - 1);
        return elaboratedParams;
    }

    void SphericalGaussian::setGraphData(
        const torch::Tensor& params,
        const torch::Tensor& mixtureWeights,
        network::GraphsData& graphData,
        const bool isTraining,
        const int depth = 0
    )
    {
        auto [_, scale] = splitParams(params);
        const torch::Tensor scaleWeightedMean = (scale * mixtureWeights.unsqueeze(-1)).sum(1).mean().unsqueeze(-1);
        const std::vector<torch::Tensor> hostTensors = network::downloadTensors(
            {
                scaleWeightedMean
            }
        );
        if(isTraining)
        {
            graphData.addData("Concentration", "Iteration", "Concentration", "Concentration Train", hostTensors[0].item<float>());
        }
        else
        {
            graphData.addData("Concentration", "Iteration", "Concentration", "Concentration Inference", hostTensors[0].item<float>(), depth);
        }
    }

}