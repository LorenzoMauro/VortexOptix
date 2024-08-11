#include "TcnnAutoGrad.h"
#include<unordered_set>

constexpr const char* TCNN_DATA = "TCNN_DATA";
constexpr const char* LOSS_SCALE = "loss_scale";

namespace torchTcnn
{
	void isTensorValid(torch::Tensor tensor)
	{
		if(!tensor.defined())
		{
			printf("Tensor is not defined\n");
			__debugbreak();
		}
		bool hasNan = tensor.isnan().any().item<bool>();
		bool hasInf = tensor.isinf().any().item<bool>();
		if (hasNan || hasInf)
		{
			printf("Tensor has nan or inf\n");
			__debugbreak();
		}
	}

	torch::Tensor nullTensorToNone(const torch::Tensor& tensor)
	{
		if (tensor.sizes().size() == 0)
		{
			return torch::Tensor();
		}
		return tensor;
	}

	torch::Tensor nullTensorLike(const torch::Tensor& tensor)
	{
		return torch::empty_like(tensor).to(tensor.device());
	}

	class TcnnData : public torch::CustomClassHolder
	{
	public:
		NativeModule* nativeTcnnModule = nullptr;
		tcnn::cpp::Context nativeCtx = tcnn::cpp::Context{ nullptr };
		float              lossScale;
	};
	static auto registerTcnnData = torch::class_<TcnnData>("torchTcnn", TCNN_DATA);

	torch::autograd::variable_list ModuleFunctionBackward::forward(
		torch::autograd::AutogradContext* ctx,
		torch::autograd::AutogradContext* ctxFwd,
		const torch::Tensor& dOutput,
		const torch::Tensor& input,
		const torch::Tensor& params,
		const torch::Tensor& output
	)
	{
		const c10::intrusive_ptr<TcnnData> tcnnData = ctxFwd->saved_data[TCNN_DATA].toCustomClass<TcnnData>();

		ctx->save_for_backward({ input, params, dOutput });
		ctx->saved_data[TCNN_DATA] = tcnnData;
		torch::autograd::variable_list outputList;

		{
			torch::NoGradGuard no_grad_guard; // Equivalent to `with torch.no_grad()`

			const at::Tensor scaled_grad = dOutput * tcnnData->lossScale;
			auto [inputGrad, paramsGrad] = tcnnData->nativeTcnnModule->bwd(tcnnData->nativeCtx, input, params, output, scaled_grad);

			paramsGrad = (paramsGrad.defined()) ? paramsGrad / tcnnData->lossScale : nullTensorLike(params);
			inputGrad = (inputGrad.defined()) ? inputGrad / tcnnData->lossScale : nullTensorLike(input);
			outputList = { nullTensorToNone(inputGrad), nullTensorToNone(paramsGrad) };
		}

		return outputList;
	}

	torch::autograd::variable_list ModuleFunctionBackward::backward(torch::autograd::AutogradContext* ctx,
		const torch::autograd::tensor_list& gradOutputs)
	{
		torch::autograd::variable_list saved   = ctx->get_saved_variables();
		at::Tensor&                     input   = saved[0];
		at::Tensor&                     params  = saved[1];
		at::Tensor                           dOutput = saved[2];

		const c10::intrusive_ptr<TcnnData> tcnnData = ctx->saved_data[TCNN_DATA].toCustomClass<TcnnData>();
		{
			torch::AutoGradMode enable_grad(true); // Equivalent to `with torch.enable_grad()`
			dOutput = dOutput * tcnnData->lossScale;
		}

		torch::autograd::variable_list outputList;
		{
			torch::NoGradGuard noGradGuard; // Back to `with torch.no_grad()`

			auto [dOutput_grad, params_grad, input_grad] = tcnnData->nativeTcnnModule->bwdBwdInput(
				tcnnData->nativeCtx,
				input,
				params,
				gradOutputs[0], dOutput);

			params_grad = (params_grad.defined()) ? params_grad / tcnnData->lossScale : torch::Tensor();
			input_grad = (input_grad.defined()) ? input_grad / tcnnData->lossScale : torch::Tensor();
			outputList = { torch::Tensor(), dOutput_grad, input_grad, params_grad, torch::Tensor() };
		}

		return outputList;
	}

	torch::Tensor ModuleFunction::forward(
		torch::autograd::AutogradContext* ctx,
		torchTcnn::NativeModule& native_tcnn_module,
		torch::Tensor                     input,
		torch::Tensor          params,
		float loss_scale)
	{
		// Set to not automatically materialize output gradients
		ctx->set_materialize_grads(false);
		auto [native_ctx, output] = native_tcnn_module.fwd(input, params);
		ctx->save_for_backward({ input, params, output });
		const c10::intrusive_ptr<TcnnData> tcnnDataPtr = c10::make_intrusive<TcnnData>();
		tcnnDataPtr->nativeTcnnModule = &native_tcnn_module;
		tcnnDataPtr->nativeCtx = std::move(native_ctx);
		tcnnDataPtr->lossScale = loss_scale;
		ctx->saved_data[TCNN_DATA] = tcnnDataPtr;
		return output;
	}


	torch::autograd::tensor_list ModuleFunction::backward(torch::autograd::AutogradContext*   ctx,
														  const torch::autograd::tensor_list& grad_outputs)
	{
		at::Tensor dOutput = grad_outputs[0];
		// print dOutput type
		if (!dOutput.defined())
		{
			return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
		}

		if (!dOutput.is_cuda())
		{
			std::cerr << "Warning: dOutput must be a CUDA tensor, but isn't. This indicates suboptimal performance." <<
				std::endl;
			dOutput = dOutput.to(torch::kCUDA);
		}

		const torch::autograd::variable_list saved  = ctx->get_saved_variables();
		const at::Tensor                           input  = saved[0];
		const at::Tensor                           params = saved[1];
		const at::Tensor                           output = saved[2];

		// Assuming ModuleFunctionBackward is a class similar to ModuleFunctionForward
		torch::autograd::variable_list out;
		if(true)
		{
			const c10::intrusive_ptr<TcnnData> tcnnData = ctx->saved_data[TCNN_DATA].toCustomClass<TcnnData>();

			//ctx->save_for_backward({ input, params, dOutput });
			//ctx->saved_data[TCNN_DATA] = tcnnData;
			//torch::autograd::variable_list outputList;

			{
				torch::NoGradGuard no_grad_guard; // Equivalent to `with torch.no_grad()`

				at::Tensor scaled_grad = dOutput * tcnnData->lossScale;
				auto [inputGrad, paramsGrad] = tcnnData->nativeTcnnModule->bwd(tcnnData->nativeCtx, input, params, output, scaled_grad);
				paramsGrad = (paramsGrad.defined()) ? paramsGrad / tcnnData->lossScale : nullTensorLike(params);
				inputGrad = (inputGrad.defined()) ? inputGrad / tcnnData->lossScale : nullTensorLike(input);
				// check if inputGrad or paramsGrad have inf or nan and set them to zero if so
				out = { torch::Tensor(),  nullTensorToNone(inputGrad), nullTensorToNone(paramsGrad), torch::Tensor()};
			}
		}
		else
		{
			out = ModuleFunctionBackward::apply(ctx, dOutput, input, params, output);
			const torch::Tensor& inputGrad = out[0];
			const torch::Tensor& paramsGrad = out[1];
			out = { torch::Tensor(), nullTensorToNone(inputGrad), nullTensorToNone(paramsGrad), torch::Tensor() };
		}
		
		return out ;
	}
}
