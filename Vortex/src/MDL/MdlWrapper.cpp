#include "MdlWrapper.h"
#include <execution>

#include "ShaderVisitor.h"
#include "mdlTraversal.h"
#include "Device/OptixWrapper.h"
#include "Scene/Scene.h"
#include "Scene/Nodes/Material.h"
#include "Scene/Nodes/Shader/Texture.h"
#include "Scene/Nodes/Shader/mdl/ShaderNodes.h"

namespace vtx::mdl
{
	using namespace mi;
	using namespace base;
	using namespace neuraylib;

	static MdlState mdlState;

	MdlState* getState() {
		return &mdlState;
	}

	const char* messageKindToString(const IMessage::Kind message_kind)
	{
		switch (message_kind)
		{
		case IMessage::MSG_INTEGRATION:
			return "MDL SDK";
		case IMessage::MSG_IMP_EXP:
			return "Importer/Exporter";
		case IMessage::MSG_COMILER_BACKEND:
			return "Compiler Backend";
		case IMessage::MSG_COMILER_CORE:
			return "Compiler Core";
		case IMessage::MSG_COMPILER_ARCHIVE_TOOL:
			return "Compiler Archive Tool";
		case IMessage::MSG_COMPILER_DAG:
			return "Compiler DAG generator";
		default:
			break;
		}
		return "";
	}

	bool logMessage(IMdl_execution_context* context)
	{
		MdlState& state = *getState();
		state.lastError = "";
		for (Size i = 0; i < context->get_messages_count(); ++i)
		{
			const Handle<const IMessage> message(context->get_message(i));
			state.lastError += messageKindToString(message->get_kind());
			state.lastError += " ";
			state.lastError += ": ";
			state.lastError += message->get_string();
			switch (message->get_severity()) {
			case MESSAGE_SEVERITY_ERROR:
				VTX_ERROR(state.lastError);
				break;
			case MESSAGE_SEVERITY_WARNING:
				VTX_WARN(state.lastError);
				break;
			case MESSAGE_SEVERITY_INFO:
				VTX_INFO(state.lastError);
				break;
			case MESSAGE_SEVERITY_VERBOSE:
				VTX_TRACE(state.lastError);
				break;
			case MESSAGE_SEVERITY_DEBUG:
				VTX_TRACE(state.lastError);
				break;
			default:
				break;
			}

		}
		return context->get_error_messages_count() == 0;
	}

	void loadIneuray()
	{
		MdlState& state = *getState();
		std::string filename = utl::absolutePath(getOptions()->dllPath + "libmdl_sdk.dll");
		const HMODULE handle = LoadLibraryA(filename.c_str());
		VTX_ASSERT_CLOSE(handle, "Error Loading {} with error code {}", filename, GetLastError());

		void* symbol = GetProcAddress(handle, "mi_factory");
		VTX_ASSERT_CLOSE(symbol, "ERROR: GetProcAddress(handle, \"mi_factory\") failed with error {}", GetLastError());

		state.neuray = mi_factory<INeuray>(symbol);
		if (!state.neuray.get())
		{
			const Handle<IVersion> version(mi_factory<IVersion>(symbol));
			VTX_ASSERT_CLOSE(version, "ERROR: Incompatible library. Could not determine version.");
			VTX_ASSERT_CLOSE(!version, "ERROR: Library version {} does not match header version {}", version->get_product_version(), MI_NEURAYLIB_PRODUCT_VERSION_STRING);
		}

		VTX_INFO("MDL SDK Loaded");
	}

	void loadPlugin(std::string path)
	{
		MdlState& state = *getState();
		const Handle<IPlugin_configuration> plugin_conf(state.neuray->get_api_component<IPlugin_configuration>());

		// Try loading the requested plugin before adding any special handling
		state.result = plugin_conf->load_plugin_library(path.c_str());
		VTX_ASSERT_CLOSE(state.result == 0, "load_plugin( {} ) failed with {}", path, state.result);
	}

	void configure()
	{
		// Create the MDL compiler.
		MdlState& state = *getState();
		state.compiler = state.neuray->get_api_component<IMdl_compiler>();
		VTX_ASSERT_CLOSE(state.compiler, "ERROR: Initialization of MDL compiler failed.");

		// Configure Neuray.
		state.config = state.neuray->get_api_component<IMdl_configuration>();
		VTX_ASSERT_CLOSE(state.config, "ERROR: Retrieving MDL configuration failed.");

		state.logger = make_handle(new MdlLogger());
		state.config->set_logger(state.logger.get());

		// Path for search
		state.config->add_mdl_system_paths(); //MDL_SYSTEM_PATH is set to "C:\ProgramData\NVIDIA Corporation\mdl\" by default
		state.config->add_mdl_user_paths(); //MDL_USER_PATH is set to "C:\Users\<username>\Documents\mdl\" by default
		// Add all additional MDL and resource search paths defined inside the system description file as well.
		for (auto const& path : state.searchStartupPaths)
		{
			addSearchPath(path);
		}
		// Load plugins.

		//std::string freeimagePath = utl::absolutePath(getOptions()->dllPath + "freeimage.dll");
		//const HMODULE handle = LoadLibraryA(freeimagePath.c_str());
		//VTX_ASSERT_CLOSE(handle, "Error Loading {} with error code {}", freeimagePath, GetLastError());

		const std::string ddsPath = getOptions()->dllPath + "dds.dll";
		const std::string nv_openimageioPath = utl::absolutePath(getOptions()->dllPath + "nv_openimageio.dll");


		loadPlugin(nv_openimageioPath);
		loadPlugin(ddsPath);

	}

	void startInterfaces()
	{

		MdlState& state = *getState();
		VTX_ASSERT_CLOSE(state.neuray->start() == 0, "FATAL: Starting MDL SDK failed.");

		state.database = state.neuray->get_api_component<IDatabase>();

		state.globalScope = state.database->get_global_scope();

		state.factory = state.neuray->get_api_component<IMdl_factory>();

		state.impExpApi = state.neuray->get_api_component<IMdl_impexp_api>();


		// Configure the execution context.
		// Used for various configurable operations and for querying warnings and error messages.
		// It is possible to have more than one, in order to use different settings.
		state.context = state.factory->create_execution_context();
		state.result = state.context->set_option("internal_space", "coordinate_world");  // equals default
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");
		state.result = state.context->set_option("bundle_resources", false);             // equals default
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");
		state.result = state.context->set_option("meters_per_scene_unit", 1.0f);         // equals default
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");
		state.result = state.context->set_option("wavelength_min", 380.0f);          // equals default
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");
		state.result = state.context->set_option("wavelength_max", 780.0f);          // equals default
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");
		// If true, the "geometry.normal" field will be applied to the MDL state prior to evaluation of the given DF.
		state.result = state.context->set_option("include_geometry_normal", true);       // equals default 
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");

		const Handle<IMdl_backend_api> mdl_backend_api(state.neuray->get_api_component<IMdl_backend_api>());

		state.backend = mdl_backend_api->get_backend(IMdl_backend_api::MB_CUDA_PTX);

		state.result = state.backend->set_option("num_texture_spaces", std::to_string(getOptions()->numTextureSpaces).c_str());
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture spaces");

		state.result = state.backend->set_option("num_texture_results", std::to_string(getOptions()->numTextureResults).c_str());
		VTX_ASSERT_CLOSE(state.result == 0, "Error with number of texture results");

		// Use SM 5.0 for Maxwell and above.
		state.result = state.backend->set_option("sm_version", "50");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with sm version");

		state.result = state.backend->set_option("tex_lookup_call_mode", "direct_call");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with tex look up mode");

		state.result = state.backend->set_option("enable_auxiliary", "on");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with enable auxiliary");

		state.result = state.backend->set_option("fast_math", "on");
		VTX_ASSERT_CLOSE(state.result == 0, "Error with tex fast_math mode");

		if (getOptions()->enable_derivatives) //Not supported in this renderer
		{
			// Option "texture_runtime_with_derivs": Default is disabled.
			// We enable it to get coordinates with derivatives for texture lookup functions.
			state.result = state.backend->set_option("texture_runtime_with_derivs", "on");
			VTX_ASSERT_CLOSE(state.result == 0, "Error with texture runtime with derivatives");
		}

		VTX_ASSERT_CLOSE((state.backend->set_option("inline_aggressively", "on") == 0), "Error with inline aggressive");
		//if (getOptions()->isDebug) {
		//	Sint32 result;
		//	result = state.backend->set_option("opt_level", getOptions()->mdlOptLevel);
		//	VTX_ASSERT_CLOSE((result == 0), "Error with opt level");
		//	//result = state.backend->set_option("enable_exceptions", "on");
		//}
		//else {
		//	VTX_ASSERT_CLOSE((state.backend->set_option("inline_aggressively", "on") == 0), "Error with inline aggressive");
		//}

		// FIXME Determine what scene data the renderer needs to provide here.
		// FIXME scene_data_names is not a supported option anymore!
		//if (state.mdl_backend->set_option("scene_data_names", "*") != 0)
		//{
		//  return false;
		//}

		state.imageApi = state.neuray->get_api_component<IImage_api>();
	}

	void printLibraryVersion() {
		const MdlState& state = *getState();
		// Print library version information.
		Handle<const IVersion> version(
			state.neuray->get_api_component<const IVersion>());

		VTX_INFO("MDL SDK header version          = {}", MI_NEURAYLIB_PRODUCT_VERSION_STRING);
		VTX_INFO("MDL SDK library product name    = {}", version->get_product_name());
		VTX_INFO("MDL SDK library product version = {}", version->get_product_version());
		VTX_INFO("MDL SDK library build number    = {}", version->get_build_number());
		VTX_INFO("MDL SDK library build date      = {}", version->get_build_date());
		VTX_INFO("MDL SDK library build platform  = {}", version->get_build_platform());
		VTX_INFO("MDL SDK library version string  = {}", version->get_string());

		Uuid neuray_id_libraray = version->get_neuray_iid();
		Uuid neuray_id_interface = INeuray::IID();

		VTX_INFO("MDL SDK header interface ID     = <{}, {}, {}, {}>",
		         neuray_id_interface.m_id1,
		         neuray_id_interface.m_id2,
		         neuray_id_interface.m_id3,
		         neuray_id_interface.m_id4);
		VTX_INFO("MDL SDK library interface ID    = <{}, {}, {}, {}>",
		         neuray_id_libraray.m_id1,
		         neuray_id_libraray.m_id2,
		         neuray_id_libraray.m_id3,
		         neuray_id_libraray.m_id4);

		version = 0;
	}

	void init() {
		MdlState& state = *getState();
		loadIneuray();
		printLibraryVersion();
		for (std::string path : getOptions()->mdlSearchPath) {
			state.searchStartupPaths.push_back(utl::absolutePath(path));
		}
		configure();
		startInterfaces();
		if(getOptions()->mdlCallType == MDL_INLINE || getOptions()->mdlCallType == MDL_CUDA)
		{
			setRendererModule(getOptions()->executablePath + "bc/MaterialDirectCallable.bc", "__direct_callable__EvaluateMaterial,__replace__EvaluateMaterial");
		}
	}

	void shutDown() {

		MdlState& state = *getState();
		VTX_INFO("Shutdown: MDL");
		state.imageApi.reset();
		state.backend.reset();
		state.context.reset();
		state.factory.reset();
		state.globalScope.reset();
		state.database.reset();
		state.config.reset();
		state.compiler.reset();
		state.neuray->shutdown();
	}

	void addSearchPath(std::string path)
	{
		if(path.empty())
		{
			return;
		}
		MdlState& state = *getState();
		state.result = state.config->add_mdl_path(path.c_str());
		VTX_ASSERT_CONTINUE(state.result == 0, "add_mdl_path( {} ) failed with {}", path, state.result);

		state.result = state.config->add_resource_path(path.c_str());
		VTX_ASSERT_CONTINUE(state.result == 0, "add_resource_path( {} ) failed with {}", path, state.result);
	}

	void setRendererModule(const std::string& rendererModule, const std::string& visibleFunction)
	{
		std::vector<char> binary = utl::readData(rendererModule);

		VTX_ASSERT_CLOSE(!binary.empty(), "ERROR: {} could not be opened", rendererModule);

		Sint32 result = getState()->backend->set_option_binary("llvm_renderer_module", binary.data(), binary.size());
		VTX_ASSERT_CLOSE(result == 0, "ERROR: Setting PTX option llvm_renderer_module failed");

		// limit functions for which PTX code is generated to the entry functions
		result = getState()->backend->set_option("visible_functions", visibleFunction.c_str());
		VTX_ASSERT_CLOSE(result == 0, "ERROR: Setting PTX option visible_functions failed");
	};

	void MdlState::openTransaction(const bool openFactories) {
		bool isTransactionNew = false;
		if (!tI.transaction)
		{
			//VTX_INFO("Creating global transaction + Handle");
			tI.transaction = make_handle<ITransaction>(globalScope->create_transaction());
			isTransactionNew = true;
		}
		else if (!tI.transaction->is_open()) {
			//VTX_INFO("Creating global transaction");
			tI.transaction = globalScope->create_transaction();
			isTransactionNew = true;
		}
		else
		{
			//VTX_INFO("Using existing global transaction");
		}
		if (isTransactionNew && openFactories)
		{
			//VTX_INFO("Opening Expression, Value and Type Factory");
			tI.expressionFactory = factory->create_expression_factory(tI.transaction.get());
			tI.valueFactory = factory->create_value_factory(tI.transaction.get());
			tI.typeFactory = factory->create_type_factory(tI.transaction.get());
		}
	}

	void MdlState::commitTransaction()
	{
		if (tI.transaction && tI.transaction->is_open()) {
			//VTX_WARN("Committing global transaction");
			//tI.transaction->commit();
		}
	}

	/*path expressed as relative to the search path added to mdl sdk*/
	std::string pathToModuleName(const std::string& materialPath) {
		// Replace backslashes with colons
		std::string output;

		for (size_t i = 0; i < materialPath.size(); ++i) {
			if (materialPath[i] == '\\') {
				output.push_back(':');
				output.push_back(':');
			}
			else if (materialPath[i] == '/')
			{
				output.push_back(':');
				output.push_back(':');
			}
			else {
				output.push_back(materialPath[i]);
			}
		}

		// Remove the file extension
		const std::size_t last_dot = output.find_last_of('.');
		if (last_dot != std::string::npos) {
			output.erase(last_dot);
		}

		return output;
	};

	std::string removeMdlPrefix(const std::string& name)
	{
		std::string       result = name;
		if (const std::string prefix = "mdl"; result.substr(0, prefix.size()) == prefix)
		{
			result.erase(0, prefix.size());
		}
		return result;
	}

	std::string getMaterialDatabaseName(const IModule* module, const IString* moduleDatabaseName, const std::string& materialName) {
		std::string materialDatabaseName = std::string(moduleDatabaseName->get_c_str()) + "::" + materialName;

		// Return input if it already contains a signature.
		if (materialDatabaseName.back() == ')')
		{
			return materialDatabaseName;
		}

		const Handle<const IArray> result(module->get_function_overloads(materialDatabaseName.c_str()));

		// Not supporting multiple function overloads with the same name but different signatures.
		if (!result || result->get_length() != 1)
		{
			return std::string();
		}

		const Handle<const IString> overloads(result->get_element<IString>(static_cast<Size>(0)));

		return overloads->get_c_str();

	}

	// Utility function to dump the hash, arguments, temporaries, and fields of a compiled material.
	void dump_compiled_material(
		mi::neuraylib::ITransaction* transaction,
		mi::neuraylib::IMdl_factory* mdl_factory,
		const mi::neuraylib::ICompiled_material* cm,
		std::ostream& s)
	{
		mi::base::Handle<mi::neuraylib::IValue_factory> value_factory(
			mdl_factory->create_value_factory(transaction));
		mi::base::Handle<mi::neuraylib::IExpression_factory> expression_factory(
			mdl_factory->create_expression_factory(transaction));

		mi::base::Uuid hash = cm->get_hash();
		char buffer[36];
		snprintf(buffer, sizeof(buffer),
				 "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
		s << "    hash overall = " << buffer << std::endl;

		for (mi::Uint32 i = mi::neuraylib::SLOT_FIRST; i <= mi::neuraylib::SLOT_LAST; ++i) {
			hash = cm->get_slot_hash(mi::neuraylib::Material_slot(i));
			snprintf(buffer, sizeof(buffer),
					 "%08x %08x %08x %08x", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
			s << "    hash slot " << std::setw(2) << i << " = " << buffer << std::endl;
		}

		mi::Size parameter_count = cm->get_parameter_count();
		for (mi::Size i = 0; i < parameter_count; ++i) {
			mi::base::Handle<const mi::neuraylib::IValue> argument(cm->get_argument(i));
			std::stringstream name;
			name << i;
			mi::base::Handle<const mi::IString> result(
				value_factory->dump(argument.get(), name.str().c_str(), 1));
			s << "    argument " << result->get_c_str() << std::endl;
		}

		mi::Size temporary_count = cm->get_temporary_count();
		for (mi::Size i = 0; i < temporary_count; ++i) {
			mi::base::Handle<const mi::neuraylib::IExpression> temporary(cm->get_temporary(i));
			std::stringstream name;
			name << i;
			mi::base::Handle<const mi::IString> result(
				expression_factory->dump(temporary.get(), name.str().c_str(), 1));
			s << "    temporary " << result->get_c_str() << std::endl;
		}

		mi::base::Handle<const mi::neuraylib::IExpression> body(cm->get_body());
		mi::base::Handle<const mi::IString> result(expression_factory->dump(body.get(), 0, 1));
		s << "    body " << result->get_c_str() << std::endl;

		s << std::endl;
	}

	void compileMaterial(const std::string functionDatabaseName, const std::string materialDataBaseName)
	{
		MdlState& state = *getState();
		const TransactionInterfaces* tI = state.getTransactionInterfaces();
		{
			Handle<const IFunction_call> materialCall(tI->transaction->access<mi::neuraylib::IFunction_call>(functionDatabaseName.c_str()));
			//Create material Instance
			const Handle<const IMaterial_instance> materialInstance(materialCall->get_interface<IMaterial_instance>());

			//Create compiled material
			const Uint32 flags = IMaterial_instance::CLASS_COMPILATION;
			const Handle<ICompiled_material> compiledMaterial(materialInstance->create_compiled_material(flags, state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);

			base::Uuid hash = compiledMaterial->get_hash();

			VTX_INFO("Compiled material hash:{} {} {} {} ", hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);

			//dump_compiled_material(tI->transaction.get(), state.factory.get(), compiledMaterial.get(), std::cout);
			tI->transaction->store(compiledMaterial.get(), materialDataBaseName.c_str());
		}
	}

	bool isValidDistribution(IExpression const* expr)
	{
		if (expr == nullptr)
		{
			return false;
		}

		IExpression::Kind exprKind = expr->get_kind();

		if (exprKind == IExpression::EK_CONSTANT)
		{
			const Handle<IExpression_constant const> expr_constant(expr->get_interface<IExpression_constant>());
			const Handle<IValue const> value(expr_constant->get_value());

			IValue::Kind a = value->get_kind();

			if (value->get_kind() == IValue::VK_INVALID_DF)
			{
				return false;
			}
		}
		else if (exprKind == IExpression::EK_DIRECT_CALL)
		{
			const Handle<IExpression_direct_call const> expr_direct_call(expr->get_interface<IExpression_direct_call>());
			const Handle<IExpression_list const> args(expr_direct_call->get_arguments());
			size_t numArgs = args->get_size();
			for(size_t i =0; i < numArgs; ++i)
			{
				const Handle<IExpression const> arg(args->get_expression(i));
				if (!isValidDistribution(arg.get()))
				{
					return false;
				}
			}
		}
		return true;
	}
	
	graph::Configuration determineShaderConfiguration(const std::string& materialDbName)
	{
		MdlState& state = *getState();
		TransactionInterfaces* tI = state.getTransactionInterfaces();
		graph::Configuration config;
		do {
			const Handle<const ICompiled_material> compiledMaterial((tI->transaction->access(materialDbName.c_str())->get_interface<ICompiled_material>()));
			//const ICompiled_material* compiledMaterial = shader->compilation.compiledMaterial.get();

			ExprEvaluation<bool> thinWalledEval = analyzeExpression<bool>(compiledMaterial, "thin_walled");
			config.isThinWalledConstant = thinWalledEval.isConstant;
			if(config.isThinWalledConstant)
			{
				config.isThinWalled = thinWalledEval.value;
			}
			//SLOT_SURFACE_SCATTERING
			ExprEvaluation<void*> surfaceEval = analyzeExpression<void*>(compiledMaterial, "surface.scattering");
			config.isSurfaceBsdfValid = surfaceEval.isValid;

			config.isBackfaceBsdfValid = false;

			if (!config.isThinWalledConstant || config.isThinWalled)
			{
				ExprEvaluation<void*> backFaceEval = analyzeExpression<void*>(compiledMaterial, "backface.scattering");
				// When backface == bsdf() MDL uses the surface scattering on both sides, irrespective of the thin_walled state.
				Handle<IExpression const> backface_scattering_expr(compiledMaterial->lookup_sub_expression("backface.scattering"));

				config.isBackfaceBsdfValid = backFaceEval.isValid;

				if (config.isBackfaceBsdfValid)
				{
					// Only use the backface scattering when it's valid and different from the surface scattering expression.
					config.isBackfaceBsdfValid = (compiledMaterial->get_slot_hash(SLOT_SURFACE_SCATTERING) != compiledMaterial->get_slot_hash(SLOT_BACKFACE_SCATTERING));
				}
			}


			ExprEvaluation<bool> surfaceEdfEval = analyzeExpression<bool>(compiledMaterial, "surface.emission.emission");
			config.isSurfaceEdfValid = surfaceEdfEval.isValid;
			if (config.isSurfaceEdfValid)
			{
				ExprEvaluation<math::vec3f> surfaceEdfIntensityEval = analyzeExpression<math::vec3f>(compiledMaterial, "surface.emission.intensity");
				config.isSurfaceIntensityConstant = surfaceEdfIntensityEval.isConstant;
				if(surfaceEdfIntensityEval.isConstant)
				{
					config.surfaceIntensity = surfaceEdfIntensityEval.value;
				}
				ExprEvaluation<int> surfaceEdfModeEval = analyzeExpression<int>(compiledMaterial, "surface.emission.mode");
				config.isSurfaceIntensityModeConstant = surfaceEdfModeEval.isConstant;
				if (surfaceEdfModeEval.isConstant)
				{
					config.surfaceIntensityMode = surfaceEdfModeEval.value;
				}
			}

			if ((!config.isThinWalledConstant || config.isThinWalled))
			{
				ExprEvaluation<bool> backfaceEdfEval = analyzeExpression<bool>(compiledMaterial, "backface.emission.emission");
				config.isBackfaceEdfValid = backfaceEdfEval.isValid;
				if (config.isBackfaceEdfValid)
				{
					ExprEvaluation<math::vec3f> backfaceEdfIntensityEval = analyzeExpression<math::vec3f>(compiledMaterial, "backface.emission.intensity");
					config.isBackfaceIntensityConstant = backfaceEdfIntensityEval.isConstant;
					if (backfaceEdfIntensityEval.isConstant)
					{
						config.backfaceIntensity = backfaceEdfIntensityEval.value;
					}
					ExprEvaluation<int> backfaceEdfModeEval = analyzeExpression<int>(compiledMaterial, "backface.emission.mode");
					config.isBackfaceIntensityModeConstant = backfaceEdfModeEval.isConstant;
					if (backfaceEdfModeEval.isConstant)
					{
						config.backfaceIntensityMode = backfaceEdfModeEval.value;
					}
					// When surface and backface expressions are identical, reuse the surface expression to generate less code.
					config.useBackfaceEdf = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_EDF_EMISSION) != compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_EDF_EMISSION));

					// If the surface and backface emission use different intensities then use the backface emission intensity.
					config.useBackfaceIntensity = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_INTENSITY) != compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_INTENSITY));

					// If the surface and backface emission use different modes (radiant exitance vs. power) then use the backface emission intensity mode.
					config.useBackfaceIntensityMode = (compiledMaterial->get_slot_hash(SLOT_SURFACE_EMISSION_MODE) != compiledMaterial->get_slot_hash(SLOT_BACKFACE_EMISSION_MODE));
				}
			}

			ExprEvaluation<math::vec3f> iorEval = analyzeExpression<math::vec3f>(compiledMaterial, "ior");

			config.isIorConstant = iorEval.isConstant;
			if(config.isIorConstant)
			{
				config.ior = iorEval.value;
			}

			// If the VDF is valid, it is the df::anisotropic_vdf(). ::vdf() is not a valid VDF.
			// Though there aren't any init, sample, eval or pdf functions genereted for a VDF.
			ExprEvaluation<void*> volumeScatteringEval = analyzeExpression<void*>(compiledMaterial, "volume.scattering");

			config.isVdfValid = volumeScatteringEval.isValid;

			// Absorption coefficient. Can be used without valid VDF.

			ExprEvaluation<math::vec3f> absorptionCoeffEval = analyzeExpression<math::vec3f>(compiledMaterial, "volume.absorption_coefficient");
			config.isAbsorptionCoefficientConstant = absorptionCoeffEval.isConstant;
			if(config.isAbsorptionCoefficientConstant)
			{
				config.absorptionCoefficient = absorptionCoeffEval.value;
				if(config.absorptionCoefficient != math::vec3f(0.0f))
				{
					config.useVolumeAbsorption = true;
				}
			}
			else
			{
				config.useVolumeAbsorption = true;
			}


			if(config.isVdfValid)
			{
				// Scattering coefficient. Only used when there is a valid VDF. 
				ExprEvaluation<math::vec3f> scatteringCoeffEval = analyzeExpression<math::vec3f>(compiledMaterial, "volume.scattering_coefficient");
				config.isScatteringCoefficientConstant = scatteringCoeffEval.isConstant;
				if (config.isScatteringCoefficientConstant)
				{
					config.scatteringCoefficient = scatteringCoeffEval.value;
					if (config.scatteringCoefficient != math::vec3f(0.0f))
					{
						config.useVolumeScattering = true;
					}
				}
				else
				{
					config.useVolumeScattering = true;
				}

				// Directional bias (Henyey_Greenstein g factor.) Only used when there is a valid VDF and volume scattering coefficient not zero.
				ExprEvaluation<float> directionalBiasEval = analyzeExpression<float>(compiledMaterial, "volume.scattering.directional_bias");
				if(!directionalBiasEval.isValid)
				{
					config.isDirectionalBiasConstant = false;
				}
				config.isDirectionalBiasConstant = directionalBiasEval.isConstant;
				if (config.isDirectionalBiasConstant)
				{
					config.directionalBias = directionalBiasEval.value;
				}
			}

			// geometry.displacement is not supported by this renderer.

			// geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

			config.cutoutOpacity = 1.0f; // Default is fully opaque.
			config.isCutoutOpacityConstant = compiledMaterial->get_cutout_opacity(&config.cutoutOpacity); // This sets cutout opacity to -1.0 when it's not constant!
			config.useCutoutOpacity = !config.isCutoutOpacityConstant || config.cutoutOpacity < 1.0f;

			ExprEvaluation<void*> hairBsdfEval = analyzeExpression<void*>(compiledMaterial, "hair");

			config.isHairBsdfValid = hairBsdfEval.isValid; // True if hair != hair_bsdf().

			// Check if front face is emissive

			compiledMaterial->release();

		} while (0);
		state.commitTransaction();
		return config;
	}

	std::vector<Target_function_description> createShaderDescription(const graph::Configuration& config, const vtxID& shaderIndex, graph::FunctionNames& fNames) {
		std::vector<Target_function_description> descriptions;

		if(getOptions()->mdlCallType == MDL_DIRECT_CALL)
		{
			// These are all expressions required for a materials which does everything supported in this renderer. 
			// The Target_function_description only stores the C-pointers to the base names!
			// Make sure these are not destroyed as long as the descs vector is used.
			fNames = graph::Scene::getSim()->getNode<graph::Material>(shaderIndex)->getFunctionNames();

			// Centralize the init functions in a single material init().
			// This will only save time when there would have been multiple init functions inside the shader.
			// Also for very complicated materials with cutout opacity this is most likely a loss,
			// because the geometry.cutout is only needed inside the anyhit program and 
			// that doesn't need additional evalations for the BSDFs, EDFs, or VDFs at that point.

			descriptions.emplace_back("init", fNames.init.c_str());

			if (!config.isThinWalledConstant)
			{
				descriptions.emplace_back("thin_walled", fNames.thinWalled.c_str());
			}
			if (config.isSurfaceBsdfValid)
			{
				descriptions.emplace_back("surface.scattering", fNames.surfaceScattering.c_str());
			}
			if (config.isSurfaceEdfValid)
			{
				descriptions.emplace_back("surface.emission.emission", fNames.surfaceEmissionEmission.c_str());
				if (!config.isSurfaceIntensityConstant)
				{
					descriptions.emplace_back("surface.emission.intensity", fNames.surfaceEmissionIntensity.c_str());
				}
				if (!config.isSurfaceIntensityModeConstant)
				{
					descriptions.emplace_back("surface.emission.mode", fNames.surfaceEmissionMode.c_str());
				}
			}
			if (config.isBackfaceBsdfValid)
			{
				descriptions.emplace_back("backface.scattering", fNames.backfaceScattering.c_str());
			}
			if (config.isBackfaceEdfValid)
			{
				if (config.useBackfaceEdf)
				{
					descriptions.emplace_back("backface.emission.emission", fNames.backfaceEmissionEmission.c_str());
				}
				if (config.useBackfaceIntensity && !config.isBackfaceIntensityConstant)
				{
					descriptions.emplace_back("backface.emission.intensity", fNames.backfaceEmissionIntensity.c_str());
				}
				if (config.useBackfaceIntensityMode && !config.isBackfaceIntensityModeConstant)
				{
					descriptions.emplace_back("backface.emission.mode", fNames.backfaceEmissionMode.c_str());
				}
			}
			if (!config.isIorConstant)
			{
				descriptions.emplace_back("ior", fNames.ior.c_str());
			}
			if (!config.isAbsorptionCoefficientConstant)
			{
				descriptions.emplace_back("volume.absorption_coefficient", fNames.volumeAbsorptionCoefficient.c_str());
			}
			if (config.isVdfValid)
			{
				// DAR This fails in ILink_unit::add_material(). The MDL SDK is not generating functions for VDFs!
				//descriptions.push_back(Target_function_description("volume.fNames...c_str()));

				// The scattering coefficient and directional bias are not used when there is no valid VDF.
				if (!config.isScatteringCoefficientConstant)
				{
					descriptions.emplace_back("volume.scattering_coefficient", fNames.volumeScatteringCoefficient.c_str());
				}

				if (!config.isDirectionalBiasConstant)
				{
					descriptions.emplace_back("volume.scattering.directional_bias", fNames.volumeDirectionalBias.c_str());
				}

				// volume.scattering.emission_intensity is not implemented.
			}

			// geometry.displacement is not implemented.

			// geometry.normal is automatically handled because of set_option("include_geometry_normal", true);

			if (config.useCutoutOpacity)
			{
				descriptions.emplace_back("geometry.cutout_opacity", fNames.geometryCutoutOpacity.c_str());
			}
			if (config.isHairBsdfValid)
			{
				descriptions.emplace_back("hair", fNames.hairBsdf.c_str());
			}

		}
		else if (getOptions()->mdlCallType == MDL_CUDA || getOptions()->mdlCallType == MDL_INLINE)
		{
			descriptions.emplace_back(Target_function_description("init", "init"));
			descriptions.emplace_back(Target_function_description("thin_walled", "thinWalled"));
			descriptions.emplace_back(Target_function_description("surface.scattering", "frontBsdf"));
			descriptions.emplace_back(Target_function_description("surface.emission.emission", "frontEdf"));
			descriptions.emplace_back(Target_function_description("surface.emission.intensity", "frontEdfIntensity"));
			descriptions.emplace_back(Target_function_description("surface.emission.mode", "frontEdfMode"));
			descriptions.emplace_back(Target_function_description("backface.scattering", "backBsdf"));
			descriptions.emplace_back(Target_function_description("backface.emission.emission", "backEdf"));
			descriptions.emplace_back(Target_function_description("backface.emission.intensity", "backEdfIntensity"));
			descriptions.emplace_back(Target_function_description("backface.emission.mode", "backEdfMode"));
			descriptions.emplace_back(Target_function_description("ior", "iorEvaluation"));
			//descriptions.emplace_back(Target_function_description("volume.scattering", ));
			//descriptions.emplace_back(Target_function_description("volume.absorption_coefficient", "volumeAbsorptionCoefficient"));
			//descriptions.emplace_back(Target_function_description("volume.scattering_coefficient", "volumeScatteringCoefficient"));
			//descriptions.emplace_back(Target_function_description("volume.scattering.directional_bias", "volumeDirectionalBias"));
			//descriptions.emplace_back(Target_function_description("geometry.displacement", ));
			descriptions.emplace_back(Target_function_description("geometry.cutout_opacity", "opacityEvaluation"));
			//descriptions.emplace_back(Target_function_description("geometry.normal", ));
		}
		else{
			descriptions.emplace_back(Target_function_description("init", "init"));
			descriptions.emplace_back(Target_function_description("thin_walled", "thinWalled"));
			descriptions.emplace_back(Target_function_description("surface.scattering", "frontBsdf"));
			descriptions.emplace_back(Target_function_description("surface.emission.emission", "frontEdf"));
			descriptions.emplace_back(Target_function_description("surface.emission.intensity", "frontEdfIntensity"));
			descriptions.emplace_back(Target_function_description("surface.emission.mode", "frontEdfMode"));
			descriptions.emplace_back(Target_function_description("backface.scattering", "backBsdf"));
			descriptions.emplace_back(Target_function_description("backface.emission.emission", "backEdf"));
			descriptions.emplace_back(Target_function_description("backface.emission.intensity", "backEdfIntensity"));
			descriptions.emplace_back(Target_function_description("backface.emission.mode", "backEdfMode"));
			descriptions.emplace_back(Target_function_description("ior", "iorEvaluation"));
			//descriptions.emplace_back(Target_function_description("volume.scattering", ));
			//descriptions.emplace_back(Target_function_description("volume.absorption_coefficient", "volumeAbsorptionCoefficient"));
			//descriptions.emplace_back(Target_function_description("volume.scattering_coefficient", "volumeScatteringCoefficient"));
			//descriptions.emplace_back(Target_function_description("volume.scattering.directional_bias", "volumeDirectionalBias"));
			//descriptions.emplace_back(Target_function_description("geometry.displacement", ));
			descriptions.emplace_back(Target_function_description("geometry.cutout_opacity", "opacityEvaluation"));
			//descriptions.emplace_back(Target_function_description("geometry.normal", ));
		}
		
		return descriptions;


	}

	void addMaterialToLinkUnit(const std::string& materialDbName, const graph::Configuration& config, const vtxID& shaderIndex, Handle<ILink_unit>& linkUnit)
	{
		MdlState& state = *getState();
		const TransactionInterfaces* tI = state.getTransactionInterfaces();
		Handle<ITarget_code const>   targetCode;
		do {
			const Handle<const ICompiled_material> compiledMaterial(tI->transaction->access<ICompiled_material>(materialDbName.c_str()));

			graph::FunctionNames fNames;
			std::vector<Target_function_description> descriptions = createShaderDescription(config, shaderIndex, fNames);

			if (linkUnit.get() == nullptr)
			{
				linkUnit = make_handle(state.backend->create_link_unit(tI->transaction.get(), state.context.get()));
			}

			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);
			const Sint32 result = linkUnit->add_material(compiledMaterial.get(), descriptions.data(), descriptions.size(), state.context.get());
			VTX_ASSERT_BREAK((result == 0 && logMessage(state.context.get())), state.lastError);

		} while (false);

		state.commitTransaction();
	}

	Handle<ITarget_code const> generateTargetCode(Handle<ILink_unit>& linkUnit)
	{
		MdlState& state = *getState();
		const TransactionInterfaces* tI = state.getTransactionInterfaces();
		Handle<ITarget_code const>   targetCode;

		{
			targetCode = make_handle(state.backend->translate_link_unit(linkUnit.get(), state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);

		}
		state.commitTransaction();
		return targetCode;
	}

	std::vector<std::shared_ptr<graph::Texture>> createTextureResources(Handle<ITarget_code const>& targetCode)
	{
		std::vector<std::shared_ptr<graph::Texture>> textures;
		// TODO We have to store the textures, light profiles and bsdf indices since these will be refencered by the mdl lookup functions
		for (Size i = 1, n = targetCode->get_texture_count(); i < n; ++i) {
			auto texture = ops::createNode<graph::Texture>(targetCode->get_texture(i), targetCode->get_texture_shape(i));
			texture->mdlIndex = i;
			textures.emplace_back(texture);
		}
		return textures;
	}

	std::vector<std::shared_ptr<graph::LightProfile>> createLightProfileResources(Handle<ITarget_code const>& targetCode)
	{
		std::vector<std::shared_ptr<graph::LightProfile>> lightProfiles;
		if (targetCode->get_light_profile_count() > 0)
		{
			for (mi::Size i = 1, n = targetCode->get_light_profile_count(); i < n; ++i)
			{
				auto lightProfile = ops::createNode<graph::LightProfile>(targetCode->get_light_profile(i));
				lightProfile->mdlIndex = i;
				lightProfiles.emplace_back(lightProfile);
			}
		}
		return lightProfiles;
	}

	std::vector<std::shared_ptr<graph::BsdfMeasurement>> createBsdfMeasurementResources(Handle<ITarget_code const>& targetCode)
	{
		std::vector<std::shared_ptr<graph::BsdfMeasurement>> bsdfMeasurements;
		if (targetCode->get_bsdf_measurement_count() > 0)
		{
			for (mi::Size i = 1, n = targetCode->get_bsdf_measurement_count(); i < n; ++i)
			{
				auto bsdfMeasurement = ops::createNode<graph::BsdfMeasurement>(targetCode->get_bsdf_measurement(i));
				bsdfMeasurement->mdlIndex = i;
				bsdfMeasurements.emplace_back(bsdfMeasurement);
			}
		}

		return bsdfMeasurements;
	}

	Handle<ITarget_code const> createTargetCode(const std::string& materialDbName, const graph::Configuration& config, const vtxID& shaderIndex)
	{

		MdlState&                    state = *getState();
		const TransactionInterfaces* tI    = state.getTransactionInterfaces();
		Handle<ITarget_code const>   targetCode;
		do {
			const Handle<const ICompiled_material> compiledMaterial(tI->transaction->access<ICompiled_material>(materialDbName.c_str()));

			//const Handle<const ICompiled_material> compiledMaterial((tI->transaction->access(("compiledMaterial_" + materialDbName).c_str())->get_interface<ICompiled_material>()));
			//const Handle<const IFunction_definition> materialDefinition(tI->transaction->access<IFunction_definition>(materialDbName.c_str()));

			graph::FunctionNames fNames;
			std::vector<Target_function_description> descriptions = createShaderDescription(config, shaderIndex, fNames);

			VTX_INFO("Creating target code for shader {} index {} with {} functions.", materialDbName, shaderIndex, descriptions.size());

			Handle<ILink_unit> linkUnit(tI->transaction->edit<ILink_unit>("VORTEX_linkUnit"));
			if(linkUnit.get()==nullptr)
			{
				linkUnit = make_handle(state.backend->create_link_unit(tI->transaction.get(), state.context.get()));
				tI->transaction->store(linkUnit.get(), ("VORTEX_linkUnit"));
			}

			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);
			const Sint32 result = linkUnit->add_material(compiledMaterial.get(), descriptions.data(), descriptions.size(), state.context.get());
			VTX_ASSERT_BREAK((result == 0 && logMessage(state.context.get())), state.lastError);

			targetCode = make_handle(state.backend->translate_link_unit(linkUnit.get(), state.context.get()));
			VTX_ASSERT_BREAK((logMessage(state.context.get())), state.lastError);
			//mi::neuraylib::ITarget_code::Prototype_language

		} while (false);

		state.commitTransaction();
		//tI->transaction->commit();

		return targetCode;
	}


	graph::shader::Annotation getAnnotation(Handle<IAnnotation_block const> annoBlock)
	{
		// Check for annotation info
		graph::shader::Annotation annotation;
		if (annoBlock)
		{
			Annotation_wrapper annos(annoBlock.get());
			Size annoIndex;

			annoIndex = annos.get_annotation_index("::anno::display_name(string)");
			if (annoIndex != static_cast<Size>(-1))
			{
				char const* displayName = nullptr;
				annos.get_annotation_param_value(annoIndex, 0, displayName);
				annotation.displayName = displayName;
			}

			std::string rangeType = "float";
			annoIndex = annos.get_annotation_index("::anno::hard_range(float,float)");
			if (annoIndex == static_cast<Size>(-1))
			{
				annoIndex = annos.get_annotation_index("::anno::soft_range(float,float)");
			}
			if (annoIndex == static_cast<Size>(-1))
			{
				annoIndex = annos.get_annotation_index("::anno::hard_range(int,int)");
				rangeType = "int";
			}
			if (annoIndex == static_cast<Size>(-1))
			{
				annoIndex = annos.get_annotation_index("::anno::soft_range(int,int)");
				rangeType = "int";
			}
			if (annoIndex != static_cast<Size>(-1) && rangeType=="float")
			{
				annos.get_annotation_param_value(annoIndex, 0, annotation.range[0]);
				annos.get_annotation_param_value(annoIndex, 1, annotation.range[1]);
			}
			else if (annoIndex != static_cast<Size>(-1) && rangeType == "int")
			{
				int range[2];
				annos.get_annotation_param_value(annoIndex, 0, range[0]);
				annos.get_annotation_param_value(annoIndex, 1, range[1]);
				annotation.range[0] = static_cast<float>(range[0]);
				annotation.range[1] = static_cast<float>(range[1]);
			}


			annoIndex = annos.get_annotation_index("::anno::in_group(string)");
			if (annoIndex != static_cast<Size>(-1))
			{
				char const* groupName = nullptr;
				annos.get_annotation_param_value(annoIndex, 0, groupName);
				annotation.groupName = groupName;
			}
			annotation.isValid = true;
		}

		return annotation;
	}

	graph::shader::ParameterInfo generateParamInfo(
		const size_t index,
		const Handle<const ICompiled_material>& compiledMat,
		char* argBlockData,
		Handle<ITarget_value_layout const>& argBlockLayout,
		const Handle<IAnnotation_list const>& annoList,
		std::map<std::string, std::shared_ptr<graph::shader::EnumTypeInfo>>& mapEnumTypes
	)
	{
		using graph::shader::EnumValue;
		using graph::shader::EnumTypeInfo;
		using graph::shader::ParameterInfo;

		const char* name = compiledMat->get_parameter_name(index);
		if (name == nullptr) return {};

		Handle argument = make_handle<IValue const>(compiledMat->get_argument(index));
		const IValue::Kind kind = argument->get_kind();
		auto paramKind = vtx::graph::shader::PK_UNKNOWN;
		auto paramArrayElemKind = vtx::graph::shader::PK_UNKNOWN;
		Size paramArraySize = 0;
		Size paramArrayPitch = 0;
		const EnumTypeInfo* enumType = nullptr;

		switch (kind)
		{
			case IValue::VK_FLOAT:
				paramKind = vtx::graph::shader::PK_FLOAT;
				break;
			case IValue::VK_COLOR:
				paramKind = vtx::graph::shader::PK_COLOR;
				break;
			case IValue::VK_BOOL:
				paramKind = vtx::graph::shader::PK_BOOL;
				break;
			case IValue::VK_INT:
				paramKind = vtx::graph::shader::PK_INT;
				break;
			case IValue::VK_VECTOR:
			{
				const Handle val = make_handle<const IValue_vector>(argument->get_interface<const IValue_vector>());
				const Handle valType = make_handle<const IType_vector>(val->get_type());

				if (const Handle elemType = make_handle<const IType_atomic>(valType->get_element_type()); elemType->get_kind() == IType::TK_FLOAT)
				{
					switch (valType->get_size())
					{
						case 2:
							paramKind = vtx::graph::shader::PK_FLOAT2;
							break;
						case 3:
							paramKind = vtx::graph::shader::PK_FLOAT3;
							break;
					}
				}
			}
			break;
			case IValue::VK_ARRAY:
			{
				const Handle val = make_handle<const IValue_array>(argument->get_interface<const IValue_array>());
				const Handle valType = make_handle<const IType_array>(val->get_type());

				// we currently only support arrays of some values
				switch (const Handle elemType = make_handle<const IType>(valType->get_element_type()); elemType->get_kind())
				{
					case IType::TK_FLOAT:
						paramArrayElemKind = vtx::graph::shader::PK_FLOAT;
						break;
					case IType::TK_COLOR:
						paramArrayElemKind = vtx::graph::shader::PK_COLOR;
						break;
					case IType::TK_BOOL:
						paramArrayElemKind = vtx::graph::shader::PK_BOOL;
						break;
					case IType::TK_INT:
						paramArrayElemKind = vtx::graph::shader::PK_INT;
						break;
					case IType::TK_VECTOR:
					{
						const Handle valType = make_handle<const IType_vector>(elemType->get_interface<const IType_vector>());

						if (const Handle vElemType = make_handle<const IType_atomic>(valType->get_element_type()); vElemType->get_kind() == IType::TK_FLOAT)
						{
							switch (valType->get_size())
							{
								case 2:
									paramArrayElemKind = vtx::graph::shader::PK_FLOAT2;
									break;
								case 3:
									paramArrayElemKind = vtx::graph::shader::PK_FLOAT3;
									break;
							}
						}
					}
					break;
					default:
						break;
				}
				if (paramArrayElemKind != vtx::graph::shader::PK_UNKNOWN)
				{
					paramKind = vtx::graph::shader::PK_ARRAY;
					paramArraySize = valType->get_size();

					// determine pitch of array if there are at least two elements
					if (paramArraySize > 1)
					{
						const Target_value_layout_state arrayState(argBlockLayout->get_nested_state(index));
						const Target_value_layout_state nextElemState(argBlockLayout->get_nested_state(1, arrayState));

						IValue::Kind kind;
						Size paramSize;

						const Size startOffset = argBlockLayout->get_layout(kind, paramSize, arrayState);
						const Size nextOffset = argBlockLayout->get_layout(kind, paramSize, nextElemState);

						paramArrayPitch = nextOffset - startOffset;
					}
				}
			}
			break;
			case IValue::VK_ENUM:
			{
				const Handle val = make_handle<const IValue_enum>(argument->get_interface<const IValue_enum>());
				const Handle valType = make_handle<const IType_enum>(val->get_type());

				EnumTypeInfo* info = nullptr;
				std::string enumTypeName = valType->get_symbol();

				if (auto it = mapEnumTypes.find(enumTypeName); it != mapEnumTypes.end()) {
					info = it->second.get();
				}
				// prepare info for this enum type if not seen so far
				else {
					std::shared_ptr<EnumTypeInfo> enumTypeInfo(new EnumTypeInfo());
					for (Size i = 0, n = valType->get_size(); i < n; ++i)
					{
						std::string enumName = valType->get_value_name(i);
						int			enumValue = valType->get_value_code(i);
						enumTypeInfo->add(enumName, enumValue);
					}
					mapEnumTypes[enumTypeName] = enumTypeInfo;
					info = enumTypeInfo.get();
				}
				enumType = info;
				paramKind = vtx::graph::shader::PK_ENUM;
			}
			break;
			case IValue::VK_STRING:
				paramKind = vtx::graph::shader::PK_STRING;
				break;
			case IValue::VK_TEXTURE:
				paramKind = vtx::graph::shader::PK_TEXTURE;
				break;
			case IValue::VK_LIGHT_PROFILE:
				paramKind = vtx::graph::shader::PK_LIGHT_PROFILE;
				break;
			case IValue::VK_BSDF_MEASUREMENT:
				paramKind = vtx::graph::shader::PK_BSDF_MEASUREMENT;
				break;
			default:
				// Unsupported? -> skip
				return {};
		}

		// Get the offset of the argument within the target argument block
		const Target_value_layout_state targetValueLayoutState(argBlockLayout->get_nested_state(index));
		IValue::Kind kind2;
		Size paramSize;
		const Size offset = argBlockLayout->get_layout(kind2, paramSize, targetValueLayoutState);
		if (kind != kind2)
		{
			VTX_WARN("Argument kind mismatch During Material Parameter Generation {}", name);
			return {};  // layout is invalid -> skip
		}

		char* dataPtr = argBlockData + offset;


		ParameterInfo paramInfo;

		paramInfo.index = index;
		paramInfo.argumentName = name;
		paramInfo.annotation.groupName = "";
		paramInfo.kind = paramKind;
		paramInfo.arrayElemKind = paramArrayElemKind;
		paramInfo.arraySize = paramArraySize;
		paramInfo.arrayPitch = paramArrayPitch;
		paramInfo.dataPtr = dataPtr;
		paramInfo.enumInfo = enumType;
		paramInfo.annotation = getAnnotation(make_handle<IAnnotation_block const>(annoList->get_annotation_block(name)));

		return paramInfo;
	}

	std::vector<graph::shader::ParameterInfo> getArgumentBlockData(
		const std::string&                                                   materialDbName,
		const std::string&                                                   functionDefinitionSignature,
		const Handle<ITarget_code const>&                                    targetCode,
		Handle<ITarget_argument_block>&                                      argumentBlockClone,
		std::map<std::string, std::shared_ptr<graph::shader::EnumTypeInfo>>& mapEnumTypes,
		const int                                                            materialAdditionIndex)
	{
		char* argBlockData = nullptr;
		Handle<ITarget_value_layout const> argBlockLayout;

		//for (int id = 0; id < targetCode->get_callable_function_count(); id++) {
		//	const char* fun = targetCode->get_callable_function(id);
		//	//targetCode->get_argument_block(id);
		//	VTX_INFO("MDL WRAPPER: Target code callable function: {} ", fun);
		//}

		if (targetCode->get_argument_block_count() > materialAdditionIndex)
		{
			const auto argumentBlock = make_handle<ITarget_argument_block const>(targetCode->get_argument_block(materialAdditionIndex));
			argumentBlockClone = Handle(argumentBlock->clone());
			argBlockData = argumentBlockClone->get_data();
			argBlockLayout = make_handle<ITarget_value_layout const>(targetCode->get_argument_block_layout(materialAdditionIndex));
		}
		else
		{
			VTX_WARN("MDL WRAPPER: It was impossible to create argument block clone for target code");
		}


		MdlState&                    state = *getState();
		std::vector<graph::shader::ParameterInfo> paramInfos;
		const TransactionInterfaces* tI    = state.getTransactionInterfaces();
		{
			const Handle	compiledMaterial = make_handle(tI->transaction->access<ICompiled_material>(materialDbName.c_str()));
			const Handle	materialDefinition = make_handle(tI->transaction->access<IFunction_definition>(functionDefinitionSignature.c_str()));
			const Handle	annoList = make_handle<const IAnnotation_list>(materialDefinition->get_parameter_annotations());
			const Size		numParams = compiledMaterial->get_parameter_count();

			for (Size j = 0; j < numParams; ++j)
			{
				graph::shader::ParameterInfo paramInfo = generateParamInfo(j, compiledMaterial, argBlockData, argBlockLayout, annoList, mapEnumTypes);

				if((void*)paramInfo.dataPtr != nullptr)
				{
					//VTX_INFO("Material {} Parameter: {} Found!", materialDbName, paramInfo.argumentName);
					paramInfos.push_back(paramInfo);
				}
				else
				{
					VTX_INFO("Material {} Parameter: {} Not available for use!", materialDbName, compiledMaterial->get_parameter_name(j));
				}
			}
		}
		state.commitTransaction();
		//tI->transaction->commit();
		return paramInfos;
	}

	void  loadFromFile(std::shared_ptr<graph::Texture> textureNode)
	{
		MdlState&                       state = *getState();
		const TransactionInterfaces*    tI    = state.getTransactionInterfaces();
		{
			// Load environment texture
			const Handle image(tI->transaction->create<IImage>("Image"));
			const Sint32 result = image->reset_file(textureNode->filePath.c_str());
			VTX_ASSERT_BREAK(result == 0, "Error with creating new Texture image {}", textureNode->filePath);
			const std::string imageDbName = "Image::" + utl::getFileName(textureNode->filePath);
			tI->transaction->store(image.get(), imageDbName.c_str());

			// Create a new texture instance and set its properties
			const Handle texture(tI->transaction->create<ITexture>("Texture"));

			texture->set_image(imageDbName.c_str());

			const std::string textureDbName = "userTexture::" + utl::getFileName(textureNode->filePath);

			tI->transaction->store(texture.get(), textureDbName.c_str());

			const ITarget_code::Texture_shape shape = ITarget_code::Texture_shape::Texture_shape_2d;

			textureNode->databaseName = textureDbName;
			textureNode->shape = shape;
		}
		state.commitTransaction();
	}

	void fetchTextureData(const std::shared_ptr<graph::Texture>& textureNode)
	{

		MdlState&                    state = *getState();
		const TransactionInterfaces* tI    = state.getTransactionInterfaces();
		{

			const auto* mdlState = getState();
			const Handle<IImage_api>& imageApi = mdlState->imageApi;

			const ITarget_code::Texture_shape	shape = textureNode->shape;
			std::string							textureDbName = textureNode->databaseName;

			//The following objects will be created by analyzing the mdl texture information
			size_t& pixelBytesSize = textureNode->pixelBytesSize;
			CUarray_format_enum& format = textureNode->format;
			math::vec4ui& dimension = textureNode->dimension;
			std::vector<const void*>& imageLayersPointers = textureNode->imageLayersPointers;

			// Access texture image and canvas
			const Handle texture = make_handle(tI->transaction->access<ITexture>(textureNode->databaseName.c_str()));
			const Handle image   = make_handle<const IImage>(tI->transaction->access<IImage>(texture->get_image()));
			const char*  url     = image->get_filename(0, 0);
			if(textureNode->filePath.empty())
			{
				if (url != nullptr)
				{
					textureNode->filePath = url;
				}
				else
				{
					textureNode->filePath = textureNode->databaseName;
				}
			}
			Handle       canvas  = make_handle<const ICanvas>(image->get_canvas(0, 0, 0));
			//const Float32						effectiveGamma			= texture->get_effective_gamma(0, 0);

			if (image->is_uvtile() || image->is_animated())
			{
				VTX_ERROR("MDL TEXTURE: uvtile and/or animated textures not supported! Texture name {}", textureDbName);
				return;
			}

			// MDL pixel types.
			//"Sint8"      // Signed 8-bit integer
			//"Sint32"     // Signed 32-bit integer
			//"Float32"    // 32-bit IEEE-754 single-precision floating-point number
			//"Float32<2>" // 2 x Float32
			//"Float32<3>" // 3 x Float32
			//"Float32<4>" // 4 x Float32
			//"Rgb"        // 3 x Uint8 representing RGB color
			//"Rgba"       // 4 x Uint8 representing RGBA color
			//"Rgbe"       // 4 x Uint8 representing RGBE color
			//"Rgbea"      // 5 x Uint8 representing RGBEA color
			//"Rgb_16"     // 3 x Uint16 representing RGB color
			//"Rgba_16"    // 4 x Uint16 representing RGBA color
			//"Rgb_fp"     // 3 x Float32 representing RGB color
			//"Color"      // 4 x Float32 representing RGBA color
			const std::string imageType(image->get_type(0, 0));

			if (imageType == "Rgb")
			{
				canvas = imageApi->convert(canvas.get(), "Rgba"); // Append an alpha channel with 0xFF.
				format = CU_AD_FORMAT_UNSIGNED_INT8;
				pixelBytesSize = sizeof(Uint8);
			}
			else if (imageType == "Rgba")
			{
				format = CU_AD_FORMAT_UNSIGNED_INT8;
				pixelBytesSize = sizeof(Uint8);
			}
			else {
				format = CU_AD_FORMAT_FLOAT;
				pixelBytesSize = sizeof(Float32);

				// Convert image to linear color space if necessary.
				if (texture->get_effective_gamma(0, 0) != 1.0f)
				{
					// Copy/convert to float4 canvas and adjust gamma from "effective gamma" to 1.0.
					const Handle gammaCanvas = make_handle<ICanvas>(imageApi->convert(canvas.get(), "Color"));
					gammaCanvas->set_gamma(texture->get_effective_gamma(0, 0));
					imageApi->adjust_gamma(gammaCanvas.get(), 1.0f);
					canvas = gammaCanvas;
				}
				else if (imageType != "Color" && imageType != "Float32<4>")
				{
					canvas = imageApi->convert(canvas.get(), "Color");
				}
			}
			dimension.x = canvas->get_resolution_x();
			dimension.y = canvas->get_resolution_y();
			// Copy image data to GPU array depending on texture shape
			if (shape == ITarget_code::Texture_shape_cube ||
				shape == ITarget_code::Texture_shape_3d ||
				shape == ITarget_code::Texture_shape_bsdf_data)
			{
				dimension.z = canvas->get_layers_size();
				// Cubemap and 3D texture objects require 3D CUDA arrays.
				VTX_ASSERT_BREAK((shape != ITarget_code::Texture_shape_cube || dimension.z == 6),
								 "ERROR: prepareTextureMDL() Invalid number of layers ({}), cube Maps must have 6 layers!", dimension.z);
			}
			else
			{
				dimension.z = 0;
			}
			dimension.w = 4;
			textureNode->effectiveGamma = texture->get_effective_gamma(0, 0);


			//Fetch pointers to the texture layers data (for non 3d images the number of layers is 0!)
			for (unsigned int z = 0; z < canvas->get_layers_size(); ++z)
			{
				const auto tile = make_handle<const ITile>(canvas->get_tile(z));
				void* dst = malloc(dimension.x * dimension.y * pixelBytesSize * dimension.w);
				memcpy(dst, tile->get_data(), dimension.x * dimension.y * pixelBytesSize * dimension.w);
				// Debug set every pixel to red
				//for(unsigned int i = 0; i < dimension.x * dimension.y * dimension.w; i += 4)
				//{
				//	if(pixelBytesSize == sizeof(Float32))
				//	{
				//		((Float32*)dst)[i] = 1.0f;
				//		((Float32*)dst)[i + 1] = 0.0f;
				//		((Float32*)dst)[i + 2] = 0.0f;
				//		((Float32*)dst)[i + 3] = 1.0f;
				//	}
				//	else if(pixelBytesSize == sizeof(Uint8))
				//	{
				//		((Uint8*)dst)[i] = 255;
				//		((Uint8*)dst)[i + 1] = 0;
				//		((Uint8*)dst)[i + 2] = 0;
				//		((Uint8*)dst)[i + 3] = 255;
				//	}
				//	
				//}
				imageLayersPointers.push_back(dst);
			}
		}
		state.commitTransaction();
		//tI->transaction->commit();
	}

	graph::BsdfMeasurement::BsdfPartData fetchBsdfData(const std::string& bsdfDbName, const Mbsdf_part part)
	{

		MdlState&                            state = *getState();
		const TransactionInterfaces*         tI    = state.getTransactionInterfaces();
		graph::BsdfMeasurement::BsdfPartData data{};
		{
			bool success = true;
			// Get access to the MBSDF data by the texture database name from the target code.
			const Handle bsdfMeasurement = make_handle<const IBsdf_measurement>(tI->transaction->access<IBsdf_measurement>(bsdfDbName.c_str()));

			switch (part)
			{
				case(MBSDF_DATA_REFLECTION):
				{
					if (const Handle reflectionDataset = make_handle(bsdfMeasurement->get_reflection<Bsdf_isotropic_data>()))
					{
						data.angularResolution.x = reflectionDataset->get_resolution_theta();
						data.angularResolution.y = reflectionDataset->get_resolution_phi();
						data.numChannels = (reflectionDataset->get_type() == BSDF_SCALAR) ? 1 : 3;
						// {1, 3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)
						data.srcData = make_handle<const IBsdf_buffer>(reflectionDataset->get_bsdf_buffer())->get_data();
					}
				}
				case(MBSDF_DATA_TRANSMISSION):
				{
					if (const Handle transmissionDataset = make_handle(bsdfMeasurement->get_transmission<Bsdf_isotropic_data>()))
					{
						data.angularResolution.x = transmissionDataset->get_resolution_theta();
						data.angularResolution.y = transmissionDataset->get_resolution_phi();
						data.numChannels = (transmissionDataset->get_type() == BSDF_SCALAR) ? 1 : 3;
						// {1, 3} * (index_theta_in * (res_phi * res_theta) + index_theta_out * res_phi + index_phi)
						data.srcData = make_handle<const IBsdf_buffer>(transmissionDataset->get_bsdf_buffer())->get_data();
					}
				}
			}

		}
		state.commitTransaction();
		//tI->transaction->commit();
		return data;
	}

	graph::LightProfile::LightProfileData fetchLightProfileData(const std::string& lightDbName)
	{
		MdlState&                             state = *getState();
		const TransactionInterfaces*          tI    = state.getTransactionInterfaces();
		graph::LightProfile::LightProfileData data{};
		{
			// Get access to the light_profile data.
			const Handle lightProfile = make_handle<const ILightprofile>(tI->transaction->access<ILightprofile>(lightDbName.c_str()));

			data.resolution.x = lightProfile->get_resolution_theta();
			data.resolution.y = lightProfile->get_resolution_phi();
			data.start.x = lightProfile->get_theta(0);
			data.start.y = lightProfile->get_phi(0);
			data.delta.x = lightProfile->get_theta(1) - data.start.x;
			data.delta.y = lightProfile->get_phi(1) - data.start.y;
			data.sourceData = lightProfile->get_data();
			data.candelaMultiplier = lightProfile->get_candela_multiplier();

		}
		state.commitTransaction();
		//tI->transaction->commit();
		return data;
	}

	void dumpModuleInfo(const IModule* module, IMdl_factory* factory, ITransaction* transaction, const bool dumpDefinitions = false) {
		MdlState&                    state = *getState();
		const TransactionInterfaces* tI    = state.getTransactionInterfaces();

		std::stringstream ss;
		// Print the module name and the file name it was loaded from.
		ss << "Loaded file " << module->get_filename() << "\n";
		ss << "Found module " << module->get_mdl_name() << "\n";
		ss << "\n";

		// Dump imported modules.
		const Size module_count = module->get_import_count();
		if (module_count > 0) {
			ss << "The module imports the following modules:" << "\n";
			for (Size i = 0; i < module_count; i++)
				ss << "    " << module->get_import(i) << "\n";
		}
		else {
			ss << "The module doesn't import other modules." << "\n";
		}

		// Dump exported types.
		const Handle<IType_factory> type_factory(factory->create_type_factory(transaction));
		const Handle<const IType_list> types(module->get_types());
		if (types->get_size() > 0) {
			ss << "\n";
			ss << "The module contains the following types: " << "\n";
			for (Size i = 0; i < types->get_size(); ++i) {
				Handle<const IType> type(types->get_type(i));
				const Handle<const IString> result(type_factory->dump(type.get(), 1));
				ss << "    " << result->get_c_str() << "\n";
			}
		}
		else {
			ss << "The module doesn't contain any types " << "\n";
		}

		// Dump exported constants.
		const Handle<IValue_factory> value_factory(factory->create_value_factory(transaction));
		const Handle<const IValue_list> constants(module->get_constants());
		if (constants->get_size() > 0) {
			ss << "\n";
			ss << "The module contains the following constants: " << "\n";
			for (Size i = 0; i < constants->get_size(); ++i) {
				const char* name = constants->get_name(i);
				Handle<const IValue> constant(constants->get_value(i));
				const Handle<const IString> result(value_factory->dump(constant.get(), 0, 1));
				ss << "    " << name << " = " << result->get_c_str() << "\n";
			}
		}
		else {
			ss << "The module doesn't contain any constants " << "\n";
		}

		// Dump function definitions of the module.
		const Size function_count = module->get_function_count();
		if (function_count > 0) {
			ss << "\n";
			ss << "The module contains the following function definitions:" << "\n";
			for (Size i = 0; i < function_count; i++) {
				ss << "    " << module->get_function(i) << "\n";
				// Dump a function definition from the module.
				if (dumpDefinitions) {
					ss << "Dumping function definition \"" << module->get_function(i) << "\":"
						<< "\n";
					Handle<const IFunction_definition> function_definition(tI->transaction->access<IFunction_definition>(module->get_function(i)));
					ss << dumpDefinition(transaction, factory, function_definition.get(), 1);
				}
			}
		}
		else {
			ss << "The module doesn't contains any function definitions" << "\n";
		}


		// Dump material definitions of the module.
		const Size material_count = module->get_material_count();
		if (material_count > 0) {
			ss << "\n";
			ss << "The module contains the following material definitions:" << "\n";
			for (Size i = 0; i < material_count; i++) {
				ss << "    " << module->get_material(i) << "\n";
				// Dump a material definition from the module.
				if (dumpDefinitions) {
					ss << "Dumping material definition \"" << module->get_material(i) << "\":"
						<< "\n";
					Handle<const IFunction_definition> material_definition(tI->transaction->access<IFunction_definition>(module->get_material(i)));
					ss << dumpDefinition(transaction, factory, material_definition.get(), 1);
				}
			}
		}
		else {
			ss << "The module doesn't contains any material definitions" << "\n";
		}


		// Dump the resources referenced by this module
		if (module->get_resources_count() == 0) {
			ss << "This Module has no resources \n";
		}
		else {
			ss << "Dumping resources of this module: \n";
			for (Size r = 0, rn = module->get_resources_count(); r < rn; ++r)
			{
				const Handle<const IValue_resource> resource(
					module->get_resource(r));
				const char* db_name = resource->get_value();
				const char* mdl_file_path = resource->get_file_path();

				if (db_name == nullptr)
				{
					// resource is either not used and therefore has not been loaded or
					// could not be found.
					ss << "    db_name:               none" << "\n";
					ss << "    mdl_file_path:         " << mdl_file_path << "\n"
						<< "\n";
					continue;
				}
				ss << "    db_name:               " << db_name << "\n";
				ss << "    mdl_file_path:         " << mdl_file_path << "\n";

				const Handle<const IType_resource> type(
					resource->get_type());
				switch (type->get_kind())
				{
				case IType::TK_TEXTURE:
					{
						const Handle<const ITexture> texture(
							tI->transaction->access<ITexture>(db_name));
						if (texture)
						{
							const Handle<const IImage> image(
								tI->transaction->access<IImage>(texture->get_image()));

							for (Size f = 0, fn = image->get_length(); f < fn; ++f)
								for (Size t = 0, tn = image->get_frame_length(f); t < tn; ++t)
								{
									const char* resolved_file_path = image->get_filename(f, t);
									ss << "    resolved_file_path[" << f << "," << t << "]: "
										<< resolved_file_path << "\n";
								}
						}
						break;
					}

				case IType::TK_LIGHT_PROFILE:
					{
						const Handle<const ILightprofile> light_profile(
							tI->transaction->access<ILightprofile>(db_name));
						if (light_profile)
						{
							const char* resolved_file_path = light_profile->get_filename();
							ss << "    resolved_file_path:    " << resolved_file_path << "\n";
						}
						break;
					}

				case IType::TK_BSDF_MEASUREMENT:
					{
						const Handle<const IBsdf_measurement> mbsdf(
							tI->transaction->access<IBsdf_measurement>(db_name));
						if (mbsdf)
						{
							const char* resolved_file_path = mbsdf->get_filename();
							ss << "    resolved_file_path:    " << resolved_file_path << "\n";
						}
						break;
					}

				default:
					break;
				}
			}
			ss << "\n";
		}
		VTX_INFO("Module Info:\n{}", ss.str());
	}

	void getFunctionSignature(MdlFunctionInfo* functionInfo)
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		const Handle<IMdl_execution_context> context = state->context;
		const Handle<IMdl_impexp_api>        impExp = state->impExpApi;
		Sint32                               result;

		if (functionInfo->signature.empty())
		{
			//TODO check if module is already loaded
			result = impExp->load_module(tI->transaction.get(), removeMdlPrefix(functionInfo->module).c_str(), context.get());
			VTX_ASSERT_CONTINUE((result >= 0 && mdl::logMessage(context.get())), state->lastError);


			const Handle<const IString> moduleDatabaseName(state->factory->get_db_module_name(removeMdlPrefix(functionInfo->module).c_str()));
			const Handle<const IModule> module(tI->transaction->access<IModule>(moduleDatabaseName.get()->get_c_str()));
			VTX_ASSERT_BREAK((module.is_valid_interface()), "Error with module access");

			//const Handle<const IModule> module(tI->transaction->access<IModule>(functionInfo->module.c_str()));
			const Handle<const IArray>  overloads(module->get_function_overloads(functionInfo->name.c_str()));
			const Handle<const IString> mdlSignature(overloads->get_element<IString>(0));
			functionInfo->signature = mdlSignature->get_c_str();

		}
		const Handle<const IFunction_definition> functionDefinition(tI->transaction->access<IFunction_definition>(functionInfo->signature.c_str()));
		functionInfo->returnType = functionDefinition->get_return_type()->skip_all_type_aliases();
	}

	std::vector<graph::shader::ParameterInfo> getFunctionParameters(const MdlFunctionInfo& functionInfo, const std::string callingNodeName)
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		const Handle<const IFunction_definition> functionDefinition(tI->transaction->access<IFunction_definition>(functionInfo.signature.c_str()));
		const Handle<const IType_list>           types(functionDefinition->get_parameter_types());
		const Handle<const IAnnotation_list>     anno(functionDefinition->get_parameter_annotations());
		const Handle<const IExpression_list>	defaults(functionDefinition->get_defaults());
		const Size								paramCount = functionDefinition->get_parameter_count();
		std::vector<graph::shader::ParameterInfo>               parameters;

		for (int i = 0; i < paramCount; i++)
		{
			graph::shader::ParameterInfo paramInfo;
			paramInfo.argumentName = types->get_name(i);
			paramInfo.expressionKind                 = types->get_type(i)->skip_all_type_aliases()->get_kind();
			paramInfo.annotation		   = getAnnotation(make_handle(anno->get_annotation_block(anno->get_index(paramInfo.argumentName.c_str()))));
			paramInfo.index                = i;
			parameters.push_back(paramInfo);
		}
		return parameters;
	}

	std::string analizeDirectCallResult(const Sint32 result, std::string signature, Handle<IExpression_list> arguments)
	{
		if (result < 0) {
			std::stringstream ss;
			ss << "Failed Direct Call Creation  of expression \n\t" << signature << "\n\tWith error : ";

			switch (result)
			{
				case -1:
					ss << "\t\tAn argument for a non-existing parameter was provided in arguments.";
					break;
				case -2:
					ss << "\t\tThe type of an argument in arguments does not have the correct type.";
					break;
				case -3:
					ss << "\t\tA parameter that has no default was not provided with an argument value.";
					break;
				case -4:
					ss << "\t\tThe function or material definition can not be instantiated because it is not exported.";
					break;
				case -5:
					ss << "\t\tA parameter type is uniform, but the corresponding argument has a varying return type.";
					break;
				case -6:
					ss << "\t\tAn argument expression is not a constant, a direct call, nor a parameter.";
					break;
				case -7:
					ss << "\t\tInvalid parameters (NULL pointer) or name is not a valid DB name of a function or material definition.";
					break;
				case -8:
					ss << "\t\tOne of the parameter types is uniform, but the corresponding argument or default is a call expression and the return type of the called function or material definition is effectively varying since the function or material definition itself is varying.";
					break;
				case -9:
					ss << "\t\tThe function or material definition is invalid due to a module reload.";
					break;
			}
			for (int i = 0; i < arguments->get_size(); i++)
			{
				if(arguments->get_expression(i)==nullptr)
				{
					ss << "\n\t\tArgument: " << i << " : nullptr";
				}
				else
				{
					const IExpression::Kind kind = arguments->get_expression(i)->get_kind();
					const auto              name = arguments->get_name(i);
					ss << "\n\t\tArgument: " << i << " : name " << name << " : kind" << kind;

				}
			}
			return ss.str();
		}
		else
		{
			return std::string{};
		}
	}

	bool generateFunctionExpression(const std::string& functionSignature, std::map<std::string, graph::shader::ShaderNodeSocket>& sockets, std::string callerNodeName)
	{
		VTX_INFO("Generating function expression for:\n\tNode {}\n\tSignature {}", callerNodeName, functionSignature);
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		ModuleCreationParameters& mcp = state->moduleCreationParameter;
		const Handle<IExpression_factory>& ef = tI->expressionFactory;
		{
			//const Handle<IExpression_list> callArguments(ef->create_expression_list());
			const Handle<IExpression_list> definitionCallArg(ef->create_expression_list());

			const Handle<const IFunction_definition> definition(tI->transaction->access<IFunction_definition>(functionSignature.c_str()));
			Handle<const IExpression_list> defaultsExprList(definition->get_defaults());

			for (auto& [name, socket] : sockets)
			{
				std::shared_ptr<graph::shader::ShaderNode> shaderNodeSocket = socket.node;
				graph::shader::ParameterInfo&              param            = socket.parameterInfo;

				if(shaderNodeSocket)
				{
					Handle<IExpression> socketCall(ef->create_call(shaderNodeSocket->name.c_str()));
					definitionCallArg->add_expression(param.argumentName.c_str(), socketCall.get());
				}
				else if(socket.directExpression.is_valid_interface())
				{
					definitionCallArg->add_expression(param.argumentName.c_str(), socket.directExpression.get());
				}
				
			}

			Sint32 result;
			Handle<IFunction_call>       call(definition->create_function_call(definitionCallArg.get(), &result));
			result = tI->transaction->store(call.get(), callerNodeName.c_str());

			Handle<const IFunction_call> function_call(tI->transaction->access<mi::neuraylib::IFunction_call>(callerNodeName.c_str()));
			//dumpInstance(ef.get(), function_call.get(), callerNodeName);

			if(result==0)
			{
				return true;
			}
			return false;
			
		}
	}

	IType::Kind getExpressionKind(const std::string& dbExpressionName)
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		return tI->transaction->access<IExpression>(dbExpressionName.c_str())->get_type()->skip_all_type_aliases()->get_kind();
	}

	Handle<IExpression> createConstantColor(const math::vec3f& color)
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		const Handle<IValue_factory>& vf = tI->valueFactory;
		const Handle<IExpression_factory>& ef = tI->expressionFactory;
		const Handle<IValue>               colorValue(vf->create_color(color.x, color.y, color.z));
		const Handle<IExpression>          colorExpr(ef->create_constant(colorValue.get()));
		return colorExpr;
	}

	Handle<IExpression> createConstantFloat(const float value)
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		const Handle<IValue_factory>& vf = tI->valueFactory;
		const Handle<IExpression_factory>& ef = tI->expressionFactory;
		const Handle<IValue>               floatValue(vf->create_float(value));
		const Handle<IExpression>          floatExpr(ef->create_constant(floatValue.get()));
		return floatExpr;
	}

	Handle<IExpression> createConstantBool(const bool value)
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		const Handle<IValue_factory>& vf = tI->valueFactory;
		const Handle<IExpression_factory>& ef = tI->expressionFactory;
		const Handle<IValue>				boolValue(vf->create_bool(value));
		const Handle<IExpression>			boolExpr(ef->create_constant(boolValue.get()));

		return boolExpr;
	}

	Handle<IExpression> createConstantInt(const int value)
	{
		MdlState* state = getState();
		const TransactionInterfaces*		tI = state->getTransactionInterfaces();
		const Handle<IValue_factory>&		vf = tI->valueFactory;
		const Handle<IExpression_factory>&	ef = tI->expressionFactory;
		const Handle<IValue>				intValue(vf->create_int(value));
		const Handle<IExpression>			intExpr(ef->create_constant(intValue.get()));

		return intExpr;
	}

	Handle<IExpression> createTextureConstant(const std::string& texturePath, const IType_texture::Shape shape, const float gamma)
	{
		try
		{
			MdlState* state = getState();
			const TransactionInterfaces* tI = state->getTransactionInterfaces();

			const std::string textureFolder = utl::getFolder(texturePath);
			const std::string textureName = "/" + utl::getFile(texturePath);
			addSearchPath(textureFolder);

			const Handle<IMdl_factory>& factory = state->factory;
			const Handle<IExpression_factory>& ef = tI->expressionFactory;

			const Handle<IValue_texture>   argValue(factory->create_texture(tI->transaction.get(), textureName.c_str(), shape, gamma, nullptr, true, nullptr));
			VTX_INFO("Texture {} loaded with gamma : {} ", textureName, argValue->get_gamma());
			const Handle<IExpression>      argExpr(ef->create_constant(argValue.get()));

			return argExpr;
		}
		catch (const std::exception& e)
		{
			VTX_WARN("createTextureConstant : {}", e.what());
			return Handle<IExpression>();
		}
		catch (...)
		{
			VTX_WARN("createTextureConstant : unknown exception");
			return Handle<IExpression>();
		}
		
	};


	std::string getTexturePathFromExpr(const Handle<IExpression>& expr)
	{
		const IExpression::Kind            kind = expr->get_kind();
		if(kind != IExpression::EK_CONSTANT)
		{
			VTX_WARN("getTexturePathFromExpr : expr is not a constant");
			return "";
		}
		const Handle<IExpression_constant> argConstant(expr->get_interface<IExpression_constant>());
		const Handle<IValue>               value(argConstant->get_value());
		const IValue::Kind                 valueKind = value->get_kind();
		if (valueKind != IValue::VK_TEXTURE)
		{
			VTX_WARN("getTexturePathFromExpr : value is not a texture");
			return "";
		}
		const Handle<IValue_texture>       argValueTexture(value->get_interface<IValue_texture>());

		const char*                  texturePathFromExpr = argValueTexture->get_file_path();
		if (texturePathFromExpr == nullptr)
		{
			// return the DB Name
			texturePathFromExpr = argValueTexture->get_value();
		}
		if (texturePathFromExpr == nullptr)
		{
			// return the DB Name
			return "";
		}
		return texturePathFromExpr;
	}

	void createShaderGraphFunctionCalls(std::shared_ptr<graph::shader::ShaderNode> shaderGraph)
	{
		//MdlState* state = getState();
		//const TransactionInterfaces* tI = state->getTransactionInterfaces();
		//ModuleCreationParameters& mcp = state->moduleCreationParameter;
		//mcp.reset();
		ShaderVisitor visitor;
		shaderGraph->traverse(visitor);
	}

	void ModuleCreationParameters::reset()
	{
		MdlState* state = getState();
		const TransactionInterfaces* tI = state->getTransactionInterfaces();
		const Handle<IExpression_factory> ef = tI->expressionFactory;
		const Handle<IType_factory>       tf = tI->typeFactory;
		const Handle<IMdl_factory>        factory = state->factory;
		const Handle<IMdl_impexp_api>     impExp = state->impExpApi;
		parameters = tf->create_type_list();
		defaults = ef->create_expression_list();
		parameterAnnotations = ef->create_annotation_list();
		annotations = ef->create_annotation_block();
		returnAnnotations = ef->create_annotation_block();
		paramCount = 0;
	}


}



